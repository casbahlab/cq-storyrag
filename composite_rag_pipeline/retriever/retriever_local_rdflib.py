#!/usr/bin/env python3
# retriever_local_rdflib.py
"""
Local retriever (rdflib) with:
- simple literal bindings ([Key] and {Key})
- robust diagnostics + per-CQ logs + JSONL stream
- per-row URL detection and optional metadata + content enrichment
- timeout per SPARQL query
- (NEW) optional meta lookup for SPARQL by CQ id
- (NEW) Hybrid KG enrichment: labels/descriptions + 1-hop neighbors
- (NEW) optional evidence JSONL compatible with generator_dual.py

CLI (examples)
--------------
# Basic: execute plan SPARQL against local KG, write combined JSON + evidence JSONL
python retriever_local_rdflib.py \
  --plan plan.json \
  --rdf kg/liveaid_instances_master.ttl kg/schema/liveaid_schema.ttl \
  --bindings bindings.json \
  --require_sparql \
  --per_item_sample 5 \
  --timeout_s 10 \
  --log_dir run_trace/logs \
  --errors_jsonl run_trace/retriever.jsonl \
  --evidence_out evidence_KG.jsonl \
  --out plan_with_evidence.json

# With URL enrichment (titles/status) + content text extraction
python retriever_local_rdflib.py \
  --plan plan.json \
  --rdf kg/liveaid_instances_master.ttl kg/schema/liveaid_schema.ttl \
  --bindings bindings.json \
  --require_sparql \
  --per_item_sample 5 \
  --timeout_s 10 \
  --enrich_urls \
  --fetch_url_content \
  --url_timeout_s 5 \
  --max_urls_per_item 5 \
  --log_dir run_trace/logs \
  --errors_jsonl run_trace/retriever.jsonl \
  --evidence_out evidence_Hybrid.jsonl \
  --out plan_with_evidence.json

# Using meta to resolve SPARQL by CQ id + Hybrid KG enrichment (labels + neighbors)
python retriever_local_rdflib.py \
  --mode Hybrid \
  --plan plan_Hybrid.json \
  --meta ../index/Hybrid/cq_metadata.json \
  --rdf ../kg/data.ttl ../kg/schema/liveaid_schema.ttl \
  --bindings params.json \
  --hy_enrich_labels --hy_enrich_neighbors 8 --hy_enrich_incoming 4 \
  --evidence_out evidence_Hybrid.jsonl \
  --out plan_with_evidence.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from rdflib import Graph, ConjunctiveGraph, URIRef, Literal, BNode, Namespace, term

# Optional deps for URL fetching/parsing
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


# =============================================================================
# Simple literal bindings
# =============================================================================

def simple_replace(template: str, bindings: dict) -> str:
    """
    Literal text replacement:
      - Replaces "[Key]" and "{Key}" with bindings[Key]
      - Case-sensitive keys; values inserted verbatim (include <> / quotes yourself)
      - Replaces longer keys first to avoid 'Venue' vs 'Venue1' collisions
    """
    if not isinstance(bindings, dict):
        raise TypeError(f"bindings must be a dict, got {type(bindings).__name__}")
    out = template or ""
    for k in sorted(bindings.keys(), key=lambda x: len(str(x)), reverse=True):
        v = str(bindings[k])
        key = str(k)
        out = out.replace(f"[{key}]", v).replace(f"{{{key}}}", v)
    return out


# Placeholder finder (for diagnostics)
TOKEN_FINDER_RX = re.compile(r"(\[\s*([A-Za-z0-9_:-]+)\s*\]|\{\s*([A-Za-z0-9_:-]+)\s*\})")


# =============================================================================
# SPARQL helpers
# =============================================================================

def _ensure_limit(q: str, n: int) -> str:
    return q if re.search(r"\blimit\s+\d+\b", q, flags=re.I) else (q.rstrip() + f"\nLIMIT {n}")

def _n3(x: term.Node) -> str:
    try:
        return x.n3()
    except Exception:
        return str(x)

def _plain(x: term.Node) -> str:
    if isinstance(x, URIRef): return str(x)
    if isinstance(x, Literal): return str(x)
    if isinstance(x, BNode): return f"_:{x}"
    return str(x)

def _rows_sample_dual(res, max_rows: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Return (rows_plain, rows_n3)"""
    rows_p: List[Dict[str, str]] = []
    rows_n: List[Dict[str, str]] = []
    for b in getattr(res, "bindings", [])[:max_rows]:
        rows_p.append({k: _plain(v) for k, v in b.items()})
        rows_n.append({k: _n3(v)    for k, v in b.items()})
    return rows_p, rows_n

def _classify_error(msg: str) -> str:
    m = (msg or "").lower()
    if "timeout" in m:
        return "timeout"
    if "prefix" in m and ("unbound" in m or "undefined" in m or "not bound" in m):
        return "missing_prefix"
    if "bad iri" in m or "not a valid uri" in m or "illegal iri" in m:
        return "bad_iri"
    if "unbound" in m and "variable" in m:
        return "unbound_variable"
    if "expected" in m and "found" in m and ("line" in m or "col" in m):
        return "syntax"
    if "parse" in m or "syntax" in m:
        return "syntax"
    if "name lookup" in m or "unknown" in m:
        return "unknown_term"
    return "error"

def _execute_with_timeout(graph: Graph, query: str, timeout_s: Optional[float]):
    if timeout_s and timeout_s > 0:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(graph.query, query)
            return fut.result(timeout=timeout_s)
    return graph.query(query)


# =============================================================================
# Logging helpers
# =============================================================================

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "NA")

def _write_per_cq_files(seq: int, cid: str, query: str, result_obj: Dict[str, Any], log_dir: Optional[Path]):
    if not log_dir:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{seq:03d}_{_sanitize(cid)}_query.rq").write_text(query or "", encoding="utf-8")
    (log_dir / f"{seq:03d}_{_sanitize(cid)}_result.json").write_text(
        json.dumps(result_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _append_jsonl(rec: Dict[str, Any], jsonl_path: Optional[Path]):
    if not jsonl_path:
        return
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =============================================================================
# URL extraction, enrichment, and content
# =============================================================================

_URL_CANDIDATE_RE = re.compile(r"\bhttps?://[^\s<>\)\"']+", re.I)

def _strip_quotes_angle(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1]
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1]
    return t.strip()

def _dedupe_keep_order(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _extract_urls_from_value(v: Any) -> List[str]:
    t = _strip_quotes_angle(v)
    found: List[str] = []
    if t.lower().startswith("http://") or t.lower().startswith("https://"):
        found.append(t)
    else:
        for m in _URL_CANDIDATE_RE.finditer(t):
            found.append(m.group(0))
    norm = []
    for u in found:
        u2 = u.strip().rstrip(").,;\"'")
        if u2:
            norm.append(u2)
    return _dedupe_keep_order(norm)

def _url_domain(u: str) -> str:
    try:
        return urlparse(u).netloc or ""
    except Exception:
        return ""

def _clean_html_get_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;?", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _extract_main_text_html(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        return _clean_html_get_text(html)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    selectors = [
        "article", "main", "[role='main']",
        ".article__content", ".article-content", ".story-body",
        ".entry-content", ".post-content", "#content", "#main",
    ]
    candidates = []
    for sel in selectors:
        candidates.extend(soup.select(sel))
    if not candidates:
        candidates = soup.find_all(["div", "section", "article"], limit=30)
    best_text = ""
    best_len = 0
    for node in candidates:
        t = node.get_text(" ", strip=True)
        if len(t) > best_len:
            best_len = len(t)
            best_text = t
    if best_text:
        return re.sub(r"\s+", " ", best_text).strip()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()

def _fetch_url_meta(
    u: str,
    timeout_s: float,
    *,
    with_content: bool = True,
    max_bytes: int = 250_000,
    max_chars: int = 5000,
    user_agent: str = "WembleyRewind/1.0 (+research; polite)",
) -> Dict[str, Any]:
    """
    Return: {
      url, domain, status, content_type, title?, fetched_at,
      content_text?, content_len?, content_sha1?, truncated?, error?
    }
    """
    info: Dict[str, Any] = {
        "url": u,
        "domain": _url_domain(u),
        "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if requests is None:
        info["error"] = "requests_not_installed"
        return info

    headers = {"User-Agent": user_agent, "Accept": "text/html, text/plain;q=0.8, */*;q=0.5"}
    try:
        # HEAD (best-effort)
        status = None
        content_type = ""
        try:
            h = requests.head(u, timeout=timeout_s, allow_redirects=True, headers=headers)
            status = h.status_code
            content_type = (h.headers.get("Content-Type") or "").lower()
        except Exception:
            pass

        need_get = ("html" in content_type) or ("text" in content_type) or (not content_type)
        if need_get:
            r = requests.get(u, timeout=timeout_s, headers=headers, stream=True)
            status = r.status_code
            content_type = (r.headers.get("Content-Type") or content_type or "").lower()

            # read up to max_bytes
            body = b""
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    body += chunk
                    if len(body) >= max_bytes:
                        break

            enc = r.encoding or getattr(r, "apparent_encoding", None) or "utf-8"
            try:
                html = body.decode(enc, errors="replace")
            except Exception:
                html = body.decode("utf-8", errors="replace")

            # title
            title = None
            if BeautifulSoup is not None:
                soup = BeautifulSoup(html, "html.parser")
                og = soup.find("meta", property="og:title")
                if og and og.get("content"):
                    title = og.get("content").strip()
                if not title and soup.title and soup.title.string:
                    title = soup.title.string.strip()
            else:
                m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
                if m:
                    title = re.sub(r"\s+", " ", m.group(1)).strip()
            if title:
                info["title"] = title

            # content extraction
            if with_content:
                if "html" in content_type:
                    text = _extract_main_text_html(html)
                elif "text/plain" in content_type or (not content_type and html and "<" not in html[:200]):
                    text = _clean_html_get_text(html)
                else:
                    text = ""
                text = text[:max_chars].strip()
                info["content_text"] = text
                info["content_len"] = len(text)
                info["truncated"] = len(body) >= max_bytes or len(text) >= max_chars
                if text:
                    info["content_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()

        info["status"] = status
        info["content_type"] = content_type
        return info

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info

def _attach_url_info_to_rows(
    rows: List[Dict[str, Any]],
    *,
    url_timeout_s: float,
    max_urls_per_item: int,
    with_content: bool = False,
    content_max_bytes: int = 250_000,
    content_max_chars: int = 5000,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Mutates each row:
      - __url_candidates: [str]
      - __url_info: [{...}] (if enrich=True)
    Returns (item_level_candidates, item_level_info)
    """
    agg_candidates: List[str] = []
    agg_info: List[Dict[str, Any]] = []

    for r in rows or []:
        row_urls: List[str] = []
        for _, v in r.items():
            row_urls.extend(_extract_urls_from_value(v))
        row_urls = _dedupe_keep_order(row_urls)[:max_urls_per_item]
        r["__url_candidates"] = row_urls

        if row_urls:
            infos = [
                _fetch_url_meta(
                    u,
                    timeout_s=url_timeout_s,
                    with_content=with_content,
                    max_bytes=content_max_bytes,
                    max_chars=content_max_chars,
                )
                for u in row_urls
            ]
            r["__url_info"] = infos
            agg_info.extend(infos)

        agg_candidates.extend(row_urls)

    agg_candidates = _dedupe_keep_order(agg_candidates)[:max_urls_per_item]
    # dedupe info by url
    seen_u, uniq_info = set(), []
    for info in agg_info:
        u = info.get("url")
        if u and u not in seen_u:
            seen_u.add(u)
            uniq_info.append(info)

    return agg_candidates, uniq_info


# =============================================================================
# KG enrichment (labels + 1-hop neighbors) for Hybrid mode
# =============================================================================

RDFS    = Namespace("http://www.w3.org/2000/01/rdf-schema#")
SKOS    = Namespace("http://www.w3.org/2004/02/skos/core#")
SCHEMA  = Namespace("http://schema.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

LABEL_PROPS = [RDFS.label, SKOS.prefLabel, SCHEMA.name]
DESC_PROPS  = [RDFS.comment, SCHEMA.description, DCTERMS.description]

def _first_label(g: Graph, node) -> Optional[str]:
    if isinstance(node, URIRef):
        for lp in LABEL_PROPS:
            for lo in g.objects(node, lp):
                if isinstance(lo, Literal) and (lo.language in (None, "", "en") or lo.language.lower().startswith("en")):
                    return str(lo)
        for lp in LABEL_PROPS:
            for lo in g.objects(node, lp):
                if isinstance(lo, Literal):
                    return str(lo)
    return None

def fetch_labels_local(graph: Graph, uris: Iterable[str]) -> Dict[str, Dict[str, Optional[str]]]:
    results: Dict[str, Dict[str, Optional[str]]] = {}
    for u in uris:
        uri = URIRef(u)
        label = None
        desc  = None
        for p in LABEL_PROPS:
            for o in graph.objects(uri, p):
                if isinstance(o, Literal) and (o.language in (None,"","en") or o.language.lower().startswith("en")):
                    label = str(o); break
            if label: break
        if not label:
            for p in LABEL_PROPS:
                for o in graph.objects(uri, p):
                    if isinstance(o, Literal):
                        label = str(o); break
                if label: break
        for p in DESC_PROPS:
            for o in graph.objects(uri, p):
                if isinstance(o, Literal) and (o.language in (None,"","en") or o.language.lower().startswith("en")):
                    desc = str(o); break
            if desc: break
        results[u] = {"label": label, "desc": desc}
    return results

def fetch_neighbors_local(graph: Graph, uri: str, max_out: int, max_in: int) -> List[Dict[str, Any]]:
    uref = URIRef(uri)
    triples: List[Tuple[str, str, Optional[str], str, Optional[str], str]] = []

    out_count = 0
    for p, o in graph.predicate_objects(uref):
        if not isinstance(o, (URIRef, BNode)):
            continue
        triples.append(("out", str(p), _first_label(graph, p), _plain(o), _first_label(graph, o) if isinstance(o, URIRef) else None, ""))
        out_count += 1
        if out_count >= max_out:
            break

    in_count = 0
    for s, p in graph.subject_predicates(uref):
        if not isinstance(s, (URIRef, BNode)):
            continue
        triples.append(("in", str(p), _first_label(graph, p), _plain(s), _first_label(graph, s) if isinstance(s, URIRef) else None, ""))
        in_count += 1
        if in_count >= max_in:
            break

    out = []
    for direction, puri, plabel, nuri, nlabel, _ in triples:
        out.append({
            "direction": direction,
            "predicate": {"uri": puri, "label": plabel},
            "node": {"uri": nuri, "label": nlabel}
        })
    return out


# =============================================================================
# Core API
# =============================================================================

def guess_format(path: Path) -> Optional[str]:
    s = path.suffix.lower()
    return {
        ".ttl":"turtle", ".nt":"nt", ".nq":"nquads",
        ".rdf":"xml", ".owl":"xml", ".xml":"xml",
        ".trig":"trig", ".jsonld":"json-ld"
    }.get(s, None)

def _load_graph(files: List[str]) -> Graph:
    g: Graph = ConjunctiveGraph()
    for f in files:
        fp = Path(f)
        fmt = guess_format(fp) or ("turtle" if fp.suffix.lower()==".ttl" else None)
        g.parse(str(fp), format=fmt)
    return g

def run(
    plan: Dict[str, Any],
    rdf_files: List[str],
    bindings: Dict[str, Any] | None = None,
    per_item_sample: int = 3,
    require_sparql: bool = False,
    timeout_s: Optional[float] = None,
    log_dir: Optional[str] = None,
    errors_jsonl: Optional[str] = None,
    include_stack: bool = True,
    include_executed_query: bool = True,
    strict_bindings: bool = False,
    execute_on_unbound: bool = False,
    # URL enrichment controls
    enrich_urls: bool = False,
    fetch_url_content: bool = False,
    url_timeout_s: float = 5.0,
    max_urls_per_item: int = 5,
    content_max_bytes: int = 250_000,
    content_max_chars: int = 5000,
    # Hybrid KG enrichment controls
    mode: str = "KG",
    hy_enrich_labels: bool = False,
    hy_enrich_neighbors: int = 0,
    hy_enrich_incoming: int = 0,
    # Meta (optional): use to resolve SPARQL by CQ id if plan items don't carry it
    meta_rows: Dict[str, Any] | None = None,
    evidence_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Execute plan items' SPARQL against local RDF data and (optionally) enrich URLs and KG entities.
    """
    # Load KG
    g = _load_graph(rdf_files)

    log_path_dir = Path(log_dir) if log_dir else None
    jsonl_path = Path(errors_jsonl) if errors_jsonl else None
    ev_fp = open(evidence_out, "w", encoding="utf-8") if evidence_out else None

    enriched: List[Dict[str, Any]] = []
    seq = 0

    for it in plan.get("items", []):
        seq += 1
        cid = it.get("id") or f"item_{seq}"
        beat = it.get("beat") or it.get("beat_title") or "Unspecified"
        beat_obj = it.get("beat") if isinstance(it.get("beat"), dict) else {
            "index": it.get("beat_index"),
            "title": it.get("beat_title") or (beat if isinstance(beat, str) else "Beat")
        }
        # get SPARQL: prefer plan, else meta
        sparql_tpl = (it.get("sparql") or (meta_rows or {}).get(cid, {}).get("sparql") or "")
        sparql_src = (meta_rows or {}).get(cid, {}).get("sparql_source")

        t0 = time.perf_counter()

        executed_query = ""
        kg_ok = False
        kg_reason = "empty"
        kg_error_class = None
        kg_error_message = None
        row_count = 0
        rows_plain: List[Dict[str, str]] = []
        rows_n3: List[Dict[str, str]] = []
        stack_txt = None
        url_candidates: List[str] = []
        url_info: List[Dict[str, Any]] = []
        enrichment: Dict[str, Any] = {}

        # binding diagnostics
        found_tokens = [m.group(2) or m.group(3) for m in TOKEN_FINDER_RX.finditer(sparql_tpl or "")]
        leftovers_after = []

        # No SPARQL
        if not sparql_tpl and require_sparql:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            rec = {
                "cq_id": cid, "beat": beat, "status": "no_sparql",
                "row_count": 0, "elapsed_ms": elapsed_ms,
                "executed_query": executed_query,
                "error_class": None, "error_message": None,
                "binding_report": {"found": found_tokens, "leftovers": []},
                "url_candidates": [], "url_info": [], "rows": [],
            }
            enriched.append({
                **it, "kg_ok": False, "kg_reason": "no_sparql",
                "kg_error_class": None, "kg_error_message": None,
                "row_count": 0, "rows": [],
                "elapsed_ms": elapsed_ms,
                "executed_query": executed_query if include_executed_query else "",
                "binding_report": {"found": found_tokens, "leftovers": []},
                "url_candidates": [], "url_info": [],
            })
            _write_per_cq_files(seq, cid, executed_query, rec, log_path_dir)
            _append_jsonl(rec, jsonl_path)
            if ev_fp:
                ev_fp.write(json.dumps({
                    "id": cid, "mode": mode, "beat": beat_obj, "question": it.get("question",""),
                    "sparql": executed_query, "sparql_source": sparql_src or "plan",
                    "bindings": [], "error": "no_sparql"
                }, ensure_ascii=False) + "\n")
            continue

        try:
            # Bind placeholders (literal)
            executed_query = simple_replace(sparql_tpl, bindings or {})
            executed_query = _ensure_limit(executed_query, per_item_sample)

            # Leftover tokens?
            leftovers_after = [m.group(0) for m in TOKEN_FINDER_RX.finditer(executed_query)]
            if leftovers_after:
                if strict_bindings:
                    raise ValueError(f"Unbound placeholders: {sorted(set(leftovers_after))}")
                elif not execute_on_unbound:
                    raise RuntimeError(f"Unbound placeholders (non-strict, skipped execution): {sorted(set(leftovers_after))}")

            # Execute
            res = _execute_with_timeout(g, executed_query, timeout_s) if executed_query else None
            if res is not None:
                # rows for human debugging (n3) and generator (plain)
                rows_plain, rows_n3 = _rows_sample_dual(res, per_item_sample)
                row_count = len(res.bindings)
            kg_ok = (row_count > 0) or (not require_sparql and not sparql_tpl)
            kg_reason = "ok" if kg_ok else "empty"

            # URL enrichment (per-row) + aggregates
            if enrich_urls and rows_plain:
                url_candidates, url_info = _attach_url_info_to_rows(
                    rows_plain,
                    url_timeout_s=url_timeout_s,
                    max_urls_per_item=max_urls_per_item,
                    with_content=fetch_url_content,
                    content_max_bytes=content_max_bytes,
                    content_max_chars=content_max_chars,
                )

            # Hybrid KG enrichment
            if mode == "Hybrid" and rows_plain:
                # collect URIs from rows
                uris: Set[str] = set()
                for r in rows_plain:
                    for v in r.values():
                        if isinstance(v, str) and v.startswith("http"):
                            uris.add(v)
                if uris:
                    if hy_enrich_labels:
                        labels = fetch_labels_local(g, uris)
                        if labels:
                            enrichment["entities"] = [{"uri": u, "label": d.get("label"), "desc": d.get("desc")} for u,d in labels.items()]
                    if hy_enrich_neighbors > 0 or hy_enrich_incoming > 0:
                        cards = []
                        for u in list(uris)[:16]:
                            nbs = fetch_neighbors_local(g, u, hy_enrich_neighbors, hy_enrich_incoming)
                            if nbs:
                                cards.append({"uri": u, "neighbors": nbs})
                        if cards:
                            enrichment["neighbors"] = cards

        except Exception as e:
            kg_ok = False
            kg_error_class = type(e).__name__
            kg_error_message = str(e)
            kg_reason = _classify_error(kg_error_message)
            if include_stack:
                stack_txt = traceback.format_exc()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Output (combined JSON item)
        out_item = {
            **it,
            "kg_ok": kg_ok,
            "kg_reason": kg_reason,
            "kg_error_class": kg_error_class,
            "kg_error_message": kg_error_message,
            "row_count": row_count,
            "rows": rows_plain,        # plain values (URIs detectable)
            "rows_n3": rows_n3,        # n3 for inspection (quoted/typed)
            "elapsed_ms": elapsed_ms,
            "binding_report": {"found": found_tokens, "leftovers": leftovers_after},
            "url_candidates": url_candidates,
            "url_info": url_info,
            "executed_query": executed_query if include_executed_query else "",
            "sparql_source": sparql_src or "plan",
        }
        if enrichment:
            out_item["enrichment"] = enrichment
        enriched.append(out_item)

        # Logs (include rows so row↔URL mapping is visible)
        rec = {
            "cq_id": cid,
            "beat": beat,
            "status": "ok" if kg_ok else kg_reason,
            "row_count": row_count,
            "rows": rows_plain,
            "elapsed_ms": elapsed_ms,
            "executed_query": executed_query,
            "error_class": kg_error_class,
            "error_message": kg_error_message,
            "stack": stack_txt,
            "binding_report": {"found": found_tokens, "leftovers": leftovers_after},
            "url_candidates": url_candidates,
            "url_info": url_info,
        }
        _write_per_cq_files(seq, cid, executed_query, rec, log_path_dir)
        _append_jsonl(rec, jsonl_path)

        # Evidence JSONL (for generator_dual.py)
        if ev_fp:
            ev_line = {
                "id": cid,
                "mode": mode,
                "beat": beat_obj,
                "question": it.get("question",""),
                "sparql": executed_query,
                "sparql_source": sparql_src or "plan",
                "bindings": rows_plain,
            }
            if enrichment:
                ev_line["enrichment"] = enrichment
            if not kg_ok:
                ev_line["error"] = kg_reason
            ev_fp.write(json.dumps(ev_line, ensure_ascii=False) + "\n")

    if ev_fp:
        ev_fp.close()

    # Stats + combined output JSON
    out = {
        **plan,
        "items": enriched,
        "retriever_stats": {
            "checked": len(plan.get("items", [])),
            "kept": sum(1 for r in enriched if r.get("kg_ok")),
            "dropped": sum(1 for r in enriched if not r.get("kg_ok")),
            "timeout_s": timeout_s,
            "require_sparql": require_sparql,
            "strict_bindings": strict_bindings,
            "execute_on_unbound": execute_on_unbound,
            "fetch_url_content": fetch_url_content,
            "max_urls_per_item": max_urls_per_item,
            "mode": mode,
            "hy_enrich_labels": hy_enrich_labels,
            "hy_enrich_neighbors": hy_enrich_neighbors,
            "hy_enrich_incoming": hy_enrich_incoming,
        },
    }
    return out


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Local retriever (rdflib) with bindings, logging, per-row URL enrichment, and Hybrid KG enrichment.")
    ap.add_argument("--mode", default="KG", choices=["KG","Hybrid"], help="Affects KG enrichment (Hybrid adds labels/neighbors).")
    ap.add_argument("--plan", required=True, help="Planner output JSON (items with 'sparql' and/or 'id').")
    ap.add_argument("--meta", default=None, help="Optional cq_metadata.json; used to locate SPARQL by CQ id.")
    ap.add_argument("--rdf", nargs="+", required=True, help="RDF files to load locally (ttl/nt/n3/rdf/xml/jsonld/trig).")
    ap.add_argument("--bindings", default=None, help="JSON file with placeholder bindings (include <> for IRIs).")
    ap.add_argument("--require_sparql", action="store_true", help="Mark items w/o SPARQL as failures.")
    ap.add_argument("--per_item_sample", type=int, default=3, help="Sample rows per item; used as LIMIT if missing.")
    ap.add_argument("--timeout_s", type=float, default=None, help="Per-query timeout in seconds (soft).")
    ap.add_argument("--strict_bindings", action="store_true", help="Fail items with any unbound placeholders.")
    ap.add_argument("--execute_on_unbound", action="store_true", help="(Non-strict) still execute with leftovers.")

    ap.add_argument("--log_dir", default=None, help="Directory for per-CQ query/result logs.")
    ap.add_argument("--errors_jsonl", default=None, help="Append one JSON object per CQ to this JSONL file.")

    # URL enrichment flags
    ap.add_argument("--enrich_urls", action="store_true", help="Fetch URL metadata for URLs found in rows.")
    ap.add_argument("--fetch_url_content", action="store_true", help="Also fetch and extract readable text content.")
    ap.add_argument("--url_timeout_s", type=float, default=5.0)
    ap.add_argument("--max_urls_per_item", type=int, default=5)
    ap.add_argument("--content_max_bytes", type=int, default=250000)
    ap.add_argument("--content_max_chars", type=int, default=5000)

    # Hybrid KG enrichment
    ap.add_argument("--hy_enrich_labels", action="store_true", help="Attach labels/descriptions for entity URIs (Hybrid).")
    ap.add_argument("--hy_enrich_neighbors", type=int, default=0, help="Outgoing neighbor sample size (Hybrid).")
    ap.add_argument("--hy_enrich_incoming", type=int, default=0, help="Incoming neighbor sample size (Hybrid).")

    # Evidence JSONL for generator_dual.py
    ap.add_argument("--evidence_out", default=None, help="If set, write generator-ready evidence JSONL here.")

    ap.add_argument("--out", default="plan_with_evidence.json", help="Output path for enriched plan JSON.")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    bindings = json.loads(Path(args.bindings).read_text(encoding="utf-8")) if args.bindings else {}
    meta_rows = None
    if args.meta:
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        meta_rows = meta.get("metadata") or {}

    out = run(
        plan=plan,
        rdf_files=args.rdf,
        bindings=bindings,
        per_item_sample=args.per_item_sample,
        require_sparql=args.require_sparql,
        timeout_s=args.timeout_s,
        log_dir=args.log_dir,
        errors_jsonl=args.errors_jsonl,
        include_stack=True,
        include_executed_query=True,
        strict_bindings=args.strict_bindings,
        execute_on_unbound=args.execute_on_unbound,
        enrich_urls=args.enrich_urls,
        fetch_url_content=args.fetch_url_content,
        url_timeout_s=args.url_timeout_s,
        max_urls_per_item=args.max_urls_per_item,
        content_max_bytes=args.content_max_bytes,
        content_max_chars=args.content_max_chars,
        mode=args.mode,
        hy_enrich_labels=args.hy_enrich_labels,
        hy_enrich_neighbors=args.hy_enrich_neighbors,
        hy_enrich_incoming=args.hy_enrich_incoming,
        meta_rows=meta_rows,
        evidence_out=Path(args.evidence_out) if args.evidence_out else None,
    )

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"retriever → {args.out} | kept={out['retriever_stats']['kept']} dropped={out['retriever_stats']['dropped']}")
    if args.enrich_urls and requests is None:
        print("ℹ️ URL enrichment requested but 'requests' is not installed; only extracted candidates were recorded.")

if __name__ == "__main__":
    main()
