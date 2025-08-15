#!/usr/bin/env python3
# retriever_local_rdflib.py
"""
Local retriever (rdflib) with:
- simple literal bindings ([Key] and {Key})
- robust diagnostics + per-CQ logs + JSONL stream
- per-row URL detection and optional metadata + content enrichment
- timeout per SPARQL query

CLI (examples)
--------------
# Basic: execute plan SPARQL against local KG
python retriever_local_rdflib.py \
  --plan plan.json \
  --rdf kg/liveaid_instances_master.ttl kg/schema/liveaid_schema.ttl \
  --bindings bindings.json \
  --require_sparql \
  --per_item_sample 5 \
  --timeout_s 10 \
  --log_dir run_trace/logs \
  --errors_jsonl run_trace/retriever.jsonl \
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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from rdflib import Graph, term

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

def _rows_sample(res, max_rows: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for b in getattr(res, "bindings", [])[:max_rows]:
        out.append({k: _n3(v) for k, v in b.items()})
    return out

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

    print(f"_fetch_url_meta")

    headers = {"User-Agent": user_agent, "Accept": "text/html, text/plain;q=0.8, */*;q=0.5"}
    try:
        # HEAD (best-effort)
        try:
            h = requests.head(u, timeout=timeout_s, allow_redirects=True, headers=headers)
            info["status"] = h.status_code
            info["content_type"] = h.headers.get("Content-Type", "")
        except Exception:
            h = None

        content_type = h.headers.get("Content-Type", "").lower()
        print(f"content_type :{content_type} ")
        need_get = True
        if "html" not in content_type and "text" not in content_type:
            need_get = False

        if need_get:
            r = requests.get(u, timeout=timeout_s, headers=headers, stream=True)
            print(f"_fetch_url_meta r : {r}")
            info["status"] = r.status_code
            info["content_type"] = r.headers.get("Content-Type", info.get("content_type", ""))

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

            print(f"with_content : {with_content}")
            # content extraction
            if with_content:
                print(f"_extract_main_text_html with_content")
                ct = (info.get("content_type") or "").lower()
                if "html" in ct:
                    text = _extract_main_text_html(html)
                elif "text/plain" in ct or (not ct and html and "<" not in html[:200]):
                    text = _clean_html_get_text(html)
                else:
                    text = ""

                text = text[:max_chars].strip()
                print(f"_extract_main_text_html with_content text : {text} ")
                info["content_text"] = text
                info["content_len"] = len(text)
                info["truncated"] = len(body) >= max_bytes or len(text) >= max_chars
                if text:
                    info["content_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()

        print(f"info : {info}")
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

        print(f"row_urls : {row_urls}")

        if row_urls:
            infos = [
                _fetch_url_meta(
                    u,
                    timeout_s=url_timeout_s,
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
# Core API
# =============================================================================

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
    fetch_url_content: bool = False,
    url_timeout_s: float = 5.0,
    max_urls_per_item: int = 5,
    content_max_bytes: int = 250_000,
    content_max_chars: int = 5000,
) -> Dict[str, Any]:
    """
    Execute plan items' SPARQL against local RDF data and enrich URL facts (per row).
    """
    # Load KG
    g = Graph()
    for f in rdf_files:
        g.parse(f)

    log_path_dir = Path(log_dir) if log_dir else None
    jsonl_path = Path(errors_jsonl) if errors_jsonl else None

    enriched: List[Dict[str, Any]] = []
    seq = 0

    for it in plan.get("items", []):
        seq += 1
        cid = it.get("id") or f"item_{seq}"
        beat = it.get("beat", "Unspecified")
        sparql_tpl = it.get("sparql") or ""
        t0 = time.perf_counter()

        executed_query = ""
        kg_ok = False
        kg_reason = "empty"
        kg_error_class = None
        kg_error_message = None
        row_count = 0
        rows: List[Dict[str, str]] = []
        stack_txt = None
        url_candidates: List[str] = []
        url_info: List[Dict[str, Any]] = []

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
            row_count = len(res.bindings) if res is not None else 0
            rows = _rows_sample(res, per_item_sample) if res is not None else []
            kg_ok = (row_count > 0) or (not require_sparql and not sparql_tpl)
            kg_reason = "ok" if kg_ok else "empty"

            # URL enrichment (per-row) + aggregates
            url_candidates, url_info = _attach_url_info_to_rows(
                rows,
                url_timeout_s=url_timeout_s,
                max_urls_per_item=max_urls_per_item,
                with_content=fetch_url_content,
                content_max_bytes=content_max_bytes,
                content_max_chars=content_max_chars,
            )

        except Exception as e:
            kg_ok = False
            kg_error_class = type(e).__name__
            kg_error_message = str(e)
            kg_reason = _classify_error(kg_error_message)
            if include_stack:
                stack_txt = traceback.format_exc()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Output item
        out_item = {
            **it,
            "kg_ok": kg_ok,
            "kg_reason": kg_reason,
            "kg_error_class": kg_error_class,
            "kg_error_message": kg_error_message,
            "row_count": row_count,
            "rows": rows,  # rows carry __url_candidates/__url_info
            "elapsed_ms": elapsed_ms,
            "binding_report": {"found": found_tokens, "leftovers": leftovers_after},
            "url_candidates": url_candidates,
            "url_info": url_info,
        }
        if include_executed_query:
            out_item["executed_query"] = executed_query
        enriched.append(out_item)

        # Logs (include rows so row↔URL mapping is visible)
        rec = {
            "cq_id": cid,
            "beat": beat,
            "status": "ok" if kg_ok else kg_reason,
            "row_count": row_count,
            "rows": rows,
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

    # Stats
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
        },
    }
    return out


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Local retriever (rdflib) with bindings, logging, and per-row URL enrichment.")
    ap.add_argument("--plan", required=True, help="Planner output JSON (items with 'sparql').")
    ap.add_argument("--rdf", nargs="+", required=True, help="RDF files to load locally (ttl/nt/n3/rdf/xml).")
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

    ap.add_argument("--out", default="plan_with_evidence.json", help="Output path for enriched plan JSON.")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    bindings = json.loads(Path(args.bindings).read_text(encoding="utf-8")) if args.bindings else {}

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
    )

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"retriever → {args.out} | kept={out['retriever_stats']['kept']} dropped={out['retriever_stats']['dropped']}")
    if args.enrich_urls and requests is None:
        print("ℹ️ URL enrichment requested but 'requests' is not installed; only extracted candidates were recorded.")

if __name__ == "__main__":
    main()
