#!/usr/bin/env python3
# retriever_local_rdflib.py
"""
Local retriever (rdflib) with:
- simple literal bindings ([Key] and {Key})
- robust diagnostics + per-CQ logs + JSONL stream
- per-row URL detection and optional metadata + content enrichment
- timeout per SPARQL query
- WIDE URL SCAN: scans more rows than the small sample to find URL candidates
- OPTIONAL CHUNKING: split fetched URL content into chunks for downstream LLMs
- MAX CHUNKS CAP:
    * per-URL cap:        --max_chunks_per_url
    * per-item total cap: --max_url_chunks_total_per_item

CLI example
-----------
python retriever_local_rdflib.py \
  --plan plan.json \
  --rdf kg/liveaid_instances_master.ttl \
  --bindings bindings.json \
  --require_sparql \
  --per_item_sample 5 \
  --timeout_s 10 \
  --enrich_urls --fetch_url_content \
  --url_timeout_s 5 \
  --max_urls_per_item 5 \
  --url_scan_rows 50 \
  --chunk_url_content \
  --chunk_chars 800 --chunk_overlap 120 \
  --max_chunks_per_url 6 \
  --max_url_chunks_total_per_item 18 \
  --log_dir run_trace/logs \
  --errors_jsonl run_trace/retriever.jsonl \
  --out plan_with_evidence.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import wikitextparser as wtp
import traceback

from rdflib import Graph, term


USE_URL_INDEX_CACHE = True        # <- flip to True to enable
CACHE_ONLY = False                 # <- optional: True = no network fallback
CACHE_PREFER = "summary"           # or "text"
URL_INDEX_DB = "web_index/output/url_index.db"  # <- set this

try:
    # Put url_index_retriever.py next to this file (or on PYTHONPATH)
    from retriever.url_index_retriever import UrlIndexRetriever
except Exception:
    UrlIndexRetriever = None

_URL_INDEX_INSTANCE = None

def _get_url_index_instance():
    global _URL_INDEX_INSTANCE
    if _URL_INDEX_INSTANCE is None and USE_URL_INDEX_CACHE and UrlIndexRetriever and URL_INDEX_DB:
        try:
            print(f"[INFO] Initializing URL index from {URL_INDEX_DB}")
            _URL_INDEX_INSTANCE = UrlIndexRetriever(URL_INDEX_DB)
            print(f"[INFO] Initializing URL index from {URL_INDEX_DB}")
        except Exception as e:
            print(f"[cache] init failed: {e}")
            traceback.print_exc()
            _URL_INDEX_INSTANCE = None
    return _URL_INDEX_INSTANCE

try:
    # if it's inside your package
    from wikifetch import content_under_h2, content_under_h3
except ImportError:
    # if it's a top-level file
    from retriever.wikifetch import content_under_h2, content_under_h3


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

def _extract_urls_from_value(k:Any, v: Any) -> List[str]:
    if str(k) in ("url", "sourceUrl"):
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
    else:
        return []

# def _extract_urls_from_value(kv: Tuple[str, Any]) -> List[str]:
#     k, v = kv
#     # only process when key looks like a URL field
#     if k not in ("url", "sourceUrl"):
#         return []
#
#     t = _strip_quotes_angle(v)
#     found: List[str] = []
#     if t.lower().startswith("http://") or t.lower().startswith("https://"):
#         found.append(t)
#     else:
#         for m in _URL_CANDIDATE_RE.finditer(t):
#             found.append(m.group(0))
#
#     norm = []
#     for u in found:
#         u2 = u.strip().rstrip(").,;\"'")
#         if u2:
#             norm.append(u2)
#
#     return _dedupe_keep_order(norm)


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

def safe_parse(body):
    if body is None:
        return None
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8", errors="ignore")
        except Exception:
            body = body.decode("latin-1", errors="ignore")
    elif not isinstance(body, str):
        # If it's some dict/obj, convert to string
        body = str(body)
    return wtp.parse(body)

def _orig_fetch_url_meta(
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
        try:
            h = requests.head(u, timeout=timeout_s, allow_redirects=True, headers=headers)
            info["status"] = h.status_code
            info["content_type"] = h.headers.get("Content-Type", "")
        except Exception:
            h = None

        content_type = (h.headers.get("Content-Type", "") if h is not None else "").lower()
        need_get = True
        if "html" not in content_type and "text" not in content_type:
            need_get = False

        if need_get:
            r = requests.get(u, timeout=timeout_s, headers=headers, stream=True)
            info["status"] = r.status_code
            info["content_type"] = r.headers.get("Content-Type", info.get("content_type", ""))

            # read up to max_bytes
            body = b""
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    body += chunk
                    # if len(body) >= max_bytes:
                    #     break


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
                ct = (info.get("content_type") or "").lower()
                if "html" in ct:
                    text = _extract_main_text_html(html)
                elif "text/plain" in ct or (not ct and html and "<" not in html[:200]):
                    text = _clean_html_get_text(html)
                else:
                    text = ""
                text = text[:max_chars].strip()
                info["content_text"] = text
                info["content_len"] = len(text)
                info["truncated"] = len(body) >= max_bytes or len(text) >= max_chars
                if text:
                    info["content_sha1"] = hashlib.sha1(text.encode("utf-8")).hexdigest()

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
    Returns (item_level_candidates, item_level_info) for the SAMPLE rows
    """

    global html_text, fragment
    from urllib.parse import urlparse, urlunparse

    import re
    from typing import Optional, List

    _HEADING_OPEN_RE = re.compile(r"<h([2-6])\b[^>]*>", re.IGNORECASE)
    _HEADING_SAMELEVEL_TPL = r"<h{lvl}\b[^>]*>"

    def _norm_anchor(s: str) -> str:
        # Decode URL and HTML entities, then MediaWiki-ish normalization
        s = _html.unescape(unquote(s or "")).strip()
        s = s.replace(" ", "_")
        # Wikipedia IDs are case-sensitive-ish; be tolerant by matching literally
        return s

    def _find_heading_range(html: str, anchor: str, start_pos: int, level_hint: int) -> Optional[tuple]:
        """
        Find the heading (hN) whose id/span id equals `anchor`, starting at start_pos.
        Returns (heading_level, content_start_index, next_same_level_index_or_len).
        """
        # We try three patterns:
        # 1) <hN ... id="anchor" ...> ... </hN>
        h_tag_with_id = re.compile(
            rf'<h([2-6])\b[^>]*\bid="{re.escape(anchor)}"[^>]*>(?P<head>.*?)</h\1\s*>',
            re.IGNORECASE | re.DOTALL
        )

        # 2) <hN ...> ... <span class="mw-headline" id="anchor"> ... </span> ... </hN>
        h_tag_with_span_id = re.compile(
            rf'<h([2-6])\b[^>]*>(?P<head>.*?<span\b[^>]*\bid="{re.escape(anchor)}"[^>]*>.*?</span>.*?)</h\1\s*>',
            re.IGNORECASE | re.DOTALL
        )

        # 3) Raw span (rare outside of heading, but be defensive)
        span_only = re.compile(
            rf'<span\b[^>]*\bmw-headline\b[^>]*\bid="{re.escape(anchor)}"[^>]*>.*?</span>',
            re.IGNORECASE | re.DOTALL
        )

        # Search in order of likelihood
        for pat in (h_tag_with_id, h_tag_with_span_id):
            m = pat.search(html, pos=start_pos)
            if m:
                lvl = int(m.group(1))
                # content starts right after the closing </hN> we matched
                content_start = m.end()
                # find next same-level heading from content_start
                same_level_re = re.compile(_HEADING_SAMELEVEL_TPL.format(lvl=lvl), re.IGNORECASE)
                m2 = same_level_re.search(html, pos=content_start)
                content_end = m2.start() if m2 else len(html)
                return (lvl, content_start, content_end)

        # Fallback: span outside an explicit <hN> wrapper; approximate by finding
        # the next same-level heading using the hint
        m = span_only.search(html, pos=start_pos)
        if m:
            lvl = max(2, min(6, level_hint))  # constrain to 2..6
            content_start = m.end()
            same_level_re = re.compile(_HEADING_SAMELEVEL_TPL.format(lvl=lvl), re.IGNORECASE)
            m2 = same_level_re.search(html, pos=content_start)
            content_end = m2.start() if m2 else len(html)
            return (lvl, content_start, content_end)

        return None

    def _extract_block_by_path(html: str, path: List[str]) -> Optional[str]:
        """
        Resolve a path like ["Performances", "Wembley, London"].
        Top-level assumed H2; children descend one level each (H3, H4, ...).
        """
        cur_start = 0
        cur_level = 2  # start at H2 for the first segment
        block_start = None
        block_end = None

        for i, segment in enumerate(path):
            anchor = _norm_anchor(segment)
            found = _find_heading_range(html, anchor, cur_start, cur_level)
            if not found:
                return None
            lvl, s, e = found
            # For nested segments, advance inside the matched block and descend one level.
            cur_start = s
            block_start, block_end = s, e
            cur_level = min(lvl + 1, 6)

        if block_start is None:
            return None
        return html[block_start:block_end].strip()

    def _split_selector(selector: str) -> List[str]:
        # Support "A>B>C" with optional whitespace around '>'
        return [p.strip() for p in (selector or "").split(">") if p.strip()]

    import re
    import html as _html
    from typing import Optional, List, Tuple
    from urllib.parse import unquote

    try:
        from bs4 import BeautifulSoup, Tag
    except Exception:  # pragma: no cover
        BeautifulSoup = None
        Tag = None

    _HEADING_NAMES = ["h2", "h3", "h4", "h5", "h6"]

    def _canon_text(s: str) -> str:
        """Canonicalize heading text for loose matching (case/space/entity/url tolerances)."""
        s = _html.unescape(unquote((s or "").strip()))
        s = s.replace("\xa0", " ")
        s = re.sub(r"\s+", " ", s)
        return s.casefold()

    def _id_candidates(segment: str) -> List[str]:
        """
        Generate plausible MediaWiki anchor id variants for a segment.
        MediaWiki typically uses underscores in ids; punctuation usually remains literal.
        """
        raw = _html.unescape(unquote((segment or "").strip()))
        unders = raw.replace(" ", "_")
        return [raw, unders]

    def _collect_until_next_heading(start_heading: Tag, max_level_for_stop: int) -> str:
        """
        Collect HTML after start_heading until we hit a heading with level <= max_level_for_stop.
        """
        parts: List[str] = []
        for sib in start_heading.next_siblings:
            if isinstance(sib, Tag) and sib.name in _HEADING_NAMES:
                lvl = int(sib.name[1])
                if lvl <= max_level_for_stop:
                    break
            parts.append(str(sib))
        return "".join(parts).strip()

    def _find_heading_tag(soup: "BeautifulSoup", segment: str, min_level: int = 2) -> Optional[Tuple[Tag, int]]:
        """
        Find the <hN> tag corresponding to a segment.
        Priority: match by id (on hN or span.mw-headline) using candidate variants,
        then fall back to matching by visible text.
        Returns (heading_tag, level).
        """
        # 1) Try id-based matches
        for cand in _id_candidates(segment):
            for h in _HEADING_NAMES:
                # <hN id="cand">
                tag = soup.find(h, id=cand)
                if tag:
                    return tag, int(h[1])
                # <hN> ... <span class="mw-headline" id="cand"> ... </span> ... </hN>
                tag = soup.select_one(f'{h} span.mw-headline[id="{cand}"]')
                if tag:
                    htag = tag.find_parent(h)
                    if htag:
                        return htag, int(h[1])

        # 2) Fall back to text match on the visible headline
        want = _canon_text(segment)
        for h in _HEADING_NAMES:
            for tag in soup.find_all(h):
                # Prefer mw-headline text when present
                span = tag.find("span", class_=lambda c: c and "mw-headline" in c.split())
                text = span.get_text(" ", strip=True) if span else tag.get_text(" ", strip=True)
                if _canon_text(text) == want:
                    return tag, int(h[1])

        return None

    def _extract_by_path_dom(html: str, selector_path: List[str]) -> Optional[str]:
        if not BeautifulSoup:
            return None  # will fall back to regex path below
        soup = BeautifulSoup(html, "html.parser")

        current_container = soup
        current_level = 2  # H2 for first segment
        for idx, segment in enumerate(selector_path):
            # Search within the current container only
            local_soup = BeautifulSoup(str(current_container), "html.parser")
            found = _find_heading_tag(local_soup, segment, min_level=current_level)
            if not found:
                return None
            heading_tag, level = found

            # Collect content until next heading of level <= current level
            block_html = _collect_until_next_heading(heading_tag, level)
            if idx == len(selector_path) - 1:
                return block_html

            # Descend for the next segment
            current_container = BeautifulSoup(block_html, "html.parser")
            current_level = min(level + 1, 6)

        return None

    # ---------------------------
    # Fallback (regex, last resort)
    # ---------------------------

    def _extract_by_path_regex(html: str, selector_path: List[str]) -> Optional[str]:
        """
        Simpler fallback: only supports a single segment; tries to grab content after an <hN> with matching id/text.
        """
        if not selector_path:
            return None
        seg = re.escape(selector_path[-1])
        # Try mw-headline id first
        pat = re.compile(
            rf'<h([2-6])\b[^>]*>.*?<span[^>]*\bmw-headline\b[^>]*\bid="{seg}"[^>]*>.*?</span>.*?</h\1>(?P<body>.*?)(?=<h[2-6]\b|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        m = pat.search(html)
        if m:
            return m.group("body").strip()

        # Fallback: match heading by visible text (very rough)
        pat2 = re.compile(
            rf'<h([2-6])\b[^>]*>[^<]*{seg}[^<]*</h\1>(?P<body>.*?)(?=<h[2-6]\b|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        m = pat2.search(html)
        return m.group("body").strip() if m else None

    # ---------------------------
    # Public API
    # ---------------------------

    def _extract_section_from_wikipedia_html(html: str, section_id: str) -> Optional[str]:
        """
        Extract section/subsection content from a Wikipedia page.
        section_id can be:
          - "Background"
          - "Performances>Wembley, London"
        Returns inner HTML from the matched heading until the next heading of level <= that heading.
        """
        if not html or not section_id:
        # Split on '>' for nested selectors; ignore empty parts
            path = [p.strip() for p in section_id.split(">") if p.strip()]
            out = _extract_by_path_dom(html, path)
        if out is not None:
            return out
        # Fallback to regex (single segment only)
        return _extract_by_path_regex(html, path)

    agg_candidates: List[str] = []
    agg_info: List[Dict[str, Any]] = []

    for r in rows or []:
        row_urls: List[str] = []

        row_urls = []
        if "sourceUrl" in r:
            sourceUrl = r["sourceUrl"]
            exactSelector = r["exactSelector"]
            refinedBy = r["refinedBy"]

            row_urls.extend(sourceUrl + "#" + exactSelector + "$" + refinedBy)

        else:
            for k, v in r.items():
                row_urls.extend(_extract_urls_from_value(k,v))
            row_urls = _dedupe_keep_order(row_urls)[:max_urls_per_item]
        r["__url_candidates"] = row_urls

        if row_urls:
            infos: List[Dict[str, Any]] = []
            for u in row_urls:
                parsed = urlparse(u)
                base_u = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))  # strip fragment/query
                fragment = parsed.fragment or ""
                fragment, sep, subfragment = fragment.partition("$")


                info = []
                # If this is a Wikipedia link with a fragment, try to extract that section only.
                if with_content:
                    if "wikipedia.org" in base_u:
                        cached = _lookup_url_in_index(u, with_content=with_content)
                        if cached is None:
                            if subfragment :
                                info = {
                                    "url": u,
                                    "domain": _url_domain(u),
                                    "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                }
                                section_html = "\n\n".join(content_under_h3(base_u, fragment,
                                                 subfragment, content="table",
                                                 parse_tables=True))
                                if section_html:
                                    info["content_text"] = section_html
                                else:
                                    info["__section_extracted"] = False
                            else :
                                info = {
                                    "url": u,
                                    "domain": _url_domain(u),
                                    "fetched_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                }
                                section_html = "\n\n".join(content_under_h2(base_u, fragment, content="p"))
                                if section_html:
                                    info["content_text"] = section_html
                                else:
                                    info["__section_extracted"] = False

                    else:
                        info = _fetch_url_meta(
                            base_u,
                            timeout_s=url_timeout_s,
                            max_bytes=content_max_bytes,
                            max_chars=content_max_chars,
                            with_content=with_content,
                        ) or {}

                        # Attach original and base URLs, plus fragment if present
                        info.setdefault("url", u)
                        info["__base_url"] = base_u
                        if fragment:
                            info["__section_id"] = fragment

                infos.append(info)

            r["__url_info"] = infos
            agg_info.extend(infos)

        agg_candidates.extend(row_urls)

    agg_candidates = _dedupe_keep_order(agg_candidates)[:max_urls_per_item]

    # dedupe info by url (use original full URL with fragment so different sections don't collapse)
    seen_u, uniq_info = set(), []
    for info in agg_info:
        u = info.get("url")
        if u and u not in seen_u:
            seen_u.add(u)
            uniq_info.append(info)

    return agg_candidates, uniq_info



from typing import Any, Dict, List, Tuple, Optional
import json
import re

# Reuse your existing deduper
# def _dedupe_keep_order(xs: List[str]) -> List[str]: ...

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")

def _parse_url_with_fragment(u: str) -> Tuple[str, Optional[str]]:
    """
    Split a URL into (sourceUrl, exactSelector) by '#' if present.
    Leaves the fragment as-is (no decoding); downstream can interpret '>' hierarchy if desired.
    """
    if "#" in u:
        base, frag = u.split("#", 1)
        return base, frag or None
    return u, None

def _find_urls_in_text(val: Any) -> List[str]:
    """Return all http(s) URLs found in val (string-ish)."""
    if val is None:
        return []
    s = str(val).strip()
    return _URL_RE.findall(s)

def _canon_candidate(source_url: str,
                     exact_selector: Optional[str] = None,
                     refined_by: Optional[Any] = None) -> str:
    """
    Build a JSON-encoded candidate string so the extraction code can json.loads() it.
    """
    payload: Dict[str, Any] = {"sourceUrl": source_url}
    if exact_selector:
        payload["exactSelector"] = exact_selector
    if refined_by not in (None, "", []):
        payload["refinedBy"] = refined_by
    return json.dumps(payload, ensure_ascii=False)

def _row_to_candidates(r: Dict[str, Any], max_urls: int) -> List[str]:
    """
    Build candidates from a single row using (url | sourceUrl, exactSelector, refinedBy).
    Priority:
      1) If 'url' present, treat it as the main candidate source (may include #fragment).
      2) Else, use 'sourceUrl' (+ optional 'exactSelector'/'refinedBy').
    If 'url'/'sourceUrl' fields contain multiple URLs, emit multiple candidates (up to max_urls).
    """
    out: List[str] = []

    # Collect possible fields (case-sensitive per your note; adjust to .lower() if needed)
    url_field         = r.get("url")
    source_url_field  = r.get("sourceUrl")
    exact_selector    = r.get("exactSelector")
    refined_by        = r.get("refinedBy")

    # 1) Prefer 'url'
    urls = _find_urls_in_text(url_field)
    if urls:
        for u in urls:
            base, frag = _parse_url_with_fragment(u)
            # If row also gives explicit exactSelector/refinedBy, they override the fragment
            exs = str(exact_selector).strip() if exact_selector not in (None, "") else frag
            out.append(_canon_candidate(base, exs, refined_by))
            if len(out) >= max_urls:
                return out
        return out

    # 2) Fall back to 'sourceUrl'
    srcs = _find_urls_in_text(source_url_field)
    if srcs:
        # If sourceUrl itself has a #fragment, split that; explicit exactSelector wins.
        for s in srcs:
            base, frag = _parse_url_with_fragment(s)
            exs = str(exact_selector).strip() if exact_selector not in (None, "") else frag
            out.append(_canon_candidate(base, exs, refined_by))
            if len(out) >= max_urls:
                return out

    return out

def _collect_url_candidates_only(rows: List[Dict[str, Any]], max_urls: int) -> List[str]:
    """
    Scan rows and return up to max_urls unique, JSON-encoded candidates.
    Each candidate has at least {"sourceUrl": "..."} and may include "exactSelector"/"refinedBy".
    Only considers keys: 'url', 'sourceUrl', 'exactSelector', 'refinedBy'.
    """
    agg: List[str] = []
    for r in rows or []:
        agg.extend(_row_to_candidates(r, max_urls))
        if len(agg) >= max_urls:
            break
    return _dedupe_keep_order(agg)[:max_urls]



def _chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunk_chars = max(1, int(chunk_chars))
    overlap = max(0, int(overlap))
    step = max(1, chunk_chars - overlap)
    out = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i+chunk_chars])
        if i + chunk_chars >= n:
            break
        i += step
    return out


def _maybe_chunk_info(
    info: Dict[str, Any],
    enable: bool,
    chunk_chars: int,
    overlap: int,
    max_chunks: Optional[int],
) -> None:
    if not enable:
        return
    txt = info.get("content_text") or ""
    if not txt:
        return
    chunks = _chunk_text(txt, chunk_chars, overlap)
    if isinstance(max_chunks, int) and max_chunks >= 0:
        chunks = chunks[:max_chunks]
    info["content_chunks"] = chunks


def _enforce_total_chunk_cap(url_infos: List[Dict[str, Any]], max_total: Optional[int]) -> Tuple[int, int]:
    """
    Cap total number of chunks across all URLs (preserving URL order then chunk order).
    Returns (kept, dropped).
    """
    if not isinstance(max_total, int) or max_total < 0:
        return (sum(len(info.get("content_chunks") or []) for info in url_infos), 0)

    kept = 0
    dropped = 0
    for info in url_infos:
        chunks = info.get("content_chunks") or []
        if not chunks:
            continue
        if kept >= max_total:
            dropped += len(chunks)
            info["content_chunks"] = []
            continue
        room = max_total - kept
        if len(chunks) > room:
            dropped_here = len(chunks) - room
            info["content_chunks"] = chunks[:room]
            kept += room
            dropped += dropped_here
        else:
            kept += len(chunks)
    return kept, dropped


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
    log_dir: Optional[Path] = None,
    errors_jsonl: Optional[Path] = None,
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
    url_scan_rows: int = 50,
    # chunking controls
    chunk_url_content: bool = False,
    chunk_chars: int = 800,
    chunk_overlap: int = 120,
    max_chunks_per_url: int = 6,
    max_url_chunks_total_per_item: int = 18,
) -> Dict[str, Any]:
    """
    Execute plan items' SPARQL against local RDF data and enrich URL facts.
    - Keeps a small SAMPLE (per_item_sample) for logs
    - Scans up to url_scan_rows rows to find url_candidates for the item
    - Optionally chunks fetched URL content into fixed-size segments
      * capped per-url and capped across the item
    """
    # Load KG
    g = Graph()
    for f in rdf_files:
        g.parse(f)

    log_path_dir = Path(log_dir) if log_dir else None
    jsonl_path = Path(errors_jsonl) if errors_jsonl else None

    enriched: List[Dict[str, Any]] = []
    seq = 0

    # run-level totals
    total_chunks_kept = 0
    total_chunks_dropped = 0

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
        rows_all: List[Dict[str, str]] = []
        stack_txt = None
        url_candidates: List[str] = []
        url_info: List[Dict[str, Any]] = []
        item_chunks_kept = 0
        item_chunks_dropped = 0

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
                "url_chunks_total": 0,
                "url_chunks_dropped": 0,
            })
            _write_per_cq_files(seq, cid, executed_query, rec, log_path_dir)
            _append_jsonl(rec, jsonl_path)
            continue

        try:
            # Bind placeholders (literal)
            executed_query = simple_replace(sparql_tpl, bindings or {})
            executed_query = _ensure_limit(executed_query, max(per_item_sample, url_scan_rows))

            # Leftover tokens?
            leftovers_after = [m.group(0) for m in TOKEN_FINDER_RX.finditer(executed_query)]
            if leftovers_after:
                if strict_bindings:
                    raise ValueError(f"Unbound placeholders: {sorted(set(leftovers_after))}")
                elif not execute_on_unbound:
                    raise RuntimeError(f"Unbound placeholders (non-strict, skipped execution): {sorted(set(leftovers_after))}")

            # Execute
            res = _execute_with_timeout(g, executed_query, timeout_s) if executed_query else None
            row_count = len(getattr(res, "bindings", [])) if res is not None else 0

            # Build rows_all (stringified)
            if res is not None:
                for bnd in res.bindings:
                    rows_all.append({k: _n3(v) for k, v in bnd.items()})

            # SAMPLE for logs
            rows = rows_all[:per_item_sample]
            kg_ok = (row_count > 0) or (not require_sparql and not sparql_tpl)
            kg_reason = "ok" if kg_ok else "empty"

            # URL enrichment
            if enrich_urls:
                # 1) Per-row enrichment for the SAMPLE (kept in output & logs)
                url_candidates_sample, url_info_sample = _attach_url_info_to_rows(
                    rows,
                    url_timeout_s=url_timeout_s,
                    max_urls_per_item=max_urls_per_item,
                    with_content=fetch_url_content,
                    content_max_bytes=content_max_bytes,
                    content_max_chars=content_max_chars,
                )

                # CHUNK the sample infos (if enabled)
                if chunk_url_content:
                    for info in url_info_sample:
                        _maybe_chunk_info(info, True, chunk_chars, chunk_overlap, max_chunks_per_url)

                # 2) Item-level candidates discovered from a wider scan (first N rows)
                scan = rows_all[:max(0, url_scan_rows)]
                scan_urls = _collect_url_candidates_only(scan, max_urls_per_item)

                # Avoid re-fetching URLs already fetched from the sample
                already = {info.get("url") for info in url_info_sample if isinstance(info, dict)}
                new_infos = []
                for u in scan_urls:
                    print(f"Scanning URL candidate: {u}")
                    if u in already:
                        continue
                    meta = _fetch_url_meta(
                        u,
                        timeout_s=url_timeout_s,
                        max_bytes=content_max_bytes,
                        max_chars=content_max_chars,
                        with_content=fetch_url_content,
                    )
                    if chunk_url_content:
                        _maybe_chunk_info(meta, True, chunk_chars, chunk_overlap, max_chunks_per_url)
                    new_infos.append(meta)

                # Merge (preserve sample order first)
                url_candidates = _dedupe_keep_order(list(url_candidates_sample) + scan_urls)[:max_urls_per_item]
                seen = set()
                url_info = []
                for info in list(url_info_sample) + new_infos:
                    u = info.get("url")
                    if u and u not in seen:
                        seen.add(u)
                        url_info.append(info)

                # Enforce per-item total chunk cap (after merge)
                if chunk_url_content:
                    item_chunks_kept, item_chunks_dropped = _enforce_total_chunk_cap(
                        url_info, max_url_chunks_total_per_item
                    )
                    total_chunks_kept += item_chunks_kept
                    total_chunks_dropped += item_chunks_dropped
            else:
                url_candidates, url_info = [], []

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
            "rows": rows,  # SAMPLE rows (each may carry __url_candidates/__url_info)
            "elapsed_ms": elapsed_ms,
            "binding_report": {"found": found_tokens, "leftovers": leftovers_after},
            "url_candidates": url_candidates,
            "url_info": url_info,
            "url_chunks_total": item_chunks_kept,
            "url_chunks_dropped": item_chunks_dropped,
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
            "url_chunks_total": item_chunks_kept,
            "url_chunks_dropped": item_chunks_dropped,
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
            "url_scan_rows": url_scan_rows,
            "chunk_url_content": chunk_url_content,
            "chunk_chars": chunk_chars,
            "chunk_overlap": chunk_overlap,
            "max_chunks_per_url": max_chunks_per_url,
            "max_url_chunks_total_per_item": max_url_chunks_total_per_item,
            "total_url_chunks_kept": total_chunks_kept,
            "total_url_chunks_dropped": total_chunks_dropped,
        },
    }
    return out


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Local retriever (rdflib) with bindings, logging, URL enrichment, and chunking.")
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
    ap.add_argument("--enrich_urls", action="store_true", help="Extract and fetch URL metadata for http(s) in result rows.")
    ap.add_argument("--fetch_url_content", action="store_true", help="Also fetch and extract readable text content.")
    ap.add_argument("--url_timeout_s", type=float, default=5.0)
    ap.add_argument("--max_urls_per_item", type=int, default=5)
    ap.add_argument("--content_max_bytes", type=int, default=250000)
    ap.add_argument("--content_max_chars", type=int, default=5000)
    ap.add_argument("--url_scan_rows", type=int, default=50, help="Rows to scan for URL candidates (beyond the sample).")

    # chunking flags
    ap.add_argument("--chunk_url_content", action="store_true", help="Split fetched URL content into fixed-size chunks.")
    ap.add_argument("--chunk_chars", type=int, default=800, help="Chunk size in characters.")
    ap.add_argument("--chunk_overlap", type=int, default=120, help="Overlap between adjacent chunks (chars).")
    ap.add_argument("--max_chunks_per_url", type=int, default=6, help="Cap on number of chunks per URL (use 0 to include none).")
    ap.add_argument("--max_url_chunks_total_per_item", type=int, default=18, help="Cap on total chunks across all URLs in an item.")

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
        errors_jsonl=Path(args.errors_jsonl) if args.errors_jsonl else None,
        include_stack=True,
        include_executed_query=True,
        strict_bindings=args.strict_bindings,
        execute_on_unbound=args.execute_on_unbound,
        # URL enrichment controls
        enrich_urls=args.enrich_urls,
        fetch_url_content=args.fetch_url_content,
        url_timeout_s=args.url_timeout_s,
        max_urls_per_item=args.max_urls_per_item,
        content_max_bytes=args.content_max_bytes,
        content_max_chars=args.content_max_chars,
        url_scan_rows=args.url_scan_rows,
        # chunking controls
        chunk_url_content=args.chunk_url_content,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        max_chunks_per_url=args.max_chunks_per_url,
        max_url_chunks_total_per_item=args.max_url_chunks_total_per_item,
    )

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"retriever → {args.out} | "
        f"kept={out['retriever_stats']['kept']} dropped={out['retriever_stats']['dropped']} | "
        f"chunks kept={out['retriever_stats']['total_url_chunks_kept']} "
        f"dropped={out['retriever_stats']['total_url_chunks_dropped']}"
    )
    if args.enrich_urls and requests is None:
        print("ℹ️ URL enrichment requested but 'requests' is not installed; only extracted candidates were recorded.")


from typing import Optional, Dict, Any

def _lookup_url_in_index(
    u: str,
    *,
    with_content: bool,
    prefer: str = None,
) -> Optional[Dict[str, Any]]:
    """
    Try to fetch URL content from the local index.
    Returns a normalized 'cached' response dict on hit, or None on miss/error.
    """
    if not USE_URL_INDEX_CACHE:
        return None

    print(f"Checking cache for {u} ...")
    idx = _get_url_index_instance()
    print(f"Checking idx {idx} ...")

    if idx is None:
        return None

    print(f"Looking up {u} in index ...")
    try:
        doc = idx.get_by_url(u, prefer=(prefer or CACHE_PREFER))
        print(f"Cache hit for {u}: {doc}")
    except Exception as e:
        print(f"Index lookup failed for {u}: {e}")
        doc = None

    if doc and (doc.page_content or "").strip():
        txt = (doc.page_content or "").strip()
        return {
            "url": u,
            "domain": _url_domain(u),
            "status": "cached",
            "content_type": "",
            "title": (doc.metadata or {}).get("title", ""),
            "fetched_at": (doc.metadata or {}).get("fetched_at", ""),
            "summarized_at": (doc.metadata or {}).get("summarized_at", ""),
            "content_text": txt if with_content else "",
            "content_len": len(txt),
            "truncated": False,
            "from_cache": True,
        }

    return None


def _fetch_url_meta(
    u: str,
    *,
    timeout_s: float,
    with_content: bool,
    max_bytes: int,
    max_chars: int,
):
    """
    Cache-first shim: if USE_URL_INDEX_CACHE is True and URL is in the local index,
    return it; otherwise (or on cache miss) fall back to the original network fetch.
    """
    if USE_URL_INDEX_CACHE:
        cached = _lookup_url_in_index(u, with_content=with_content)
        if cached is not None:
            return cached

        if CACHE_ONLY:
            return {
                "url": u,
                "domain": _url_domain(u),
                "status": "cache_miss",
                "content_type": "",
                "title": "",
                "fetched_at": "",
                "summarized_at": "",
                "content_text": "",  # empty either way on miss
                "content_len": 0,
                "truncated": False,
                "from_cache": False,
            }

    # fallback to original behavior
    return _orig_fetch_url_meta(
        u,
        timeout_s=timeout_s,
        with_content=with_content,
        max_bytes=max_bytes,
        max_chars=max_chars,
    )



if __name__ == "__main__":
    main()
