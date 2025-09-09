#!/usr/bin/env python3
"""
ttl_url_index.py — URL indexer and summarizer for TTL graphs

This build adds filtering:
- --exclude-prefix : skip any URL that starts with any of the given prefixes
- --exclude-domain : skip any URL whose hostname matches any of the given domains (incl. subdomains)
- --include-domain : (optional) only keep URLs whose hostname matches these domains (incl. subdomains)

Example:
  python ttl_url_index.py extract-objects \
    --data ./kg --mask "*.ttl" \
    --out-urls urls.txt \
    --exclude-domain wembrewind.live \
    --exclude-prefix http://wembrewind.live https://wembrewind.live
"""
import argparse
import asyncio
import csv
import datetime as dt
import os
import re
import sqlite3
import sys
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from rdflib import Graph, URIRef, Literal
import httpx
from bs4 import BeautifulSoup
import trafilatura

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS url_index (
  url TEXT PRIMARY KEY,
  title TEXT,
  text TEXT,
  summary TEXT,
  status TEXT,
  content_type TEXT,
  fetched_at TEXT,
  summarized_at TEXT
);
"""

URL_RE = re.compile(r"""(?i)\bhttps?://[^\s<>"']+""")

def open_db(path: Path):
    conn = sqlite3.connect(path.as_posix())
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(DB_SCHEMA)
    return conn

def now_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ------------------------------- helpers -----------------------------------
def parse_multi(values: Optional[List[str]]) -> List[str]:
    """Allow repeated args and/or comma-separated lists."""
    out = []
    if not values:
        return out
    for v in values:
        if v is None:
            continue
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    return out

def hostname(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc.lower()
    except Exception:
        return ""

def domain_match(host: str, doms: List[str]) -> bool:
    """Return True if host equals or is a subdomain of any dom in doms."""
    if not host:
        return False
    h = host.lstrip(".").lower()
    for d in doms:
        d = d.lstrip(".").lower()
        if h == d or h.endswith("." + d):
            return True
    return False

def should_keep(url: str, exclude_prefixes: List[str], exclude_domains: List[str], include_domains: List[str]) -> bool:
    for p in exclude_prefixes:
        if url.startswith(p):
            return False
    host = hostname(url)
    if exclude_domains and domain_match(host, exclude_domains):
        return False
    if include_domains and not domain_match(host, include_domains):
        return False
    return True

# ---------------------------------------------------------------------------
# Strict RDF parse → collect ONLY object-position URLs
# ---------------------------------------------------------------------------
def extract_object_urls(paths: List[Path], mask: Optional[str], include_iri: bool, include_literal: bool,
                        pred_regex: Optional[str]=None, out_urls: Optional[Path]=None,
                        out_context_csv: Optional[Path]=None,
                        exclude_prefixes: Optional[List[str]]=None,
                        exclude_domains: Optional[List[str]]=None,
                        include_domains: Optional[List[str]]=None) -> List[str]:
    urls = []
    seen = set()

    pred_re = re.compile(pred_regex, re.I) if pred_regex else None
    exclude_prefixes = exclude_prefixes or []
    exclude_domains = exclude_domains or []
    include_domains = include_domains or []

    def iter_files(p: Path, mask: Optional[str]):
        if p.is_file():
            yield p
        else:
            patt = mask or "*"
            for f in p.rglob(patt):
                if f.is_file():
                    yield f

    context_rows = []

    for path in paths:
        for file in iter_files(path, mask):
            g = Graph()
            parsed = False
            for fmt in ["turtle", "xml", "n3", "nt", "trig", "json-ld"]:
                try:
                    g.parse(file.as_posix(), format=fmt)
                    parsed = True
                    break
                except Exception:
                    continue
            if not parsed:
                continue

            for s, p, o in g:
                if pred_re and not pred_re.search(str(p)):
                    continue
                if include_iri and isinstance(o, URIRef):
                    so = str(o)
                    if so.startswith("http://") or so.startswith("https://"):
                        if so not in seen and should_keep(so, exclude_prefixes, exclude_domains, include_domains):
                            seen.add(so); urls.append(so)
                            if out_context_csv:
                                context_rows.append((str(s), str(p), so))
                if include_literal and isinstance(o, Literal):
                    ss = str(o)
                    m = URL_RE.search(ss)
                    if m:
                        so = m.group(0)
                        if so not in seen and should_keep(so, exclude_prefixes, exclude_domains, include_domains):
                            seen.add(so); urls.append(so)
                            if out_context_csv:
                                context_rows.append((str(s), str(p), ss[:500]))

    if out_urls:
        out_urls.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    if out_context_csv and context_rows:
        with out_context_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["subject","predicate","object_snippet_or_url"]); w.writerows(context_rows)
    return urls

# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------
async def fetch_one(client: httpx.AsyncClient, url: str, timeout: float=20.0) -> Tuple[str, str, str]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True)
        ctype = r.headers.get("content-type", "")
        status = f"{r.status_code}"
        raw = r.text
        extracted = trafilatura.extract(raw, include_comments=False, include_tables=False, favor_recall=True)
        text = extracted or ""
        if not text:
            soup = BeautifulSoup(raw, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            body_text = soup.get_text(" ", strip=True)
            text = "\n".join([title, body_text]).strip()
        return status, ctype, text
    except Exception as e:
        return f"ERR:{type(e).__name__}", "", ""

async def crawl_urls(urls: List[str], db_path: Path, max_urls: Optional[int]=None, concurrency: int=6, force: bool=False):
    conn = open_db(db_path)
    cur = conn.cursor()
    existing = {row[0]: (row[1] or "", row[2] or "") for row in cur.execute("SELECT url,title,text FROM url_index")}

    to_do = []
    for u in urls:
        if (not force) and u in existing and existing[u][1]:
            continue
        to_do.append(u)
        if max_urls and len(to_do) >= max_urls:
            break

    if not to_do:
        conn.close()
        return 0

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers={"User-Agent": "ttl-url-index/1.0"}) as client:
        async def worker(u):
            async with sem:
                status, ctype, text = await fetch_one(client, u)
                title = text.splitlines()[0].strip()[:140] if text else ""
                cur.execute(
                    "INSERT INTO url_index(url,title,text,status,content_type,fetched_at) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(url) DO UPDATE SET title=excluded.title, text=excluded.text, status=excluded.status, "
                    "content_type=excluded.content_type, fetched_at=excluded.fetched_at",
                    (u, title, text, status, ctype, now_iso())
                )
        await asyncio.gather(*[worker(u) for u in to_do])
    conn.commit()
    n = len(to_do)
    conn.close()
    return n

# ---------------------------------------------------------------------------
# Summarization (Gemini-only)
# ---------------------------------------------------------------------------
def summarize_with_gemini(text: str, api_key: str, target_words: int=120) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai not installed. pip install google-generativeai") from e
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = textwrap.dedent(f"""
    Summarize the following webpage content in under {target_words} words.
    Make it factual, neutral, and self-contained. Mention key named entities and dates if present.

    Content:
    {text[:12000]}
    """).strip()
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

def run_summarize(db_path: Path, target_words: int=120, api_key: Optional[str]=None, limit: Optional[int]=None, overwrite: bool=False):
    conn = open_db(db_path)
    cur = conn.cursor()
    rows = list(cur.execute("SELECT url, text, summary FROM url_index ORDER BY fetched_at DESC"))
    todo = []
    for url, text, summary in rows:
        if not text: continue
        if overwrite or not summary:
            todo.append((url, text))
        if limit is not None and len(todo) >= limit:
            break
    if not todo:
        conn.close()
        return 0
    n_done = 0
    for url, text in todo:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set env var or pass --api-key.")
        summ = summarize_with_gemini(text, key, target_words=target_words)
        cur.execute("UPDATE url_index SET summary=?, summarized_at=? WHERE url=?", (summ, now_iso(), url))
        n_done += 1
    conn.commit()
    conn.close()
    return n_done

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_extract_objects(args):
    paths = [Path(p) for p in args.data]
    out_urls = Path(args.out_urls) if args.out_urls else None
    out_ctx = Path(args.out_context_csv) if args.out_context_csv else None
    urls = extract_object_urls(
        paths, mask=args.mask,
        include_iri=not args.no_iri, include_literal=not args.no_literal,
        pred_regex=args.pred_regex, out_urls=out_urls, out_context_csv=out_ctx,
        exclude_prefixes=parse_multi(args.exclude_prefix),
        exclude_domains=parse_multi(args.exclude_domain),
        include_domains=parse_multi(args.include_domain),
    )
    print(f"[extract-objects] collected {len(urls)} unique object-URLs")
    if out_urls:
        print(f"[extract-objects] wrote URL list → {out_urls}")
    if out_ctx:
        print(f"[extract-objects] wrote context CSV → {out_ctx}")

def cmd_ingest(args):
    path = Path(args.file)
    if not path.exists():
        print(f"[ingest] file not found: {path}"); sys.exit(1)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    urls = [u for u in lines if u.startswith("http://") or u.startswith("https://")]
    bad = [u for u in lines if u not in urls]
    if bad:
        print(f"[ingest] warning: {len(bad)} invalid lines skipped")
    conn = open_db(Path(args.db)); cur = conn.cursor()
    for u in urls:
        cur.execute("INSERT OR IGNORE INTO url_index(url,status) VALUES (?,?)", (u, "NEW"))
    conn.commit(); conn.close()
    print(f"[ingest] added {len(urls)} URLs into {args.db}")

def cmd_crawl(args):
    conn = open_db(Path(args.db))
    cur = conn.cursor()
    urls = [row[0] for row in cur.execute("SELECT url FROM url_index")]
    conn.close()
    if not urls:
        print("[crawl] no URLs in DB. Run 'extract-objects' + 'ingest' first.")
        return
    n = asyncio.run(crawl_urls(urls, db_path=Path(args.db), max_urls=args.max, concurrency=args.concurrency, force=args.force))
    print(f"[crawl] processed {n} URLs")

def cmd_summarize(args):
    n = run_summarize(db_path=Path(args.db), target_words=args.target_words, api_key=args.api_key, limit=args.limit, overwrite=args.overwrite)
    print(f"[summarize] summarized {n} entries")

def cmd_build(args):
    tmp = Path(args.temp_urls or "extracted_object_urls.txt")
    cmd_extract_objects(argparse.Namespace(
        data=args.data, mask=args.mask, out_urls=str(tmp), out_context_csv=None,
        no_iri=args.no_iri, no_literal=args.no_literal, pred_regex=args.pred_regex,
        exclude_prefix=args.exclude_prefix, exclude_domain=args.exclude_domain, include_domain=args.include_domain
    ))
    cmd_ingest(argparse.Namespace(file=str(tmp), db=args.db))
    cmd_crawl(argparse.Namespace(db=args.db, max=args.max, concurrency=args.concurrency, force=args.force))
    cmd_summarize(argparse.Namespace(db=args.db, target_words=args.target_words, api_key=args.api_key, limit=args.limit, overwrite=args.overwrite))

def cmd_query(args):
    conn = open_db(Path(args.db))
    cur = conn.cursor()
    row = cur.execute("SELECT url, title, summary, text, fetched_at, summarized_at FROM url_index WHERE url=?", (args.url,)).fetchone()
    conn.close()
    if not row:
        print("NOT FOUND in index"); sys.exit(1)
    url, title, summary, text, fetched_at, summarized_at = row
    print("="*80); print(url)
    print(f"Fetched: {fetched_at}  |  Summarized: {summarized_at}")
    print("-"*80)
    print(summary or "(No summary cached — run summarize)")
    if args.verbose_text:
        print("\n[Full extracted text]\n"); print((text or "")[:4000])

def main():
    ap = argparse.ArgumentParser(description="TTL URL indexer and summarizer (object-only extraction + filters)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--db", default="url_index.db", help="SQLite DB path")

    s0 = sub.add_parser("extract-objects", help="Extract ONLY object-position URLs into a file (and optional context CSV)")
    s0.add_argument("--data", nargs="+", required=True, help="TTL/RDF files or directories")
    s0.add_argument("--mask", default="*.ttl", help="Glob used when directories are provided")
    s0.add_argument("--out-urls", required=True, help="Output text file with one URL per line")
    s0.add_argument("--out-context-csv", default=None, help="Optional CSV capturing (subject,predicate,object-snippet)")
    s0.add_argument("--no-iri", action="store_true", help="Exclude IRI objects")
    s0.add_argument("--no-literal", action="store_true", help="Exclude literal objects that look like URLs")
    s0.add_argument("--pred-regex", default=None, help="Only include triples whose predicate IRI matches this regex")
    s0.add_argument("--exclude-prefix", action="append", default=None, help="Skip URLs that start with these prefix(es). Repeat or comma-separate.")
    s0.add_argument("--exclude-domain", action="append", default=None, help="Skip URLs whose host matches these domain(s). Repeat or comma-separate.")
    s0.add_argument("--include-domain", action="append", default=None, help="Keep ONLY URLs whose host matches these domain(s). Repeat or comma-separate.")
    s0.set_defaults(func=cmd_extract_objects)

    s1 = sub.add_parser("ingest", help="Ingest a newline-separated URL list file into the DB")
    s1.add_argument("--file", required=True, help="Path to URL list (one URL per line)")
    s1.set_defaults(func=cmd_ingest)

    s2 = sub.add_parser("crawl", help="Crawl URLs from DB and extract readable text")
    s2.add_argument("--max", type=int, default=None, help="Max URLs to process this run")
    s2.add_argument("--concurrency", type=int, default=6, help="Concurrent fetches")
    s2.add_argument("--force", action="store_true", help="Re-fetch even if text exists")
    s2.set_defaults(func=cmd_crawl)

    s3 = sub.add_parser("summarize", help="Summarize extracted texts with Gemini")
    s3.add_argument("--target-words", type=int, default=120, help="Target word budget")
    s3.add_argument("--api-key", default=None, help="Gemini API key (or env GEMINI_API_KEY)")
    s3.add_argument("--limit", type=int, default=None, help="Max rows to summarize")
    s3.add_argument("--overwrite", action="store_true", help="Re-summarize even if summary exists")
    s3.set_defaults(func=cmd_summarize)

    s4 = sub.add_parser("build", help="Extract objects → ingest → crawl → summarize")
    s4.add_argument("--data", nargs="+", required=True, help="Paths for extract step")
    s4.add_argument("--mask", default="*.ttl", help="Glob for extract step")
    s4.add_argument("--no-iri", action="store_true", help="Exclude IRI objects")
    s4.add_argument("--no-literal", action="store_true", help="Exclude literal objects that look like URLs")
    s4.add_argument("--pred-regex", default=None, help="Only include triples whose predicate IRI matches this regex")
    s4.add_argument("--exclude-prefix", action="append", default=None, help="Skip URLs that start with these prefix(es). Repeat or comma-separate.")
    s4.add_argument("--exclude-domain", action="append", default=None, help="Skip URLs whose host matches these domain(s). Repeat or comma-separate.")
    s4.add_argument("--include-domain", action="append", default=None, help="Keep ONLY URLs whose host matches these domain(s). Repeat or comma-separate.")
    s4.add_argument("--temp-urls", default=None, help="Temp URL file path (defaults to extracted_object_urls.txt)")
    s4.add_argument("--max", type=int, default=None, help="Max URLs for crawl step")
    s4.add_argument("--concurrency", type=int, default=6, help="Concurrent fetches")
    s4.add_argument("--force", action="store_true", help="Re-fetch even if text exists")
    s4.add_argument("--target-words", type=int, default=120, help="Target word budget")
    s4.add_argument("--api-key", default=None, help="Gemini API key")
    s4.add_argument("--limit", type=int, default=None, help="Max rows to summarize")
    s4.add_argument("--overwrite", action="store_true", help="Re-summarize even if summary exists")
    s4.set_defaults(func=cmd_build)

    s5 = sub.add_parser("query", help="Query cached summary for a specific URL (no live calls)")
    s5.add_argument("--url", required=True, help="URL to query")
    s5.add_argument("--verbose-text", action="store_true", help="Also print extracted text (first 4k chars)")
    s5.set_defaults(func=cmd_query)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
