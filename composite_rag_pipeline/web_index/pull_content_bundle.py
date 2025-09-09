#!/usr/bin/env python3
# pull_content_bundle.py
#
# Read a list of URLs and fetch content:
# - For Wikipedia URLs **with a fragment** (e.g., .../wiki/Page#Broadcasts), extract the content
#   under that H2 using wikifetch.content_under_h2(h2_key=<fragment>).
# - For all other URLs (and wiki URLs without a fragment), extract generic readable text.
#
# Output: JSONL — one object per URL with fields:
#   { "url": str, "kind": "wiki_h2"|"generic", "title": str, "section": str|None, "text": str }
#
# Usage:
#   python pull_content_bundle.py --in urls.txt --out content_bundle.jsonl
#
import argparse, json, sys, textwrap
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import httpx
import trafilatura
from bs4 import BeautifulSoup

# import wikifetch from local directory
import importlib.util, os
_here = Path(__file__).resolve().parent
_wikifetch_path = _here / "wikifetch.py"
if not _wikifetch_path.exists():
    # allow usage if it's installed elsewhere in PYTHONPATH
    import wikifetch as _wikifetch
else:
    spec = importlib.util.spec_from_file_location("wikifetch", _wikifetch_path.as_posix())
    _wikifetch = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(_wikifetch)

def is_wiki(u: str) -> bool:
    host = urlparse(u).netloc.lower()
    return host.endswith("wikipedia.org")

def canonical_title(soup: BeautifulSoup) -> str:
    t = soup.find("title")
    if t:
        return t.get_text(strip=True)[:180]
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)[:180]
    return ""

def generic_extract(u: str, timeout: float=20.0) -> tuple[str, str]:
    """Return (title, text)."""
    try:
        r = httpx.get(u, timeout=timeout, follow_redirects=True, headers={"User-Agent": "kg-puller/1.0"})
        raw = r.text
        text = trafilatura.extract(raw, include_comments=False, include_tables=False, favor_recall=True) or ""
        if not text:
            soup = BeautifulSoup(raw, "html.parser")
            title = canonical_title(soup)
            body_text = soup.get_text(" ", strip=True)
            return title, body_text
        # best-effort title
        soup = BeautifulSoup(raw, "html.parser")
        title = canonical_title(soup)
        return title, text
    except Exception as e:
        return "", ""

def wiki_h2_extract(u: str) -> tuple[Optional[str], Optional[str]]:
    """If URL has a fragment, return (section, text) using wikifetch.content_under_h2."""
    parsed = urlparse(u)
    frag = parsed.fragment
    if not frag:
        return None, None
    # Normalize h2 key: unquote and strip, convert underscores to spaces
    key = unquote(frag).strip()
    key = key.lstrip("#")
    key_variants = [key, key.replace("_", " ")]
    # Try variants until one works
    for k in key_variants:
        try:
            paras = _wikifetch.content_under_h2(u, h2_key=k, content="p")  # returns list[str]
            text = "\n\n".join(paras).strip()
            if text:
                return k, text
        except Exception:
            continue
    return frag, None

def main():
    ap = argparse.ArgumentParser(description="Pull content for a list of URLs with Wikipedia H2 handling.")
    ap.add_argument("--in", dest="infile", required=True, help="Text file with one URL per line")
    ap.add_argument("--out", dest="outfile", required=True, help="Output JSONL path")
    ap.add_argument("--min-chars", type=int, default=400, help="Skip entries whose extracted text is shorter than this")
    ap.add_argument("--max-urls", type=int, default=None, help="Optional limit on number of URLs processed")
    args = ap.parse_args()

    src = Path(args.infile)
    out = Path(args.outfile)

    urls = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if args.max_urls:
        urls = urls[: args.max_urls]

    with out.open("w", encoding="utf-8") as fo:
        for u in urls:
            kind = "generic"
            section = None
            text = ""
            title = ""
            if is_wiki(u) and urlparse(u).fragment:
                sec, txt = wiki_h2_extract(u)
                if txt:
                    kind = "wiki_h2"
                    section = sec
                    text = txt
                # if fail, fall back to generic
            if not text:
                title, text = generic_extract(u)

            # length filter
            if len(text) < args.min_chars:
                continue

            fo.write(json.dumps({
                "url": u,
                "kind": kind,
                "section": section,
                "title": title,
                "text": text
            }) + "\n")

    print(f"[done] wrote: {out}")

if __name__ == "__main__":
    main()
