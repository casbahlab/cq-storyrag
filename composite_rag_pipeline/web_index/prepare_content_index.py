#!/usr/bin/env python3
"""
prepare_content_index.py
------------------------
End-to-end helper to:
1) Read URLs (one per line)
2) Fetch content
   - Wikipedia URLs with a #fragment → use wikifetch.content_under_h2(h2_key=<fragment>)
   - Everything else → generic readable-text extraction (Trafilatura → BS4 fallback)
3) Summarize to ~N words (default 120)
   - extractive (fast, no API) OR
   - Gemini (set GEMINI_API_KEY or pass --api-key)
4) Build a local SQLite index (url_index + url_meta)

Outputs in --out-dir:
- content_bundle.jsonl          # raw fetched content
- content_summaries.jsonl/csv   # short summaries
- url_index.db                  # SQLite index

Usage:
  python prepare_content_index.py \
    --in urls.txt \
    --out-dir outdir \
    --summarizer extractive \
    --target-words 120

  # With Gemini summarization:
  export GEMINI_API_KEY=...  # or pass --api-key
  python prepare_content_index.py \
    --in urls.txt --out-dir outdir --summarizer gemini --target-words 120
"""
import argparse, json, os, re, sqlite3, textwrap, datetime as dt
from typing import List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse, unquote

import httpx
import trafilatura
from bs4 import BeautifulSoup

# --- Optional local wikifetch.py (same folder) ---
import importlib.util as _ilu, sys as _sys
_here = Path(__file__).resolve().parent
_wikifetch_path = _here / "wikifetch.py"
if _wikifetch_path.exists():
    spec = _ilu.spec_from_file_location("wikifetch", _wikifetch_path.as_posix())
    wikifetch = _ilu.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(wikifetch)
else:
    try:
        import wikifetch  # if installed / on PYTHONPATH
    except Exception:
        wikifetch = None

# ----------------------------- Helpers -----------------------------
def is_wiki(u: str) -> bool:
    try:
        return urlparse(u).netloc.lower().endswith("wikipedia.org")
    except Exception:
        return False

def canonical_title(soup: BeautifulSoup) -> str:
    t = soup.find("title")
    if t:
        return t.get_text(strip=True)[:180]
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)[:180]
    return ""

def generic_extract(u: str, timeout: float=25.0) -> tuple[str, str]:
    """Return (title, text)."""
    try:
        r = httpx.get(u, timeout=timeout, follow_redirects=True, headers={"User-Agent": "kg-pipeline/1.0"})
        raw = r.text
        text = trafilatura.extract(raw, include_comments=False, include_tables=False, favor_recall=True) or ""
        soup = BeautifulSoup(raw, "html.parser")
        title = canonical_title(soup)
        if not text:
            body_text = soup.get_text(" ", strip=True)
            return title, body_text
        return title, text
    except Exception:
        return "", ""

def wiki_h2_extract(u: str) -> tuple[Optional[str], Optional[str]]:
    """If url has a fragment, use wikifetch.content_under_h2(h2_key=fragment)."""
    if wikifetch is None:
        return None, None
    frag = urlparse(u).fragment
    if not frag:
        return None, None
    key = unquote(frag).lstrip("#").strip()
    variants = [key, key.replace("_", " ")]
    for k in variants:
        try:
            paras = wikifetch.content_under_h2(u, h2_key=k, content="p", parse_tables=False)
            text = "\n\n".join(paras).strip()
            if text:
                return k, text
        except Exception:
            continue
    return key or frag, None

# --------------------------- Summarizers ---------------------------
_WORD_RE = re.compile(r"\w+(?:'\w+)?")

def _word_count(s: str) -> int:
    return len(_WORD_RE.findall(s or ""))

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text.replace("\r", " ").strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", t)
    if len(parts) <= 1:
        parts = t.split(". ")
        parts = [p if p.endswith(".") else p + "." for p in parts if p]
    out = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        if s.lower().startswith(("retrieved from", "see also", "external links", "references")):
            continue
        if _word_count(s) < 3:
            continue
        out.append(s)
    return out

def summarize_extractive(text: str, target_words: int=120, hard_cap: int=140) -> str:
    sents = _split_sentences(text)
    if not sents:
        return " ".join((text or "").split()[:target_words])
    total = 0
    out = []
    for s in sents:
        wc = _word_count(s)
        if total + wc > hard_cap:
            break
        out.append(s)
        total += wc
        if total >= target_words:
            break
    summ = " ".join(out).strip()
    if _word_count(summ) > hard_cap:
        words = summ.split()
        summ = " ".join(words[:hard_cap])
    return summ

def summarize_gemini(text: str, api_key: Optional[str], target_words: int=120) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini summarizer selected but GEMINI_API_KEY not provided.")
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("google-generativeai not installed. pip install google-generativeai") from e
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = textwrap.dedent(f"""
    Summarize the following page content in under {target_words} words.
    Be factual, neutral, and self-contained. Mention key named entities and dates if present.

    Content:
    {text[:12000]}
    """).strip()
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

# ---------------------------- DB helpers ----------------------------
DB_SCHEMA_INDEX = """
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
DB_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS url_meta (
  url TEXT PRIMARY KEY,
  kind TEXT,
  section TEXT,
  FOREIGN KEY(url) REFERENCES url_index(url) ON DELETE CASCADE
);
"""

def open_db(path: Path):
    import sqlite3
    conn = sqlite3.connect(path.as_posix())
    cur = conn.cursor()
    cur.execute(DB_SCHEMA_INDEX)
    cur.execute(DB_SCHEMA_META)
    conn.commit()
    return conn

# ------------------------------ Main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch → summarize → index URLs (with Wikipedia H2 handling).")
    ap.add_argument("--in", dest="infile", required=True, help="Text file with one URL per line")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--summarizer", choices=["extractive","gemini"], default="extractive", help="Summarizer to use")
    ap.add_argument("--target-words", type=int, default=120, help="Target words for summary")
    ap.add_argument("--min-chars", type=int, default=400, help="Skip items with < N chars extracted")
    ap.add_argument("--max-urls", type=int, default=None, help="Optional limit for quick runs")
    ap.add_argument("--api-key", default=None, help="Gemini API key (or env GEMINI_API_KEY)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "content_bundle.jsonl"
    summaries_jsonl = out_dir / "content_summaries.jsonl"
    summaries_csv = out_dir / "content_summaries.csv"
    db_path = out_dir / "url_index.db"

    urls = [ln.strip() for ln in Path(args.infile).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    if args.max_urls:
        urls = urls[: args.max_urls]

    # --- Step 1: Fetch content ---
    fetched = []
    with bundle_path.open("w", encoding="utf-8") as fo:
        for u in urls:
            kind = "generic"
            section = None
            title = ""
            text = ""

            if is_wiki(u) and urlparse(u).fragment:
                sec, txt = wiki_h2_extract(u)
                if txt:
                    kind = "wiki_h2"
                    section = sec
                    text = txt

            if not text:
                title, text = generic_extract(u)

            if len(text) < args.min_chars:
                continue

            rec = {"url": u, "kind": kind, "section": section, "title": title, "text": text}
            fetched.append(rec)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- Step 2: Summarize ---
    rows = []
    with summaries_jsonl.open("w", encoding="utf-8") as fo:
        for rec in fetched:
            txt = rec["text"]
            if args.summarizer == "extractive":
                summ = summarize_extractive(txt, target_words=args.target_words, hard_cap=int(args.target_words*1.2))
            else:
                summ = summarize_gemini(txt, api_key=args.api_key, target_words=args.target_words)
            row = {
                "url": rec["url"],
                "kind": rec["kind"],
                "section": rec["section"],
                "title": rec["title"],
                "summary": summ
            }
            rows.append(row)
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Also CSV
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(summaries_csv, index=False)
    except Exception:
        pass

    # --- Step 3: Index (SQLite) ---
    conn = open_db(db_path); cur = conn.cursor()
    now_iso = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    for rec, row in zip(fetched, rows):
        cur.execute("""
        INSERT INTO url_index(url,title,text,summary,status,content_type,fetched_at,summarized_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
          title=excluded.title, text=excluded.text, summary=excluded.summary,
          status=excluded.status, content_type=excluded.content_type,
          fetched_at=excluded.fetched_at, summarized_at=excluded.summarized_at
        """, (rec["url"], rec["title"], rec["text"], row["summary"], "200", "", now_iso, now_iso))
        cur.execute("""
        INSERT INTO url_meta(url, kind, section)
        VALUES (?,?,?)
        ON CONFLICT(url) DO UPDATE SET
          kind=excluded.kind, section=excluded.section
        """, (rec["url"], rec["kind"], rec["section"]))
    conn.commit(); conn.close()

    print("[done]")
    print(f"  bundle:   {bundle_path}")
    print(f"  summaries:{summaries_jsonl} ; {summaries_csv}")
    print(f"  index:    {db_path}")

if __name__ == "__main__":
    main()
