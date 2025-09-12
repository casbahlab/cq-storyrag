#!/usr/bin/env python3
# url_index_retriever.py
"""
Drop-in retriever for the local URL index we built.
- Looks up content by URL (no live calls).
- Optional keyword search over title/summary/text (FTS5 if available; falls back to LIKE).
- Returns documents ready to feed to your RAG pipeline.

DB schema expected (created by prepare_content_index.py or ttl_url_index.py):
  url_index(url PRIMARY KEY, title, text, summary, status, content_type, fetched_at, summarized_at)
  url_meta(url PRIMARY KEY, kind, section)

Usage
-----
from url_index_retriever import UrlIndexRetriever

r = UrlIndexRetriever("/path/to/url_index.db")

# 1) Direct lookup by URL
doc = r.get_by_url("https://en.wikipedia.org/wiki/Live_Aid#Broadcasts", prefer="summary")

# 2) Batch hydrate into RAG documents
docs = r.hydrate(["https://...", "https://..."], prefer="summary")

# 3) Keyword search (FTS if available), returns top-k docs
hits = r.search("Queen Live Aid setlist", k=5, prefer="summary")

# 4) Convert to your RAG framework
# docs is a list of dicts: {"page_content": str, "metadata": {...}}
"""

import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any

@dataclass
class Doc:
    page_content: str
    metadata: Dict[str, Any]

class UrlIndexRetriever:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_fts()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------ URL lookup ------------------------
    def _row_to_doc(self, row, prefer: str = "summary", min_chars: int = 40) -> Optional[Doc]:
        if not row:
            return None
        text = (row["summary"] or "").strip() if prefer == "summary" else (row["text"] or "").strip()
        if len(text) < min_chars:
            # fallback to other field
            alt = (row["text"] or "").strip() if prefer == "summary" else (row["summary"] or "").strip()
            if len(alt) > len(text):
                text = alt
        if not text:
            return None
        meta = {
            "url": row["url"],
            "title": row["title"] or "",
            "status": row["status"] or "",
            "content_type": row["content_type"] or "",
            "fetched_at": row["fetched_at"] or "",
            "summarized_at": row["summarized_at"] or "",
        }
        # join meta table if present
        try:
            m = self._conn.execute("SELECT kind, section FROM url_meta WHERE url = ?", (row["url"],)).fetchone()
            if m:
                meta["kind"] = m["kind"]
                meta["section"] = m["section"]
        except Exception:
            pass
        return Doc(page_content=text, metadata=meta)

    def get_by_url(self, url: str, prefer: str = "summary", min_chars: int = 40) -> Optional[Doc]:
        row = self._conn.execute("SELECT * FROM url_index WHERE url = ?", (url,)).fetchone()
        return self._row_to_doc(row, prefer=prefer, min_chars=min_chars)

    def hydrate(self, urls: Iterable[str], prefer: str = "summary", min_chars: int = 40) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for u in urls:
            d = self.get_by_url(u, prefer=prefer, min_chars=min_chars)
            if d:
                docs.append({"page_content": d.page_content, "metadata": d.metadata})
        return docs

    # ------------------------ Keyword search ------------------------
    def _ensure_fts(self):
        # Create FTS5 view if it doesn't exist, populated from url_index
        try:
            self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS url_fts USING fts5(
                url UNINDEXED, title, summary, text,
                content='',
                tokenize='porter'
            );
            """)
            # If table empty, populate
            n = self._conn.execute("SELECT count(*) AS n FROM url_fts").fetchone()["n"]
            if n == 0:
                rows = self._conn.execute("SELECT url, title, summary, text FROM url_index").fetchall()
                self._conn.executemany("INSERT INTO url_fts(url, title, summary, text) VALUES (?,?,?,?)",
                                       [(r["url"], r["title"], r["summary"], r["text"]) for r in rows])
                self._conn.commit()
        except Exception:
            # FTS not available in this SQLite build; silently skip
            pass

    def refresh_fts(self):
        try:
            self._conn.execute("DELETE FROM url_fts")
            rows = self._conn.execute("SELECT url, title, summary, text FROM url_index").fetchall()
            self._conn.executemany("INSERT INTO url_fts(url, title, summary, text) VALUES (?,?,?,?)",
                                   [(r["url"], r["title"], r["summary"], r["text"]) for r in rows])
            self._conn.commit()
        except Exception:
            pass

    def search(self, query: str, k: int = 5, prefer: str = "summary", min_chars: int = 40) -> List[Dict[str, Any]]:
        rows = []
        # Try FTS first
        try:
            rows = self._conn.execute("""
                SELECT url FROM url_fts
                WHERE url_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, k)).fetchall()
            urls = [r["url"] for r in rows]
            if not urls:
                raise RuntimeError("No FTS hits")
            docs = self.hydrate(urls, prefer=prefer, min_chars=min_chars)
            return docs
        except Exception:
            # Fallback: LIKE search on title/summary
            like = f"%{query}%"
            rows = self._conn.execute("""
                SELECT * FROM url_index
                WHERE (title LIKE ? OR summary LIKE ? OR text LIKE ?)
                ORDER BY summarized_at DESC, fetched_at DESC
                LIMIT ?
            """, (like, like, like, k)).fetchall()
            docs = []
            for r in rows:
                d = self._row_to_doc(r, prefer=prefer, min_chars=min_chars)
                if d:
                    docs.append({"page_content": d.page_content, "metadata": d.metadata})
            return docs

# Optional CLI for quick manual checks
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="Fetch by exact URL")
    g.add_argument("--q", help="Search query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--prefer", choices=["summary","text"], default="summary")
    args = ap.parse_args()

    r = UrlIndexRetriever(args.db)
    try:
        if args.url:
            d = r.get_by_url(args.url, prefer=args.prefer)
            print(json.dumps({"page_content": d.page_content, "metadata": d.metadata}, ensure_ascii=False, indent=2) if d else "NOT FOUND")
        else:
            docs = r.search(args.q, k=args.k, prefer=args.prefer)
            for i, d in enumerate(docs, 1):
                print(f"\n--- HIT {i} ---")
                print(json.dumps({"page_content": d["page_content"][:500], "metadata": d["metadata"]}, ensure_ascii=False, indent=2))
    finally:
        r.close()
