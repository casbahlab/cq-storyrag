#!/usr/bin/env python3
"""
index_sanity_check.py — quick health check for your KG/Hybrid index folders.

Checks:
- cq_metadata.json exists and has records
- how many CQs have non-empty SPARQL
- per-beat distribution (via beat_title/beat_slug)
- embeddings.npy / FAISS presence (if you built them)

Usage:
  python3 index_sanity_check.py ../index/KG
  python3 index_sanity_check.py ../index/Hybrid
"""
from __future__ import annotations
import json, os, sys, re
from pathlib import Path
from typing import Any, Dict

def _slug_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")

def _to_text(x: Any) -> str:
    """Robustly convert beat_title-like values to a readable string.
       - str: return as-is
       - list: first non-empty string; else join stringified items with ' / '
       - dict: try common keys; else join stringified values
       - other: str(x)
    """
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        for y in x:
            if isinstance(y, str) and y.strip():
                return y
        return " / ".join(str(y) for y in x if y is not None)
    if isinstance(x, dict):
        for k in ("title", "label", "name", "value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # fallback: join stringy values
        vals = [str(v) for v in x.values() if isinstance(v, (str, int, float))]
        return " / ".join(vals)
    if x is None:
        return ""
    return str(x)

def slug(x: Any) -> str:
    return _slug_text(_to_text(x))

def load_meta(p: Path):
    j = p / "cq_metadata.json"
    if not j.exists():
        return None, f"Missing: {j}"
    try:
        data = json.loads(j.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"Bad JSON: {j} ({e})"
    meta = data.get("metadata") or {}
    if not isinstance(meta, dict):
        return None, f"'metadata' not a dict in {j}"
    return meta, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 index_sanity_check.py <index_dir>")
        sys.exit(2)
    idx = Path(sys.argv[1])

    print(f"\n== Index: {idx} ==")

    meta, err = load_meta(idx)
    if err:
        print("ERROR:", err)
        sys.exit(1)

    total = len(meta)
    with_sparql = 0
    with_beat = 0
    by_beat: Dict[str, int] = {}
    empty_qids = []
    non_string_beats = 0

    for cid, rec in meta.items():
        sp = (rec.get("sparql") or "").strip()
        if sp:
            with_sparql += 1
        else:
            empty_qids.append(cid)

        # Accept multiple possible fields and types for beat
        bt_raw = rec.get("beat_title")
        if not bt_raw:
            bt_raw = rec.get("Beat") or rec.get("beat") or rec.get("beatLabel") or rec.get("beat_name") or rec.get("beats")

        bt_txt = _to_text(bt_raw)
        bt_slug = _slug_text(bt_txt)

        if bt_raw is not None and not isinstance(bt_raw, str):
            non_string_beats += 1

        if bt_txt.strip():
            with_beat += 1
        if bt_slug:
            by_beat[bt_slug] = by_beat.get(bt_slug, 0) + 1

    print(f"Total CQs: {total}")
    print(f"With SPARQL: {with_sparql} ({(with_sparql/total*100 if total else 0):.1f}%)")
    print(f"With beat title: {with_beat} ({(with_beat/total*100 if total else 0):.1f}%)")

    top = sorted(by_beat.items(), key=lambda x: (-x[1], x[0]))[:10]
    print("Top beats:", top if top else "(none)")

    emb = idx / "embeddings.npy"
    faiss1 = idx / "index.faiss"
    faiss2 = idx / "faiss.index"
    print(f"embeddings.npy: {'OK' if emb.exists() else 'missing'}")
    print(f"FAISS: {'OK' if (faiss1.exists() or faiss2.exists()) else 'missing'}")

    if non_string_beats:
        print(f"\nNote: {non_string_beats} beat_title values were lists/dicts (handled).")

    if empty_qids:
        print(f"\nCQs with EMPTY SPARQL: {len(empty_qids)} (showing up to 20)")
        for cid in empty_qids[:20]:
            print(" -", cid)
        if len(empty_qids) > 20:
            print(" ...")

if __name__ == "__main__":
    main()
