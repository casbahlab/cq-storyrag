#!/usr/bin/env python3
"""
dump_sparql_manifest.py — generate a mapping of CQ IDs to SPARQL source types.

Reads your per-mode cq_metadata.json and writes sparql_manifest.json next to it.

Manifest includes:
- counts by source (direct/file/rq_scan/fallback_named/missing)
- per-ID entry with: source, has_sparql, sparql_len, preview, question

Usage:
  python3 dump_sparql_manifest.py --meta ./KG/cq_metadata.json
"""
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to cq_metadata.json")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    order = data.get("order") or []
    rows  = data.get("metadata") or {}

    counts = {k:0 for k in ["direct","file","rq_scan","fallback_named","missing"]}
    manifest = {}

    for sid in order:
        rec = rows.get(sid, {})
        src = (rec.get("sparql_source") or "missing").strip()
        sparql = rec.get("sparql") or ""
        counts[src] = counts.get(src, 0) + 1
        entry = {
            "source": src,
            "has_sparql": bool(sparql.strip()),
            "sparql_len": len(sparql),
            "preview": (sparql.strip().splitlines() or [""])[0][:200],
            "question": rec.get("question","")
        }
        manifest[sid] = entry

    out = {
        "file": str(meta_path),
        "total": len(order),
        "counts": counts,
        "manifest": manifest
    }

    out_path = meta_path.with_name("sparql_manifest.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
