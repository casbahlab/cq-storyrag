#!/usr/bin/env python3
"""
check_cq_coverage.py

Purpose
-------
For each provided SPARQL CQ file (.rq), flag the CQ IDs that are present in the
CSV master list but **missing** from that .rq file ("missing_in_rq").
Also shows IDs present in the .rq but not in the CSV ("extra_in_rq") for awareness.

Assumptions
-----------
- CSV has a column containing CQ IDs (e.g., "CQ_ID", "id", "Id", etc.).
  If not provided via --id_col, the script will try to auto-detect a column
  whose values look like "CQ-...".
- Optionally, CSV may have a retrieval-mode column (default "RetrievalMode")
  with values like "KG" and/or "Hybrid". If present, and if the .rq filename
  suggests a mode (contains "hybrid" or "kg"), the script filters the CSV IDs
  to that mode automatically (can be overridden via --mode_filter).
- CQ IDs inside .rq are extracted by a regex that looks for tokens like
  CQ-XYZ, CQ-L12, CQ-H-001, etc.

Usage
-----
python3 check_cq_coverage.py \
  --csv ../data/WembleyRewindCQs_with_beats_trimmed.csv \
  --rq ../data/sparql_templates/cqs_queries_template_filled_in.rq \
  --rq ../data/sparql_templates/cqs_queries_template_filled_in_hybrid.rq

Optional flags:
  --id_col CQ_ID
  --mode_col RetrievalMode
  --mode_filter KG            # force a mode for all files
  --out_json coverage_report.json
"""

from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

CQ_RX = re.compile(r"\bCQ[-A-Za-z0-9_]*\b")

def _guess_mode_from_filename(p: Path) -> Optional[str]:
    name = p.name.lower()
    if "hybrid" in name:
        return "Hybrid"
    if name in {"cqs_queries_template_filled_in_hybrid.rq"}:
        return "Hybrid"
    if "kg" in name or "graph" in name:
        return "KG"
    return None

def _read_csv_ids(
    csv_path: Path,
    id_col: Optional[str],
    mode_col: str,
    mode_filter: Optional[str]
) -> Tuple[Set[str], Dict[str,Set[str]]]:
    """
    Returns:
      (all_ids, ids_by_mode) where ids_by_mode: {"KG": set, "Hybrid": set}
    If mode_col absent, ids_by_mode will be empty.
    """
    all_ids: Set[str] = set()
    by_mode: Dict[str, Set[str]] = {"KG": set(), "Hybrid": set()}
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        # detect id column if not given
        cid_col = id_col
        if not cid_col:
            # pick the first column whose values look like "CQ-..."
            # fallbacks: common names
            candidates = [c for c in cols if c.lower() in ("cq_id","id","cq","cqid","cq-id")]
            if candidates:
                cid_col = candidates[0]
            else:
                # try all columns; stop at first row match
                sample_rows = []
                for i, row in enumerate(rdr):
                    sample_rows.append(row)
                    if i >= 20: break
                # rewind
                f.seek(0); next(f)  # skip header
                rdr = csv.DictReader(f)
                for c in cols:
                    for r in sample_rows:
                        val = (r.get(c) or "").strip()
                        if CQ_RX.search(val or ""):
                            cid_col = c
                            break
                    if cid_col:
                        break
                if not cid_col:
                    raise ValueError("Could not auto-detect CQ ID column; please pass --id_col")
        # iterate rows
        for row in rdr:
            cid = (row.get(cid_col) or "").strip()
            if not cid:
                continue
            # normalize: keep exact token that matches our CQ pattern, if present inside the cell
            m = CQ_RX.search(cid)
            if m:
                cid = m.group(0)
            all_ids.add(cid)
            # optional mode bucketing
            mode = (row.get(mode_col) or "").strip()
            if mode in by_mode:
                by_mode[mode].add(cid)
    # If mode_filter is provided, narrow all_ids accordingly
    if mode_filter in by_mode and by_mode[mode_filter]:
        all_ids = set(by_mode[mode_filter])
    return all_ids, by_mode

def _extract_cq_ids_from_rq(path: Path) -> Set[str]:
    s = path.read_text(encoding="utf-8", errors="ignore")
    return set(CQ_RX.findall(s))

def _decide_mode_for_file(path: Path, csv_modes: Dict[str,Set[str]], forced: Optional[str]) -> Optional[str]:
    if forced:
        return forced
    guess = _guess_mode_from_filename(path)
    if guess in ("KG","Hybrid"):
        return guess
    # If filename didn’t help and CSV has both modes populated, return None (use all CSV IDs)
    return None

def main():
    ap = argparse.ArgumentParser(description="Flag CSV CQ IDs that are missing from each .rq file (per file).")
    ap.add_argument("--csv", required=True, help="CSV with master list of CQ IDs")
    ap.add_argument("--rq", required=True, action="append", help="SPARQL file to check (can pass multiple)")
    ap.add_argument("--id_col", default=None, help="CSV column containing CQ IDs (auto-detected if omitted)")
    ap.add_argument("--mode_col", default="RetrievalMode", help="CSV column for mode (KG/Hybrid), if present")
    ap.add_argument("--mode_filter", default=None, choices=["KG","Hybrid"], help="Force mode subset for all files")
    ap.add_argument("--out_json", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    rq_paths = [Path(p) for p in args.rq]

    csv_all_ids, csv_by_mode = _read_csv_ids(csv_path, args.id_col, args.mode_col, args.mode_filter)

    report = {
        "csv": {
            "path": str(csv_path),
            "count_all": len(csv_all_ids),
            "count_by_mode": {k: len(v) for k,v in csv_by_mode.items()},
        },
        "files": []
    }

    for rq in rq_paths:
        rq_ids = _extract_cq_ids_from_rq(rq)
        # Determine which CSV ID pool to compare against
        mode_for_file = _decide_mode_for_file(rq, csv_by_mode, args.mode_filter)
        if mode_for_file in ("KG","Hybrid") and csv_by_mode[mode_for_file]:
            csv_ids_for_file = csv_by_mode[mode_for_file]
        else:
            csv_ids_for_file = csv_all_ids

        missing_in_rq = sorted(csv_ids_for_file - rq_ids)
        extra_in_rq   = sorted(rq_ids - csv_all_ids)

        entry = {
            "rq_path": str(rq),
            "mode": mode_for_file or "ALL",
            "csv_ids_considered": len(csv_ids_for_file),
            "rq_ids_found": len(rq_ids),
            "missing_in_rq": missing_in_rq,
            "extra_in_rq": extra_in_rq,
        }
        report["files"].append(entry)

        # Pretty print per file
        print(f"\n=== {rq.name} (mode: {entry['mode']}) ===")
        print(f"CSV IDs considered: {entry['csv_ids_considered']} | CQ IDs in .rq: {entry['rq_ids_found']}")
        print(f"- Missing in .rq (present in CSV, absent in file): {len(missing_in_rq)}")
        if missing_in_rq:
            print("  ", ", ".join(missing_in_rq))
        print(f"- Extra in .rq (present in file, absent in CSV): {len(extra_in_rq)}")
        if extra_in_rq:
            print("  ", ", ".join(extra_in_rq))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report → {args.out_json}")

if __name__ == "__main__":
    main()
