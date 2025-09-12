#!/usr/bin/env python3
"""
Freeze the current master KG (liveaid_instances_master.ttl) into a versioned snapshot.
Also bundles coverage outputs if present.

Usage:
  python kg/scripts/freeze_master_snapshot.py --minor   # default
  python kg/scripts/freeze_master_snapshot.py --major
  python kg/scripts/freeze_master_snapshot.py --patch
Options:
  --label "notes about this snapshot"    # optional free-text label (saved alongside)
"""

import argparse
import re
from pathlib import Path
import shutil
import time
import json

KG_DIR = Path(__file__).resolve().parents[1]
MASTER = KG_DIR / "liveaid_instances_master.ttl"
SNAP_DIR = KG_DIR / "kg_snapshots"
COVERAGE_SUMMARY = Path("/mnt/data/coverage_summary.csv")  # adjust if you keep it elsewhere
CQS_OUT = KG_DIR / "cqs" / "out"

SNAP_DIR.mkdir(parents=True, exist_ok=True)

VER_RE = re.compile(r"liveaid_instances_master_v(\d+)\.(\d+)\.ttl$")

def find_latest_version():
    majors, minors = [], []
    for p in SNAP_DIR.glob("liveaid_instances_master_v*.ttl"):
        m = VER_RE.match(p.name)
        if m:
            majors.append(int(m.group(1)))
            minors.append(int(m.group(2)))
    if not majors:
        return (0, 9)  # so that +minor starts at v1.0
    # find max by (major, minor)
    pairs = list(zip(majors, minors))
    latest = max(pairs)
    return latest

def bump(major, minor, mode):
    if mode == "major":
        return (major + 1, 0)
    if mode == "patch":
        # optional: patch level; here we just increment minor as a simple scheme
        return (major, minor + 1)
    # default minor
    return (major + 1, 0) if (major, minor) == (0, 9) else (major, minor + 1)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--major", action="store_true", help="bump major version")
    g.add_argument("--minor", action="store_true", help="bump minor version (default)")
    g.add_argument("--patch", action="store_true", help="bump patch (treated same as minor here)")
    ap.add_argument("--label", type=str, default="", help="Optional free-text label for this snapshot")
    args = ap.parse_args()

    if not MASTER.exists():
        raise SystemExit(f"[ERR] Master KG not found: {MASTER}")

    mode = "minor"
    if args.major: mode = "major"
    elif args.patch: mode = "patch"

    cur_major, cur_minor = find_latest_version()
    new_major, new_minor = bump(cur_major, cur_minor, mode)
    ver = f"v{new_major}.{new_minor}"

    # Copy master
    snap_ttl = SNAP_DIR / f"liveaid_instances_master_{ver}.ttl"
    shutil.copy2(MASTER, snap_ttl)

    # Bundle coverage summary + CQ CSVs if present
    bundle_dir = SNAP_DIR / f"bundle_{ver}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if COVERAGE_SUMMARY.exists():
        shutil.copy2(COVERAGE_SUMMARY, bundle_dir / f"coverage_summary_{ver}.csv")
    if CQS_OUT.exists():
        dest = bundle_dir / f"cqs_results_{ver}"
        dest.mkdir(parents=True, exist_ok=True)
        for csv in CQS_OUT.glob("*.csv"):
            shutil.copy2(csv, dest / csv.name)

    # Write a small JSON manifest for traceability
    manifest = {
        "version": ver,
        "timestamp": int(time.time()),
        "master_snapshot": str(snap_ttl),
        "coverage_summary": str(bundle_dir / f"coverage_summary_{ver}.csv") if COVERAGE_SUMMARY.exists() else None,
        "cqs_results_dir": str(bundle_dir / f"cqs_results_{ver}") if CQS_OUT.exists() else None,
        "label": args.label or ""
    }
    with open(bundle_dir / f"manifest_{ver}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Snapshot created: {snap_ttl.name}")
    print(f"[OK] Bundle: {bundle_dir}")
    if args.label:
        print(f"[OK] Label: {args.label}")

if __name__ == "__main__":
    main()
