#!/usr/bin/env python3
"""
planner_random.py — Fixed-plan random selector (KG / Hybrid)

- Reads a narrative plan file (persona -> length -> list of beats).
- For each beat, randomly selects N CQs from the given mode's index metadata.
- Ignores embeddings/FAISS; just samples uniformly at random.
- Ensures unique picks across the whole plan unless --allow_repeats is set.

Input files
-----------
- narrative_plans.json  (structure: { persona: { length: [ {step, beat}, ... ] } })
- cq_metadata.json      (from build_cq_index_v2.py; includes retrieval_mode, question, answer, sparql)

Usage
-----
python3 planner_random.py \
  --meta ../index/KG/cq_metadata.json \
  --narrative_plans ../planner/narrative_plans.json \
  --persona Emma --length Medium --mode KG \
  --items_per_beat 2 \
  --seed 42 > plan_KG.json

python3 planner_random.py \
  --meta ../index/Hybrid/cq_metadata.json \
  --narrative_plans ../planner/narrative_plans.json \
  --persona Emma --length Medium --mode Hybrid \
  --items_per_beat 2 --seed 42 > plan_Hybrid.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

def load_meta(meta_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    order = meta.get("order") or []
    rows  = meta.get("metadata") or {}
    return order, rows

def load_plan(narr_path: Path, persona: str, length: str) -> List[Dict[str, Any]]:
    narr = json.loads(narr_path.read_text(encoding="utf-8"))
    if persona not in narr:
        raise SystemExit(f"Persona '{persona}' not found in narrative_plans.json")
    if length not in narr[persona]:
        raise SystemExit(f"Length '{length}' not found for persona '{persona}' in narrative_plans.json")
    # list of {step, beat}
    beats = narr[persona][length]
    # convert to simple list of dicts with title only; items count applied later
    return [{"title": b["beat"]} for b in beats]

def choose_random(ids: List[str], k: int, rng: np.random.Generator, allow_repeats: bool) -> List[int]:
    n = len(ids)
    if n == 0:
        return []
    if allow_repeats:
        return rng.integers(0, n, size=k).tolist()
    # without replacement
    k = min(k, n)
    return rng.choice(n, size=k, replace=False).tolist()

def main():
    ap = argparse.ArgumentParser(description="Random CQ planner with fixed beats plan")
    ap.add_argument("--meta", required=True, help="Path to cq_metadata.json")
    ap.add_argument("--narrative_plans", required=True, help="Path to narrative_plans.json (fixed beats)")
    ap.add_argument("--persona", required=True)
    ap.add_argument("--length", required=True, choices=["Short","Medium","Long"])
    ap.add_argument("--mode", required=True, choices=["KG","Hybrid"])

    ap.add_argument("--items_per_beat", type=int, default=2, help="How many items to sample per beat")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--allow_repeats", action="store_true", help="Allow the same CQ to appear in multiple beats")

    args = ap.parse_args()

    # Load metadata + filter by mode
    order, rows = load_meta(Path(args.meta))

    eligible_ids = []
    for cid in order:
        rec = rows.get(cid, {})
        modes = rec.get("retrieval_mode") or []
        # include if mode matches or if metadata doesn't specify modes (fallback to include)
        include = (args.mode in {m.strip() for m in modes}) or (not modes)
        if include:
            eligible_ids.append(cid)

    if not eligible_ids:
        raise SystemExit(f"No eligible CQs found for mode '{args.mode}'.")

    # Load fixed beats plan
    beats = load_plan(Path(args.narrative_plans), args.persona, args.length)
    rng = np.random.default_rng(args.seed)

    # Choose items per beat
    chosen_global = set()
    items = []
    for b_idx, beat in enumerate(beats):
        # sample indices into eligible_ids
        idxs = choose_random(eligible_ids, args.items_per_beat, rng, allow_repeats=args.allow_repeats)
        for idx in idxs:
            cid = eligible_ids[idx]
            if not args.allow_repeats and cid in chosen_global:
                # try to find a replacement quickly
                for _ in range(10):
                    j = int(rng.integers(0, len(eligible_ids)))
                    cand = eligible_ids[j]
                    if cand not in chosen_global:
                        cid = cand
                        break
            chosen_global.add(cid)
            rec = rows[cid]
            items.append({
                "id": cid,
                "beat_index": b_idx,
                "beat_title": beat["title"],
                "question": rec.get("question",""),
                "answer": rec.get("answer",""),
                "sparql": rec.get("sparql","")
            })

        # store per-beat item count for output
        beat["items"] = args.items_per_beat

    out = {
        "persona": args.persona,
        "length": args.length,
        "mode": args.mode,
        "beats": beats,
        "items": items
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
