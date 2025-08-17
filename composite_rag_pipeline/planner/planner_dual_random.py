#!/usr/bin/env python3
"""
planner_dual_random.py — robust beat-aware planner for KG & Hybrid

Features
- Beat label normalization (slug) so CQs and narrative beats align.
- Per-beat sampling with strict items_per_beat per plan.
- Require SPARQL by default; optional backfill to avoid empty beats.
- Two outputs: --out_kg and --out_hybrid (flat items + beats summary).
- Validation mode to detect beat/CQ mismatches.

Inputs
- --kg_meta / --hy_meta: cq_metadata.json files produced during indexing.
  Expected shape: { "metadata": { "<CQ_ID>": { "question":..., "sparql":..., "beat":..., "beat_title":..., "mode":... } } }
  (Keys are tolerant: beat may be "beat", "Beat", "beat_title", etc.)
- --narrative_plans: JSON describing beats per persona/length. If not found,
  generates generic Beat 1..N from items_per_beat * 6 / 8 etc (see fallback).

Usage
------
python3 planner_dual_random.py \
  --kg_meta ../index/KG/cq_metadata.json \
  --hy_meta ../index/Hybrid/cq_metadata.json \
  --narrative_plans ../data/narrative_plans.json \
  --persona Emma --length Medium \
  --items_per_beat 2 --seed 42 \
  --match_strategy union \
  --out_kg plan_KG.json --out_hybrid plan_Hybrid.json \
  --validate
"""
from __future__ import annotations

import argparse, json, random, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------- utils ---------------------------------

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _slug(s: Optional[str]) -> str:
    if not s: return ""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]+", "", s)         # drop punctuation
    s = re.sub(r"\s+", "-", s)              # spaces -> dashes
    s = re.sub(r"-{2,}", "-", s).strip("-") # collapse dashes
    return s

def _meta_dict(meta_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not meta_path: return {}
    data = _load_json(meta_path) or {}
    meta = data.get("metadata") or {}
    # normalize each CQ record
    out: Dict[str, Dict[str, Any]] = {}
    for cid, rec in meta.items():
        r = dict(rec)
        # normalize beat fields -> beat_title
        beat_title = r.get("beat_title") or r.get("Beat") or r.get("beat") or r.get("beatLabel") or r.get("beat_name")
        r["beat_title"] = beat_title or ""
        r["beat_slug"] = _slug(beat_title or "")
        # normalize sparql
        r["sparql"] = (r.get("sparql") or "").strip()
        # normalize question
        r["question"] = r.get("question") or r.get("Question") or ""
        out[cid] = r
    return out

def _choose_rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed) if seed is not None else random.Random()

# -------------------------- narrative beats ----------------------------

from typing import Union

def _pick_beats(narrative_src: Optional[Union[Path, str, dict]], persona: str, length: str,
                fallback_meta: Optional[Dict[str,Dict[str,Any]]] = None,
                n_default: int = 6) -> List[Dict[str, Any]]:
    """Prefer beats from narrative_plans; else pick top beats from meta; else generic.
       Accepts a Path/str to a JSON file OR an already-parsed dict.
    """
    def _title_from(obj, i):
        if isinstance(obj, dict):
            return (obj.get("title")
                    or obj.get("beat")     # support 'beat' key too
                    or obj.get("name")
                    or f"Beat {i+1}")
        return str(obj) if obj is not None else f"Beat {i+1}"

    plans = None
    # Load if a path/str was given
    if isinstance(narrative_src, (str, Path)):
        p = Path(narrative_src)
        if p.exists():
            plans = _load_json(p)
    elif isinstance(narrative_src, dict):
        plans = narrative_src

    beats: List[Dict[str,Any]] = []
    if plans is not None:
        # shape: { persona: { length: [ {title|beat: str}, ... ] } }
        node = isinstance(plans, dict) and (plans.get(persona) or {}).get(length)
        if isinstance(node, list) and node:
            for i, b in enumerate(node):
                beats.append({"index": i, "title": _title_from(b, i)})

        # alt shape: {"plans":[{"persona":...,"length":...,"beats":[...]}]}
        if not beats and isinstance(plans, dict) and isinstance(plans.get("plans"), list):
            for plan in plans["plans"]:
                if (plan.get("persona") == persona) and (plan.get("length") == length):
                    for i, b in enumerate(plan.get("beats") or []):
                        beats.append({"index": i, "title": _title_from(b, i)})
                    break

    if not beats and fallback_meta:
        by_beat: Dict[str,int] = {}
        for r in fallback_meta.values():
            s = r.get("beat_slug") or ""
            if s: by_beat[s] = by_beat.get(s,0) + 1
        tops = [k for k,_ in sorted(by_beat.items(), key=lambda x:(-x[1], x[0]))[:n_default]]
        for i, s in enumerate(tops):
            title = next((r.get("beat_title","") for r in fallback_meta.values() if r.get("beat_slug")==s),
                         s.replace("-", " ").title())
            beats.append({"index": i, "title": title})

    if not beats:
        for i in range(n_default):
            beats.append({"index": i, "title": f"Beat {i+1}"})

    return beats



# ------------------------------ sampling -------------------------------

def _pool_for_beat(meta: Dict[str,Dict[str,Any]], beat_slug: str, require_sparql: bool) -> List[Tuple[str,Dict[str,Any]]]:
    pool = []
    for cid, rec in meta.items():
        if rec.get("beat_slug","") != beat_slug:
            continue
        if require_sparql and not rec.get("sparql"):
            continue
        pool.append((cid, rec))
    return pool

def _sample_items(rng: random.Random, pool: List[Tuple[str,Dict[str,Any]]], k: int) -> List[Tuple[str,Dict[str,Any]]]:
    if not pool:
        return []
    if len(pool) <= k:
        # deterministic-ish but shuffled
        out = list(pool)
        rng.shuffle(out)
        return out[:k]
    return rng.sample(pool, k)

def _make_item(cid: str, rec: Dict[str,Any], beat_idx: int, beat_title: str, mode: str) -> Dict[str,Any]:
    return {
        "id": cid,
        "mode": mode,
        "beat": {"index": beat_idx, "title": beat_title},
        "question": rec.get("question",""),
        "sparql": rec.get("sparql",""),
        "sparql_source": "meta",
    }

def _intersect_ids(kg_pool: List[Tuple[str,Dict[str,Any]]], hy_pool: List[Tuple[str,Dict[str,Any]]]) -> List[str]:
    kg_ids = {cid for cid,_ in kg_pool}
    hy_ids = {cid for cid,_ in hy_pool}
    return sorted(kg_ids & hy_ids)

# ------------------------------ planner --------------------------------

def build_plans(
    kg_meta: Dict[str,Dict[str,Any]],
    hy_meta: Dict[str,Dict[str,Any]],
    beats: List[Dict[str,Any]],
    items_per_beat: int,
    rng: random.Random,
    match_strategy: str = "union",
    require_sparql: bool = True,
    allow_backfill: bool = True,
) -> Tuple[Dict[str,Any], Dict[str,Any], List[str]]:
    """
    Returns (plan_kg, plan_hy, warnings)
    """
    warnings: List[str] = []
    items_kg: List[Dict[str,Any]] = []
    items_hy: List[Dict[str,Any]] = []


    for b in beats:

        i = b["index"]
        title = b["title"]
        print(f"Processing beat {b} '{title}'... ", end="", flush=True)
        bslug = _slug(title)

        kg_pool = _pool_for_beat(kg_meta, bslug, require_sparql)
        hy_pool = _pool_for_beat(hy_meta, bslug, require_sparql)

        # INTERSECT means only IDs present in both pools (often tiny/empty)
        if match_strategy == "intersect":
            ids = _intersect_ids(kg_pool, hy_pool)
            if not ids:
                warnings.append(f"[beat {i}] intersect empty for '{title}'. Will backfill.")
            # map id -> rec
            kg_by_id = {cid: rec for cid,rec in kg_pool}
            hy_by_id = {cid: rec for cid,rec in hy_pool}
            # sample from common ids
            rng.shuffle(ids)
            ids_pick = ids[:items_per_beat]
            # create items (KG & Hybrid both get same ids if available)
            picked_kg = [(cid, kg_by_id.get(cid)) for cid in ids_pick if cid in kg_by_id]
            picked_hy = [(cid, hy_by_id.get(cid)) for cid in ids_pick if cid in hy_by_id]
        else:
            # UNION / default: sample independently from each pool
            picked_kg = _sample_items(rng, kg_pool, items_per_beat)
            picked_hy = _sample_items(rng, hy_pool, items_per_beat)

        # Backfill if short
        if allow_backfill:
            if len(picked_kg) < items_per_beat:
                rest = [p for p in kg_pool if p not in picked_kg]
                picked_kg += _sample_items(rng, rest, items_per_beat - len(picked_kg))
            if len(picked_hy) < items_per_beat:
                rest = [p for p in hy_pool if p not in picked_hy]
                picked_hy += _sample_items(rng, rest, items_per_beat - len(picked_hy))

        if len(picked_kg) < items_per_beat:
            warnings.append(f"[beat {i}] KG underfilled: {len(picked_kg)}/{items_per_beat} for '{title}'")
        if len(picked_hy) < items_per_beat:
            warnings.append(f"[beat {i}] Hybrid underfilled: {len(picked_hy)}/{items_per_beat} for '{title}'")

        for cid, rec in picked_kg:
            if not rec: continue
            items_kg.append(_make_item(cid, rec, i, title, "KG"))
        for cid, rec in picked_hy:
            if not rec: continue
            items_hy.append(_make_item(cid, rec, i, title, "Hybrid"))

    plan_kg = {
        "persona": None,
        "length": None,
        "mode": "KG",
        "beats": [{"index": b["index"], "title": b["title"], "items": items_per_beat} for b in beats],
        "items": items_kg,
    }
    plan_hy = {
        "persona": None,
        "length": None,
        "mode": "Hybrid",
        "beats": [{"index": b["index"], "title": b["title"], "items": items_per_beat} for b in beats],
        "items": items_hy,
    }
    return plan_kg, plan_hy, warnings

# ------------------------------ validate --------------------------------

def validate_plan(plan: Dict[str,Any], items_per_beat: int) -> List[str]:
    errs: List[str] = []
    beats = plan.get("beats") or []
    nbeats = len(beats)
    by_idx: Dict[int,int] = {b["index"]: 0 for b in beats if isinstance(b,dict) and "index" in b}
    for it in plan.get("items") or []:
        b = it.get("beat") or {}
        bi = b.get("index", None)
        bt = b.get("title","")
        if not isinstance(bi, int):
            errs.append(f"Item {it.get('id')} missing beat.index")
            continue
        if bi < 0 or bi >= nbeats:
            errs.append(f"Item {it.get('id')} has out-of-range beat index {bi} (0..{nbeats-1})")
            continue
        # title mismatch check (optional)
        exp_title = beats[bi].get("title","")
        if exp_title and bt and _slug(exp_title) != _slug(bt):
            errs.append(f"Item {it.get('id')} beat title mismatch: item='{bt}' vs plan='{exp_title}'")
        by_idx[bi] = by_idx.get(bi,0) + 1

        # basic SPARQL presence
        if not (it.get("sparql") or "").strip():
            errs.append(f"Item {it.get('id')} in beat {bi} has empty SPARQL")

    # per-beat counts
    for b in beats:
        idx = b["index"]
        cnt = by_idx.get(idx,0)
        if cnt != items_per_beat:
            errs.append(f"Beat {idx} '{b.get('title','')}' has {cnt}/{items_per_beat} items")
    return errs

# --------------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Beat-aware random planner for KG and Hybrid.")
    ap.add_argument("--kg_meta", required=True)
    ap.add_argument("--hy_meta", required=True)
    ap.add_argument("--narrative_plans", required=False, default=None)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--length", required=True)
    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--match_strategy", choices=["union","intersect"], default="union",
                    help="union: sample independently; intersect: only common IDs (usually tiny; planner will backfill)")
    ap.add_argument("--require_sparql", action="store_true", help="Drop CQs with empty SPARQL before sampling")
    ap.add_argument("--no_backfill", action="store_true", help="Do not try to backfill short beats")
    ap.add_argument("--out_kg", required=True)
    ap.add_argument("--out_hybrid", required=True)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    rng = _choose_rng(args.seed)

    kg_meta = _meta_dict(Path(args.kg_meta))
    hy_meta = _meta_dict(Path(args.hy_meta))

    narrative = _load_json(Path(args.narrative_plans)) if args.narrative_plans else None
    beats = _pick_beats(narrative, args.persona, args.length)

    plan_kg, plan_hy, warns = build_plans(
        kg_meta, hy_meta, beats,
        items_per_beat=args.items_per_beat,
        rng=rng,
        match_strategy=args.match_strategy,
        require_sparql=args.require_sparql,
        allow_backfill=(not args.no_backfill),
    )
    # stamp persona/length
    for p in (plan_kg, plan_hy):
        p["persona"] = args.persona
        p["length"] = args.length

    # Write
    Path(args.out_kg).write_text(json.dumps(plan_kg, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_hybrid).write_text(json.dumps(plan_hy, ensure_ascii=False, indent=2), encoding="utf-8")

    # Diagnostics
    if warns:
        print("\nWarnings:")
        for w in warns:
            print(" -", w)

    if args.validate:
        print("\nValidation (KG):")
        for e in validate_plan(plan_kg, args.items_per_beat):
            print(" -", e)
        print("\nValidation (Hybrid):")
        for e in validate_plan(plan_hy, args.items_per_beat):
            print(" -", e)

    print(f"\nWrote {args.out_kg} and {args.out_hybrid}")

if __name__ == "__main__":
    main()
