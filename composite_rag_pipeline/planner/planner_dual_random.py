#!/usr/bin/env python3
# planner_dual_random.py
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# =========================
# I/O helpers
# =========================

def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================
# Normalization helpers
# =========================

def _slug(s: Any) -> str:
    if isinstance(s, (list, dict)):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = ("" if s is None else str(s)).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def _norm_title(x: Any) -> str:
    if isinstance(x, str) and x.strip():
        return x.strip()
    if isinstance(x, list):
        for v in x:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return " / ".join(str(v) for v in x if v is not None) or "Unspecified"
    if isinstance(x, dict):
        for k in ("title", "label", "name", "value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        vals = [str(v) for v in x.values() if isinstance(v, (str, int, float))]
        return " / ".join(vals) or "Unspecified"
    return "Unspecified"

def _rows_from_meta(meta_obj: Any) -> List[Dict[str, Any]]:
    base = meta_obj.get("cqs") if isinstance(meta_obj, dict) and "cqs" in meta_obj else meta_obj
    rows: List[Dict[str, Any]] = []
    if isinstance(base, list):
        for v in base:
            if isinstance(v, dict):
                rows.append(v)
    elif isinstance(base, dict):
        for k, v in base.items():
            if isinstance(v, dict):
                r = {"id": k}
                r.update(v)
                rows.append(r)
    else:
        raise ValueError("Unsupported metadata shape (expect list or dict with 'cqs').")

    for r in rows:
        bt = r.get("beat_title") or r.get("beat") or r.get("beat_slug") or r.get("title")
        r["beat_title"] = _norm_title(bt)
        sp = r.get("sparql")
        if sp is None:
            r["sparql"] = ""
        if not r.get("id"):
            r["id"] = r.get("CQ_ID") or r.get("cq_id") or r.get("CQ-ID") or ""
    return rows

def _index_by_beat(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        bt = _norm_title(r.get("beat_title", "Unspecified"))
        by.setdefault(_slug(bt), []).append(r)
    return by

# =========================
# Beat resolver (robust)
# =========================

def _resolve_beats(
    narrative_plans: Path,
    persona: str,
    length: str,
    kg_meta_path: Optional[Path] = None,
    hy_meta_path: Optional[Path] = None,
    default_n: int = 6,
) -> List[Dict[str, Any]]:
    """
    Tries in order:
      1) dict[persona][length] (case-insensitive on both keys)
      2) dict["beats"]
      3) bare list (strings or objects with title)
      4) fallback: derive beats from metadata (most frequent beat_title across KG+Hybrid)
    Returns [{"index": i, "title": "..."} ...]
    """
    def _ci_get(d: Dict[str, Any], key: str) -> Any:
        if not isinstance(d, dict):
            return None
        lk = key.lower()
        for k, v in d.items():
            if isinstance(k, str) and k.lower() == lk:
                return v
        return None

    # Load narrative spec
    try:
        spec = _read_json(narrative_plans)
    except Exception:
        spec = None

    seq = None


    # 1) persona/length nested, case-insensitive
    if isinstance(spec, dict):
        p_block = _ci_get(spec, persona)
        if isinstance(p_block, dict):
            seq = _ci_get(p_block, length)

    # 2) explicit "beats"
    if seq is None and isinstance(spec, dict):
        beats_block = _ci_get(spec, "beats")
        if isinstance(beats_block, list):
            seq = beats_block

    # 3) bare list
    if seq is None and isinstance(spec, list):
        seq = spec

    if isinstance(seq, list) and seq:
        out = []
        for i, b in enumerate(seq):
            title = _norm_title(b.get("beat") if isinstance(b, dict) else b)
            out.append({"index": i, "title": title})
        if out:
            return out

    # 4) fallback from metadata
    titles: List[str] = []
    try:
        if kg_meta_path and kg_meta_path.exists():
            kg_rows = _rows_from_meta(_read_json(kg_meta_path))
            titles += [r.get("beat_title", "Unspecified") for r in kg_rows if r.get("beat_title")]
        if hy_meta_path and hy_meta_path.exists():
            hy_rows = _rows_from_meta(_read_json(hy_meta_path))
            titles += [r.get("beat_title", "Unspecified") for r in hy_rows if r.get("beat_title")]
    except Exception:
        pass

    picked: List[str] = []
    if titles:
        freq = Counter([_norm_title(t) for t in titles])
        picked = [t for t, _ in freq.most_common(default_n)]
    else:
        # very last resort hardcoded ordering
        picked = [
            "Introduction", "Context Setup", "Performance Detail",
            "Audience Interaction", "Cultural Impact", "Legacy & Reflection"
        ][:default_n]

    return [{"index": i, "title": t} for i, t in enumerate(picked)]

# =========================
# Random selection primitives
# =========================

def _rng_sample(rng: random.Random, pool: List[Any], k: int) -> List[Any]:
    if k <= 0 or not pool:
        return []
    if k >= len(pool):
        return list(pool)
    return rng.sample(pool, k)

def _pick_for_beat_unique(
    *, rng: random.Random, k: int,
    pool_pref: List[Dict[str, Any]],
    pool_fallback: List[Dict[str, Any]],
    already: set,
) -> List[Dict[str, Any]]:
    def uniq(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            cid = r.get("id")
            sp  = (r.get("sparql") or "").strip()
            if not cid or not sp or cid in already:
                continue
            out.append(r)
        return out

    take: List[Dict[str, Any]] = []
    pref = uniq(pool_pref)
    if pref:
        take = _rng_sample(rng, pref, min(k, len(pref)))

    if len(take) < k:
        need = k - len(take)
        taken_ids = {r["id"] for r in take}
        fb = [r for r in uniq(pool_fallback) if r["id"] not in taken_ids]
        extra = _rng_sample(rng, fb, min(need, len(fb))) if fb else []
        take += extra

    for r in take:
        already.add(r["id"])
    return take

# =========================
# Plan builders
# =========================

def _plan_single_mode(
    *, mode: str, rows: List[Dict[str, Any]], beats: List[Dict[str, Any]],
    items_per_beat: int, rng: random.Random
) -> Dict[str, Any]:
    by = _index_by_beat(rows)
    global_pool = [r for r in rows if (r.get("sparql") or "").strip()]
    chosen: set = set()
    items: List[Dict[str, Any]] = []

    for b in beats:
        title = b["title"]; slug = _slug(title)
        pref = by.get(slug, [])
        picks = _pick_for_beat_unique(
            rng=rng, k=items_per_beat,
            pool_pref=pref, pool_fallback=global_pool,
            already=chosen
        )
        for r in picks:
            items.append({
                "id": r["id"],
                "question": r.get("question",""),
                "beat": {"index": b["index"], "title": title},
                "sparql": r.get("sparql",""),
            })

    if not items and global_pool:
        picks = _pick_for_beat_unique(
            rng=rng, k=min(items_per_beat, len(global_pool)),
            pool_pref=global_pool, pool_fallback=global_pool, already=chosen
        )
        for r in picks:
            items.append({
                "id": r["id"],
                "question": r.get("question",""),
                "beat": {"index": 0, "title": beats[0]["title"] if beats else "Unspecified"},
                "sparql": r.get("sparql",""),
            })

    seen, out = set(), []
    for it in items:
        cid = it.get("id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(it)

    return {"mode": mode, "beats": beats, "items": out}

def _plan_intersect(
    *, rows_kg: List[Dict[str, Any]], rows_hy: List[Dict[str, Any]],
    beats: List[Dict[str, Any]], items_per_beat: int, rng: random.Random
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by_kg = _index_by_beat(rows_kg)
    by_hy = _index_by_beat(rows_hy)
    pool_kg_all = [r for r in rows_kg if (r.get("sparql") or "").strip()]
    pool_hy_all = [r for r in rows_hy if (r.get("sparql") or "").strip()]

    chosen_kg: set = set()
    chosen_hy: set = set()
    items_kg: List[Dict[str, Any]] = []
    items_hy: List[Dict[str, Any]] = []

    def idx(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        d = {}
        for r in rows:
            cid = r.get("id")
            if cid and cid not in d:
                d[cid] = r
        return d

    idx_kg_all = idx(pool_kg_all)
    idx_hy_all = idx(pool_hy_all)

    for b in beats:
        title = b["title"]; slug = _slug(title)
        kg = [r for r in by_kg.get(slug, []) if (r.get("sparql") or "").strip()]
        hy = [r for r in by_hy.get(slug, []) if (r.get("sparql") or "").strip()]

        ids_inter = list({r["id"] for r in kg}.intersection({r["id"] for r in hy}))
        rng.shuffle(ids_inter)

        take_ids: List[str] = []
        for cid in ids_inter:
            if len(take_ids) >= items_per_beat:
                break
            if cid not in chosen_kg and cid not in chosen_hy:
                take_ids.append(cid)

        for cid in take_ids:
            rkg = idx_kg_all.get(cid); rhy = idx_hy_all.get(cid)
            if rkg and cid not in chosen_kg:
                chosen_kg.add(cid)
                items_kg.append({"id": cid, "question": rkg.get("question",""),
                                 "beat": {"index": b["index"], "title": title},
                                 "sparql": rkg.get("sparql","")})
            if rhy and cid not in chosen_hy:
                chosen_hy.add(cid)
                items_hy.append({"id": cid, "question": rhy.get("question",""),
                                 "beat": {"index": b["index"], "title": title},
                                 "sparql": rhy.get("sparql","")})

        need_kg = items_per_beat - sum(1 for it in items_kg if it["beat"]["index"] == b["index"])
        need_hy = items_per_beat - sum(1 for it in items_hy if it["beat"]["index"] == b["index"])

        if need_kg > 0:
            extra_kg = _pick_for_beat_unique(
                rng=rng, k=need_kg, pool_pref=kg, pool_fallback=pool_kg_all, already=chosen_kg
            )
            for r in extra_kg:
                items_kg.append({"id": r["id"], "question": r.get("question",""),
                                 "beat": {"index": b["index"], "title": title},
                                 "sparql": r.get("sparql","")})

        if need_hy > 0:
            extra_hy = _pick_for_beat_unique(
                rng=rng, k=need_hy, pool_pref=hy, pool_fallback=pool_hy_all, already=chosen_hy
            )
            for r in extra_hy:
                items_hy.append({"id": r["id"], "question": r.get("question",""),
                                 "beat": {"index": b["index"], "title": title},
                                 "sparql": r.get("sparql","")})

    if not items_kg:
        items_kg = _plan_single_mode(mode="KG", rows=rows_kg, beats=beats,
                                     items_per_beat=items_per_beat, rng=rng)["items"]
    if not items_hy:
        items_hy = _plan_single_mode(mode="Hybrid", rows=rows_hy, beats=beats,
                                     items_per_beat=items_per_beat, rng=rng)["items"]

    def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for it in items:
            cid = it.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            out.append(it)
        return out

    return (
        {"mode": "KG", "beats": beats, "items": dedupe(items_kg)},
        {"mode": "Hybrid", "beats": beats, "items": dedupe(items_hy)},
    )

# =========================
# Public API
# =========================

def build_plans(
    *, beats: List[Dict[str, Any]], rng: random.Random,
    kg_meta: str, hy_meta: str, items_per_beat: int = 2,
    match_strategy: str = "intersect",
    persona: Optional[str] = None, length: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows_kg = _rows_from_meta(_read_json(Path(kg_meta)))
    rows_hy = _rows_from_meta(_read_json(Path(hy_meta)))

    if match_strategy.lower() == "intersect":
        plan_kg, plan_hy = _plan_intersect(
            rows_kg=rows_kg, rows_hy=rows_hy, beats=beats,
            items_per_beat=items_per_beat, rng=rng
        )
        if not plan_kg["items"] or not plan_hy["items"]:
            plan_kg = _plan_single_mode(mode="KG", rows=rows_kg, beats=beats,
                                        items_per_beat=items_per_beat, rng=rng)
            plan_hy = _plan_single_mode(mode="Hybrid", rows=rows_hy, beats=beats,
                                         items_per_beat=items_per_beat, rng=rng)
    else:
        plan_kg = _plan_single_mode(mode="KG", rows=rows_kg, beats=beats, items_per_beat=items_per_beat, rng=rng)
        plan_hy = _plan_single_mode(mode="Hybrid", rows=rows_hy, beats=beats, items_per_beat=items_per_beat, rng=rng)

    for p in (plan_kg, plan_hy):
        p["persona"] = persona or ""
        p["length"]  = length or ""
    return plan_kg, plan_hy

# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Random dual planner (robust beat resolution; no duplicates per plan).")
    ap.add_argument("--kg_meta", type=Path, required=True, help="Path to ../index/KG/cq_metadata.json")
    ap.add_argument("--hy_meta", type=Path, required=True, help="Path to ../index/Hybrid/cq_metadata.json")
    ap.add_argument("--narrative_plans", type=Path, required=True, help="Path to narrative_plans.json")
    ap.add_argument("--persona", default="Emma")
    ap.add_argument("--length", default="Medium")
    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match_strategy", default="intersect", choices=["intersect", "independent"])
    ap.add_argument("--out_kg", type=Path, required=True)
    ap.add_argument("--out_hybrid", type=Path, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # NEW: robust beat resolve with metadata fallback
    beats = _resolve_beats(
        args.narrative_plans, args.persona, args.length,
        kg_meta_path=args.kg_meta, hy_meta_path=args.hy_meta, default_n=6
    )

    rng = random.Random(args.seed)

    plan_kg, plan_hy = build_plans(
        beats=beats, rng=rng, kg_meta=str(args.kg_meta), hy_meta=str(args.hy_meta),
        items_per_beat=args.items_per_beat, match_strategy=args.match_strategy,
        persona=args.persona, length=args.length,
    )

    _write_json(args.out_kg, plan_kg)
    _write_json(args.out_hybrid, plan_hy)

    if args.debug:
        print(f"[planner] beats: {[b['title'] for b in beats]}")
        print(f"[planner] KG items: {len(plan_kg['items'])}, Hybrid items: {len(plan_hy['items'])}")
        def per_beat(items):
            c = {}
            for it in items:
                t = it["beat"]["title"]
                c[t] = c.get(t, 0) + 1
            return c
        print("[planner] per-beat KG:", per_beat(plan_kg["items"]))
        print("[planner] per-beat Hybrid:", per_beat(plan_hy["items"]))

    print(f"✓ wrote {args.out_kg} and {args.out_hybrid} (beats resolved; no duplicates per plan)")

if __name__ == "__main__":
    main()
