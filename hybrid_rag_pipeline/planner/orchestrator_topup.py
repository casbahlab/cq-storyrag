#!/usr/bin/env python3
"""
Orchestrator Top-Up:
- Reads retriever output (items with kg_ok).
- If below target, asks a supplier for more candidates and validates them locally.
- Supplier modes:
    * faiss   → use FAISS embeddings by beat (needs cq_index.faiss + cq_metadata.json)
    * planner → call your planner script again (new seed) and use its items

Usage (faiss):
  python orchestrator_topup.py \
    --retrieved plan_with_evidence.json \
    --supplier faiss \
    --index cq_index.faiss --meta cq_metadata.json \
    --rdf kg/liveaid_instances_master.ttl kg/schema/liveaid_schema.ttl \
    --bindings bindings.json \
    --per_beat_pool 16 --require_sparql --out plan_final.json

Usage (planner):
  python orchestrator_topup.py \
    --retrieved plan_with_evidence.json \
    --supplier planner \
    --planner_cmd "python planner_with_embeddings.py --index cq_index.faiss --meta cq_metadata.json --plans narrative_plans.json --persona Emma --length Medium --limit 12 --seed 7" \
    --rdf kg/liveaid_instances_master.ttl kg/schema/liveaid_schema.ttl \
    --bindings bindings.json \
    --require_sparql --out plan_final.json
"""
import argparse, json, re, subprocess, shlex
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import faiss  # type: ignore
from ollama import embed as ollama_embed  # type: ignore
from rdflib import Graph

def _embed_query(model: str, text: str) -> np.ndarray:
    v = np.array(ollama_embed(model=model, input=[text])["embeddings"][0], dtype="float32")
    n = float(np.linalg.norm(v));  v = v/n if n>0 else v
    return v.reshape(1, -1)

def _round_robin(lists_of_idx: List[List[int]], total: int) -> List[int]:
    picks, ptrs = [], [0]*len(lists_of_idx)
    while len(picks) < total:
        progressed = False
        for i, lst in enumerate(lists_of_idx):
            if len(picks) >= total: break
            p = ptrs[i]
            if p < len(lst):
                idx = lst[p]; ptrs[i] += 1
                if idx not in picks: picks.append(idx)
                progressed = True
        if not progressed: break
    return picks

def _load_meta(meta_path: Path):
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    if "rows" in m and "order" in m:
        order = m["order"]; rows = m["rows"]
        ids       = order
        beats     = [rows[c].get("beat","") for c in order]
        texts     = [rows[c].get("text","") for c in order]
        sparqls   = [rows[c].get("sparql","") for c in order]
        questions = [rows[c].get("question","") for c in order]
        answers   = [rows[c].get("answer","") for c in order]
        model     = m.get("model","nomic-embed-text")
    else:
        ids       = m["ids"]
        beats     = m.get("beats", [""]*len(ids))
        texts     = m.get("texts", [""]*len(ids))
        sparqls   = m.get("sparqls", [""]*len(ids))
        questions = m.get("questions", [""]*len(ids))
        answers   = m.get("answers", [""]*len(ids))
        model     = m.get("model","nomic-embed-text")
    return model, ids, beats, texts, sparqls, questions, answers

def _apply_bindings(sparql: str, bindings: Dict[str, str]) -> str:
    q = sparql or ""
    for k, v in (bindings or {}).items():
        q = q.replace(f"[{k}]", v)
    return q

def _ensure_limit(q: str, n: int) -> str:
    return q if re.search(r"\blimit\s+\d+\b", q, flags=re.I) else (q.rstrip() + f"\nLIMIT {n}")

def _has_results(graph: Graph, sparql: str, bindings: Dict[str, str]) -> bool:
    if not sparql: return False
    q = _ensure_limit(_apply_bindings(sparql, bindings), 1)
    try:
        res = graph.query(q)
        return bool(getattr(res, "bindings", None)) and len(res.bindings) > 0
    except Exception:
        return False

# ---- suppliers ----
def _supply_via_faiss(beats_order: List[str], need: int, exclude_ids: List[str],
                       index_path: Path, meta_path: Path, per_beat_pool: int) -> List[Dict[str, Any]]:
    index = faiss.read_index(str(index_path))
    model, ids, beats, texts, sparqls, questions, answers = _load_meta(meta_path)

    per_beat_lists: List[List[int]] = []
    for beat in beats_order:
        qv = _embed_query(model, f"[Beat] {beat}")
        topk = min(per_beat_pool, len(ids))
        sims, idxs = index.search(qv, topk)
        lst = [int(i) for i in idxs[0].tolist() if ids[int(i)] not in exclude_ids]
        per_beat_lists.append(lst)

    order_positions = _round_robin(per_beat_lists, need * 3)
    out, seen = [], set(exclude_ids)
    for pos in order_positions:
        cid = ids[pos]
        if cid in seen: continue
        seen.add(cid)
        out.append({
            "id": cid,
            "beat": beats[pos] or "Unspecified",
            "text": texts[pos],
            "question": questions[pos],
            "answer": answers[pos],
            "sparql": sparqls[pos]
        })
        if len(out) >= need * 2: break
    return out

def _supply_via_planner(planner_cmd: str, beats_order: List[str], exclude_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Calls your planner CLI (which prints JSON to stdout).
    Returns its items, minus already used IDs. No validation here.
    """
    proc = subprocess.run(shlex.split(planner_cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[planner supplier] failed: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    out = []
    for it in data.get("items", []):
        if it.get("id") in exclude_ids: continue
        # keep only beat-aligned items
        if beats_order and it.get("beat") not in beats_order:
            continue
        out.append({
            "id": it.get("id"),
            "beat": it.get("beat") or "Unspecified",
            "text": it.get("text",""),
            "question": it.get("question",""),
            "answer": it.get("answer",""),
            "sparql": it.get("sparql","")
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieved", required=True)
    ap.add_argument("--supplier", choices=["faiss","planner"], required=True)
    # faiss supplier args
    ap.add_argument("--index", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--per_beat_pool", type=int, default=16)
    # planner supplier args
    ap.add_argument("--planner_cmd", default=None, help="Full CLI to run your planner for extra candidates")
    # KG + control
    ap.add_argument("--rdf", nargs="+", required=True)
    ap.add_argument("--bindings", default=None)
    ap.add_argument("--require_sparql", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="plan_final.json")
    args = ap.parse_args()

    data = json.loads(Path(args.retrieved).read_text(encoding="utf-8"))
    plan = {k: v for k, v in data.items() if k != "items"}  # persona/length/beats/total_limit
    beats_order = plan.get("beats", [])
    total_limit = args.limit or int(plan.get("total_limit", len(data.get("items", []))))
    good = [it for it in data.get("items", []) if it.get("kg_ok")]
    already_ids = [it["id"] for it in good]
    missing = max(0, total_limit - len(good))

    if missing > 0:
        # prepare KG + bindings
        g = Graph()
        for f in args.rdf:
            g.parse(f)
        bindings = json.loads(Path(args.bindings).read_text(encoding="utf-8")) if args.bindings else {}

        # obtain candidates
        if args.supplier == "faiss":
            if not args.index or not args.meta:
                raise SystemExit("--supplier faiss requires --index and --meta")
            cands = _supply_via_faiss(beats_order, missing, already_ids, Path(args.index), Path(args.meta), args.per_beat_pool)
        else:
            if not args.planner_cmd:
                raise SystemExit("--supplier planner requires --planner_cmd")
            cands = _supply_via_planner(args.planner_cmd, beats_order, already_ids)

        # validate candidates and top up
        for c in cands:
            if len(good) >= total_limit: break
            if args.require_sparql and not c.get("sparql"): continue
            if _has_results(g, c.get("sparql",""), bindings):
                good.append({**c, "kg_ok": True})

    out = {**plan, "items": good,
           "topup_stats": {"target": total_limit,
                           "initial_kept": len(already_ids),
                           "added": len(good) - len(already_ids)}}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"orchestrator → {args.out} | {len(good)}/{total_limit} items")

if __name__ == "__main__":
    main()
