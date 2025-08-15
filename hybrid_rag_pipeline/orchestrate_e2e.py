#!/usr/bin/env python3
"""
End-to-end orchestrator with hardcoded defaults.
Planner (method call) → Retriever (rdflib) → Top-up (planner or faiss) → Generator (md + trace)

Run with zero flags:
    python orchestrate_e2e_defaults.py

Optionally override a few knobs:
    python orchestrate_e2e_defaults.py --persona Luca --length Long --limit 9 --artist U2 --mode llm
"""

from __future__ import annotations
import argparse, json, re, sys, hashlib, datetime
from pathlib import Path
from typing import Any, Dict, List
from generator.generator_dual import preprocess_items, collect_url_evidence, render_llm_gemini, render_template, build_trace, pick_style

# ========= HARD DEFAULTS (edit these once and forget the flags) =========
DEFAULTS = {
    # Files
    "INDEX": "index/cq_index.faiss",
    "META":  "index/cq_metadata.json",
    "PLANS": "data/narrative_plans.json",
    "RDF": [
        "data/liveaid_instances_master.ttl",
        "data/schema/dublin_core_terms.ttl",
        "data/schema/musicmeta.owl",
        "data/schema/oa.ttl",
        "data/schema/schemaorg.ttl",
        "data/schema/skos.ttl",
    ],

    # Story shape
    "PERSONA": "Emma",
    "LENGTH":  "Long",
    "LIMIT":   12,            # final target (can be None to use plan default)

    # Bindings (IRI strings incl. <>)
    "event":  "ex:LiveAid1985",
    "venue":  "ex:WembleyStadium",
    "singleartist": "ex:Madonna",
    "musicgroup": "ex:Queen",
    "bandmember": "ex:BrianMay",

    # Retrieval/validation
    "REQUIRE_SPARQL": True,
    "PER_ITEM_SAMPLE": 3,    # sample rows to carry along for trace/debug

    # Top-up strategy
    #   'planner' → call planner method again for extra candidates
    #   'faiss'   → use embeddings to fetch beat-aligned candidates
    "SUPPLIER": "planner",
    "PER_BEAT_POOL": 16,     # for planner/FAISS candidate pools

    # Generator
    "MODE": "template",      # 'template' or 'llm'
    "OLLAMA_MODEL": "llama3",
    "TEMPERATURE": 0.4,

    # Outputs
    "OUT_MD":   "narrative.md",
    "OUT_JSON": "narrative_trace.json",
}
# =======================================================================

# --- imports from your pipeline (must be in PYTHONPATH) ---
try:
    from planner.planner_with_embeddings import plan as planner_plan
except Exception as e:
    print("[import] Could not import planner_with_embeddings.plan(). Make sure it's on PYTHONPATH.", file=sys.stderr)
    raise

try:
    from retriever.retriever_local_rdflib import run as retriever_run
except Exception as e:
    print("[import] Could not import retriever_local_rdflib.run().", file=sys.stderr)
    raise

try:
    from generator.generator_dual import enrich_with_rows, render_template, build_trace, render_llm_gemini
except Exception as e:
    print("[import] Could not import generator_dual helpers.", file=sys.stderr)
    raise

# Optional FAISS supplier
try:
    import faiss, numpy as np
    from ollama import embed as ollama_embed
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


# ----------------- helpers -----------------
def _apply_bindings(sparql: str, bindings: Dict[str,str]) -> str:
    q = sparql or ""
    for k, v in (bindings or {}).items():
        q = q.replace(f"[{k}]", v)
    return q

def _ensure_limit(q: str, n: int) -> str:
    return q if re.search(r"\blimit\s+\d+\b", q, flags=re.I) else (q.rstrip() + f"\nLIMIT {n}")

def _has_results(graph, sparql: str, bindings: Dict[str,str]) -> bool:
    if not sparql: return False
    q = _ensure_limit(_apply_bindings(sparql, bindings), 1)
    try:
        res = graph.query(q)
        return bool(getattr(res, "bindings", None)) and len(res.bindings) > 0
    except Exception:
        return False

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

def _embed_query(model: str, text: str):
    v = np.array(ollama_embed(model=model, input=[text])["embeddings"][0], dtype="float32")
    n = float(np.linalg.norm(v));  v = v/n if n>0 else v
    return v.reshape(1, -1)

def _supply_via_faiss(beats_order: List[str], need: int, exclude_ids: List[str],
                      index_path: Path, meta_path: Path, per_beat_pool: int) -> List[Dict[str, Any]]:
    if not HAS_FAISS:
        raise RuntimeError("FAISS/Ollama not available. Install faiss-cpu and ollama or switch SUPPLIER='planner'.")
    index = faiss.read_index(str(index_path))
    model, ids, beats, texts, sparqls, questions, answers = _load_meta(meta_path)

    def round_robin(lists_of_idx: List[List[int]], total: int) -> List[int]:
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

    per_beat_lists: List[List[int]] = []
    for beat in beats_order:
        qv = _embed_query(model, f"[Beat] {beat}")
        topk = min(per_beat_pool, len(ids))
        sims, idxs = index.search(qv, topk)
        lst = [int(i) for i in idxs[0].tolist() if ids[int(i)] not in exclude_ids]
        per_beat_lists.append(lst)

    order_positions = round_robin(per_beat_lists, need * 3)
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

def _topup_with_planner(orig_plan: Dict[str, Any], added_limit: int,
                        index_path: str, meta_path: str, plans_path: str,
                        per_beat_pool: int) -> List[Dict[str, Any]]:
    """Call the planner method again to fetch extra candidates."""
    p = planner_plan(index_path, meta_path, plans_path,
                     orig_plan["persona"], orig_plan["length"],
                     limit=added_limit, per_beat_pool=per_beat_pool)
    return p.get("items", [])

from datetime import datetime  # (already imported in your file)

def _make_run_dir(run_dir: str | None, base: str = "runs") -> Path:
    """Create and return a timestamped run directory if run_dir not provided."""
    if run_dir:
        p = Path(run_dir)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        p = Path(base) / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="End-to-end orchestrator (hard defaults).")
    ap.add_argument("--persona", default=DEFAULTS["PERSONA"])
    ap.add_argument("--length",  default=DEFAULTS["LENGTH"])
    ap.add_argument("--limit",   type=int, default=DEFAULTS["LIMIT"])
    ap.add_argument("--event",   default=None, help="Override Event local name (e.g., LiveAid1985)")
    ap.add_argument("--venue",   default=None, help="Override Venue local name (e.g., WembleyStadium)")
    ap.add_argument("--artist",  default=None, help="Override Artist local name (e.g., Queen)")
    ap.add_argument("--mode",    choices=["template","llm"], default=DEFAULTS["MODE"])
    ap.add_argument("--run_dir", default=None,
                    help="Base directory to store this run's outputs (default: runs/<UTC timestamp>).")

    args = ap.parse_args()

    # Resolve paths
    INDEX = Path(DEFAULTS["INDEX"]).resolve()
    META  = Path(DEFAULTS["META"]).resolve()
    PLANS = Path(DEFAULTS["PLANS"]).resolve()
    RDFS  = [Path(p).resolve() for p in DEFAULTS["RDF"]]

    # Bindings (allow quick overrides via local names)
    bindings = {"event": DEFAULTS["event"],"venue": DEFAULTS["venue"],"singleartist": DEFAULTS["singleartist"],"musicgroup": DEFAULTS["musicgroup"],"bandmember": DEFAULTS["bandmember"]}

    # 1) PLAN
    plan = planner_plan(str(INDEX), str(META), str(PLANS),
                        args.persona, args.length,
                        limit=args.limit, per_beat_pool=DEFAULTS["PER_BEAT_POOL"])

    # 2) RETRIEVE (local rdflib)
    plan_evid = retriever_run(
        plan,
        rdf_files=[str(p) for p in RDFS],
        bindings=bindings,
        per_item_sample=DEFAULTS["PER_ITEM_SAMPLE"],
        require_sparql=DEFAULTS["REQUIRE_SPARQL"]
    )

    # 3) TOP-UP (only if needed)
    items = plan_evid["items"]
    target = int(plan.get("total_limit", len(items)))
    kept = [it for it in items if it.get("kg_ok")]
    need = max(0, target - len(kept))

    if need > 0:
        # load KG for validation
        from rdflib import Graph
        g = Graph()
        for f in RDFS:
            g.parse(str(f))

        already_ids = {it["id"] for it in kept} | {it["id"] for it in items}
        if DEFAULTS["SUPPLIER"] == "faiss":
            cands = _supply_via_faiss(plan["beats"], need, list(already_ids),
                                      INDEX, META, DEFAULTS["PER_BEAT_POOL"])
        else:
            # planner supplier
            extra = _topup_with_planner(plan, added_limit=target*2,
                                        index_path=str(INDEX), meta_path=str(META), plans_path=str(PLANS),
                                        per_beat_pool=DEFAULTS["PER_BEAT_POOL"])
            cands = [c for c in extra if c["id"] not in already_ids]

        for c in cands:
            if len(kept) >= target: break
            if DEFAULTS["REQUIRE_SPARQL"] and not c.get("sparql"): continue
            if _has_results(g, c.get("sparql",""), bindings):
                kept.append({**c, "kg_ok": True})

        final_plan = {**plan, "items": kept, "topup_stats": {
            "target": target,
            "initial_kept": len([it for it in items if it.get("kg_ok")]),
            "added": max(0, len(kept) - len([it for it in items if it.get("kg_ok")]))
        }}
    else:
        final_plan = {**plan, "items": kept, "topup_stats": {"target": target, "initial_kept": len(kept), "added": 0}}

    # final_plan = plan_evid

    # Build prompt object (LLM mode) or None (template mode)
    prompt_obj = None

    # match generator_dual.render_llm prompt
    from generator.generator_dual import pick_style, target_sentence_hint
    style = pick_style(final_plan.get("persona") or "")
    beats_order = final_plan.get("beats", [])
    by_beat = {b: [] for b in beats_order}
    for it in final_plan.get("items", []):
        b = it.get("beat", "Unspecified")
        if b in by_beat:
            by_beat[b].append({
                "q": it.get("question") or it.get("text") or "",
                "a": it.get("answer") or "",
                "rows": it.get("rows") or []
            })
    sys_msg = (
        f"You are a narrative generator.\n"
        f"Persona voice: {style['voice']}\n"
        f"Rhythm: {style['rhythm']}\n"
        f"Write in clear UK English.\n"
        f"Use {target_sentence_hint(final_plan.get('length') or '')}.\n"
        f"Ground claims in the provided answers; if data rows exist, weave 1–2 precise details.\n"
        f"Do NOT invent facts beyond what's provided.\n"
        f"Avoid bullet points unless data demands it."
    )
    user_msg = (
            f'Create a cohesive narrative for persona "{final_plan.get("persona")}" '
            f'and length "{final_plan.get("length")}". '
            f"Organise by beats in this exact order: {beats_order}.\n"
            f"Content JSON per beat with Q/A and optional data rows:\n"
            + json.dumps({"beats_order": beats_order, "content": by_beat}, ensure_ascii=False)
    )
    prompt_obj = {
        "model": DEFAULTS["OLLAMA_MODEL"],
        "temperature": DEFAULTS["TEMPERATURE"],
        "system": sys_msg,
        "user": user_msg
    }

    # # Optional: enrich rows again (visible in output narrative foldouts)
    # enrich_with_rows(final_plan["items"], [str(p) for p in RDFS], bindings,
    #                  per_item_sample=5, include_executed_query=True)

    preprocess_items(final_plan)
    evidence = collect_url_evidence(final_plan, max_docs=18, max_chars_per_doc=10000)
    md = render_llm_gemini(final_plan, model="gemini-2.5-flash", temperature=DEFAULTS["TEMPERATURE"], evidence_docs=evidence, save_prompt_path = "prompt.json")

    Path(DEFAULTS["OUT_MD"]).write_text(md, encoding="utf-8")
    trace = build_trace(final_plan, mode=args.mode, rdf_files=[str(p) for p in RDFS], bindings=bindings, include_executed_query=True)
    Path(DEFAULTS["OUT_JSON"]).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f" - narrative: {DEFAULTS['OUT_MD']}")
    print(f" - trace:     {DEFAULTS['OUT_JSON']}")
    print(f" - persona/length: {args.persona}/{args.length}")
    print(f" - bindings: event={bindings['event']} venue={bindings['venue']} singleartist={bindings['singleartist']} musicgroup={bindings['musicgroup']} bandmember={bindings['bandmember']}")

    from trace_pack import save_all

    run_dir = "run_trace"
    kg_files = [str(p) for p in RDFS]
    # plan            → initial (from planner)
    # plan_evid       → retrieved (after rdflib execution)
    # final_plan      → post top-up
    save_all(plan, plan_evid, final_plan, md_text=md, out_dir=run_dir, kg_files=kg_files, prompt=prompt_obj)

    print(f"trace bundle → {run_dir}/ "
          "narrative_output.json, trace.json, EVAL_SUMMARY.md, narrative.md")

if __name__ == "__main__":
    main()
