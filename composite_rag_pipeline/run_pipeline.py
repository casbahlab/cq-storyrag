#!/usr/bin/env python3
"""
run_pipeline.py — One-shot runner for:
  1) planner_dual_random.py     → plan_KG.json, plan_Hybrid.json
  2) retriever_local_rdflib.py  → plan_with_evidence_{KG,Hybrid}.json (+ evidence .jsonl)
  3) generator_dual.py          → story_{KG,Hybrid}.md (+ answers .jsonl)

It mirrors the exact commands you’ve been running by hand.
"""

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# ---------- helpers ----------

def _p(*parts: str) -> Path:
    return Path(*parts).resolve()

def run(cmd: list[str], cwd: Path | None = None):
    print("\n▶ " + (" ".join(shlex.quote(c) for c in cmd)))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def must_exist(path: Path, label: str = "file"):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="End-to-end KG + Hybrid pipeline runner")
    ap.add_argument("--persona", default="Emma")
    ap.add_argument("--length", default="Medium")
    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match_strategy", choices=["union","intersect"], default="intersect")

    # LLM
    ap.add_argument("--llm_provider", default="ollama")
    ap.add_argument("--llm_model", default="llama3.1-128k")

    # Repo layout (assumes this script lives in repo root)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--planner_dir", default="planner")
    ap.add_argument("--retriever_dir", default="retriever")
    ap.add_argument("--generator_dir", default="generator")

    # Inputs
    ap.add_argument("--kg_meta", default="index/KG/cq_metadata.json")
    ap.add_argument("--hy_meta", default="index/Hybrid/cq_metadata.json")
    ap.add_argument("--narrative_plans", default="data/narrative_plans.json")
    ap.add_argument("--rdf", default="data/liveaid_instances_master.ttl")
    ap.add_argument("--params", default="params.json")

    # Tuning
    ap.add_argument("--per_item_sample", type=int, default=5)
    ap.add_argument("--timeout_s", type=int, default=10)
    ap.add_argument("--url_timeout_s", type=int, default=5)
    ap.add_argument("--max_urls_per_item", type=int, default=5)
    ap.add_argument("--content_max_chars", type=int, default=240)
    ap.add_argument("--max_url_snippets", type=int, default=2)
    ap.add_argument("--snippet_chars", type=int, default=500)

    args = ap.parse_args()

    PY = sys.executable
    ROOT = _p(args.root)
    planner_dir = _p(ROOT, args.planner_dir)
    retriever_dir = _p(ROOT, args.retriever_dir)
    generator_dir = _p(ROOT, args.generator_dir)

    kg_meta = _p(ROOT, args.kg_meta)
    hy_meta = _p(ROOT, args.hy_meta)
    narrative = _p(ROOT, args.narrative_plans)
    rdf_file = _p(ROOT, args.rdf)
    params_json = _p(ROOT, args.params)

    # Outputs
    plan_kg = _p(planner_dir, "plan_KG.json")
    plan_hy = _p(planner_dir, "plan_Hybrid.json")

    retr_log_dir = _p(retriever_dir, "run_trace", "logs")
    retr_errors = _p(retriever_dir, "run_trace", "retriever.jsonl")

    plan_evid_kg = _p(retriever_dir, "plan_with_evidence_KG.json")
    plan_evid_hy = _p(retriever_dir, "plan_with_evidence_Hybrid.json")
    evid_kg = _p(retriever_dir, "evidence_KG.jsonl")
    evid_hy = _p(retriever_dir, "evidence_Hybrid.jsonl")

    ans_kg = _p(generator_dir, "answers_KG.jsonl")
    ans_hy = _p(generator_dir, "answers_Hybrid.jsonl")
    story_kg = _p(generator_dir, "story_KG.md")
    story_hy = _p(generator_dir, "story_Hybrid.md")

    # Pre-flight
    must_exist(planner_dir, "planner_dir")
    must_exist(retriever_dir, "retriever_dir")
    must_exist(generator_dir, "generator_dir")
    must_exist(kg_meta, "KG meta")
    must_exist(hy_meta, "Hybrid meta")
    must_exist(narrative, "narrative_plans")
    must_exist(rdf_file, "RDF")
    if not params_json.exists():
        print(f"⚠️  params.json not found at {params_json} (continuing)")

    retr_log_dir.mkdir(parents=True, exist_ok=True)

    # 1) Planner (KG + Hybrid)
    run([
        PY, str(_p(planner_dir, "planner_dual_random.py")),
        "--kg_meta", str(kg_meta),
        "--hy_meta", str(hy_meta),
        "--narrative_plans", str(narrative),
        "--persona", args.persona,
        "--length", args.length,
        "--items_per_beat", str(args.items_per_beat),
        "--seed", str(args.seed),
        "--match_strategy", args.match_strategy,
        "--out_kg", str(plan_kg.name),
        "--out_hybrid", str(plan_hy.name),
    ], cwd=planner_dir)
    print(f"✅ Planner → {plan_kg} | {plan_hy}")

    # 2) Retriever (KG)
    run([
        PY, str(_p(retriever_dir, "retriever_local_rdflib.py")),
        "--mode", "KG",
        "--plan", str(plan_kg),
        "--meta", str(kg_meta),
        "--rdf", str(rdf_file),
        "--bindings", str(params_json),
        "--require_sparql",
        "--per_item_sample", str(args.per_item_sample),
        "--timeout_s", str(args.timeout_s),
        "--log_dir", str(retr_log_dir),
        "--errors_jsonl", str(retr_errors),
        "--evidence_out", str(evid_kg.name),
        "--content_max_chars", str(args.content_max_chars),
        "--out", str(plan_evid_kg.name),
    ], cwd=retriever_dir)
    print(f"✅ Retriever KG → {plan_evid_kg} | {evid_kg}")

    # 3) Retriever (Hybrid)
    run([
        PY, str(_p(retriever_dir, "retriever_local_rdflib.py")),
        "--mode", "Hybrid",
        "--plan", str(plan_hy),
        "--meta", str(hy_meta),
        "--rdf", str(rdf_file),
        "--bindings", str(params_json),
        "--require_sparql",
        "--per_item_sample", str(args.per_item_sample),
        "--timeout_s", str(args.timeout_s),
        "--hy_enrich_labels",
        "--hy_enrich_neighbors", "8",
        "--hy_enrich_incoming", "4",
        "--enrich_urls",
        "--fetch_url_content",
        "--url_timeout_s", str(args.url_timeout_s),
        "--max_urls_per_item", str(args.max_urls_per_item),
        "--log_dir", str(retr_log_dir),
        "--errors_jsonl", str(retr_errors),
        "--evidence_out", str(evid_hy.name),
        "--content_max_chars", str(args.content_max_chars),
        "--out", str(plan_evid_hy.name),
    ], cwd=retriever_dir)
    print(f"Retriever Hybrid → {plan_evid_hy} | {evid_hy}")

    # 4) Generator (KG)
    run([
        PY, str(_p(generator_dir, "generator_dual.py")),
        "--plan", str(plan_kg),
        "--plan_with_evidence", str(plan_evid_kg),
        "--kg_meta", str(kg_meta),
        "--params", str(params_json),
        "--llm_provider", args.llm_provider,
        "--llm_model", args.llm_model,
        "--out", str(ans_kg.name),
        "--story_out", str(story_kg.name),
        "--include_citations",
    ], cwd=generator_dir)
    print(f"Story KG → {story_kg}")

    # 5) Generator (Hybrid)
    run([
        PY, str(_p(generator_dir, "generator_dual.py")),
        "--plan", str(plan_hy),
        "--plan_with_evidence", str(plan_evid_hy),
        "--hy_meta", str(hy_meta),
        "--params", str(params_json),
        "--llm_provider", args.llm_provider,
        "--llm_model", args.llm_model,
        "--use_url_content",
        "--max_url_snippets", str(args.max_url_snippets),
        "--snippet_chars", str(args.snippet_chars),
        "--out", str(ans_hy.name),
        "--story_out", str(story_hy.name),
        "--include_citations",
    ], cwd=generator_dir)
    print(f"Story Hybrid → {story_hy}")

    print("\nDone! Stories are ready:")
    print(f"   - {story_kg}")
    print(f"   - {story_hy}")

if __name__ == "__main__":
    main()
