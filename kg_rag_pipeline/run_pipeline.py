# kg_rag_pipeline/run_pipeline.py
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from planner.planner import build_plan_for_persona
from retriever.retriever import retrieve
from generator.generator import generate
from utils_llm import get_llm_client
from config import load_config, deep_get
from evaluation.trace import trace_fact_alignment

# Optional baseline evaluator (unchanged)
try:
    from evaluation.evaluate_narrative import evaluate_narrative
except Exception:
    evaluate_narrative = None

# Enhanced evaluator (updated)
try:
    from evaluation.enhanced_eval import evaluate_enhanced, summarize_markdown
except Exception:
    evaluate_enhanced = None
    summarize_markdown = None


def _coverage_expectation_for_length(length: str) -> float:
    m = {
        "short": 0.60,
        "medium": 0.75,
        "long": 0.90,
    }
    return m.get((length or "short").lower(), 0.60)


def run(persona: Dict[str, Any], kg_ttl: str, out_root: str, planner_cfg: Dict[str, Any], cfg: Dict[str, Any], sys_prompt: str):
    # 0) Length → default picks per beat (only if not explicitly set)
    length_to_cfg = {
        "short": {"coverage_expectation": 0.60, "picks_per_beat": 1},
        "medium": {"coverage_expectation": 0.75, "picks_per_beat": 2},
        "long": {"coverage_expectation": 0.90, "picks_per_beat": 3},
    }
    length = (persona.get("length") or "short").lower()
    if "picks_per_beat" not in planner_cfg or planner_cfg["picks_per_beat"] is None:
        planner_cfg["picks_per_beat"] = length_to_cfg.get(length, length_to_cfg["short"])["picks_per_beat"]

    # 1) Build plan
    my_plan = build_plan_for_persona(
        persona,
        top_n=planner_cfg.get("top_n", 10),
        picks_per_beat=planner_cfg.get("picks_per_beat", None),
        seed=planner_cfg.get("seed", 42),
        normalize_embeddings=planner_cfg.get("normalize_embeddings", True),
        length_style=persona.get("length")  # <-- important
    )

    # 2) Retrieve facts (with gap top-up loop)
    length = (persona.get("length") or "short").lower()

    # length → minimum facts we want per category (tune to taste)
    min_by_length = {
        "short":  {"Entry": 2, "Core": 3, "Exit": 3},
        "medium": {"Entry": 3, "Core": 5, "Exit": 4},
        "long":   {"Entry": 4, "Core": 8, "Exit": 6},
    }
    min_required = min_by_length.get(length, min_by_length["short"])

    # helper to count retrieved facts (non-empty rows) per category
    def _facts_count_by_cat(facts_dict):
        out = {c: 0 for c in ["Entry", "Core", "Exit"]}
        for c in out:
            for it in (facts_dict.get(c, []) or []):
                if it.get("rows"):
                    out[c] += 1
        return out

    # First pass
    facts = retrieve(my_plan, kg_ttl)
    counts = _facts_count_by_cat(facts)

    # Up to 2 top-up rounds if any category is under target
    MAX_TOPUPS = 2
    rounds = 0
    while rounds < MAX_TOPUPS:
        need = {c: max(0, min_required[c] - counts.get(c, 0)) for c in ["Entry", "Core", "Exit"]}
        if all(v <= 0 for v in need.values()):
            break  # we’re good

        print(f"[INFO] Coverage top-up needed: {need}")
        # augment plan in-place
        from planner.planner import top_up_plan  # local import to avoid cycles at module load
        my_plan = top_up_plan(
            persona=persona,
            plan_obj=my_plan,
            need_by_cat=need,
            top_n=planner_cfg.get("top_n", 10),
            seed=planner_cfg.get("seed", 42),
            normalize_embeddings=planner_cfg.get("normalize_embeddings", True),
        )

        # Re-retrieve with the augmented plan
        facts = retrieve(my_plan, kg_ttl)
        counts = _facts_count_by_cat(facts)
        rounds += 1

    if rounds:
        print(f"[INFO] Top-up rounds applied: {rounds} (final counts: {counts})")


    # 3) Choose LLM client and adapt to generator's single-arg signature
    provider = (cfg.get("llm") or {}).get("provider", "ollama")
    llm_raw = get_llm_client(provider)
    llm_fn = (lambda prompt: llm_raw(sys_prompt, prompt))

    # 4) Generate
    prompt, narrative = generate(llm_fn, my_plan, facts)

    # 5) Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_root) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "plan.json").write_text(json.dumps(my_plan, indent=2, ensure_ascii=False))
    (run_dir / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))
    (run_dir / "prompt.txt").write_text(prompt)
    (run_dir / "narrative.md").write_text(narrative)

    synth_path = run_dir / "narrative_output.json"
    synth_data = {"plan": my_plan, "facts": facts, "narrative": narrative}
    synth_path.write_text(json.dumps(synth_data, indent=2, ensure_ascii=False))

    # 6) Baseline evaluation (if available)
    eval_results = None
    if evaluate_narrative:
        try:
            eval_results = evaluate_narrative(str(synth_path))
            (run_dir / "eval.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"[WARN] Baseline evaluation failed: {e}")

    # 7) Enhanced evaluation with expectation thresholds
    if evaluate_enhanced is not None:
        try:
            # Derive expectation from length (or use cfg override)
            coverage_expectation = float(deep_get(cfg, "evaluation.coverage_expectation",
                                                  length_to_cfg.get(length, length_to_cfg["short"])["coverage_expectation"]))
            enhanced = evaluate_enhanced(str(synth_path), kg_ttl, coverage_expectation)
            (run_dir / "eval_enhanced.json").write_text(json.dumps(enhanced, indent=2, ensure_ascii=False))
            if summarize_markdown:
                (run_dir / "EVAL_SUMMARY.md").write_text(summarize_markdown(enhanced))
        except Exception as e:
            print(f"[WARN] Enhanced evaluation not run: {e}")

    trace = trace_fact_alignment(narrative, facts)

    # Save artifacts (existing code)
    (run_dir / "prompt.txt").write_text(prompt)
    (run_dir / "narrative.md").write_text(narrative)

    # NEW: save trace
    (run_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False)
    )

    # 8) README summary
    md = ["# KG-RAG Run Summary\n"]
    md.append(f"**Timestamp:** {ts}")
    md.append(f"\n**Persona:** {json.dumps(persona, ensure_ascii=False)}\n")
    md.append("\n**System prompt:**\n```\n")
    md.append((sys_prompt or "").strip())
    md.append("\n```\n")
    if eval_results:
        md.append("## Baseline Evaluation\n```")
        md.append(json.dumps(eval_results, indent=2, ensure_ascii=False))
        md.append("```")
    (run_dir / "README.md").write_text("\n".join(md))

    return {"plan": my_plan, "facts": facts, "narrative": narrative, "prompt": prompt}, str(run_dir)


def main():
    ap = argparse.ArgumentParser(description="KG-RAG pipeline runner")
    ap.add_argument("--config", "-c", type=str, default="configs/pipeline_config.yaml")
    ap.add_argument("--persona", type=str, default=None)
    ap.add_argument("--tone", type=str, default=None)
    ap.add_argument("--length", type=str, default=None, choices=["short", "medium", "long"],
                    help="Single length for a single run")
    ap.add_argument("--lengths", type=str, default=None,
                    help="Comma-separated lengths to run as a suite, e.g. 'short,medium,long'")
    ap.add_argument("--kg", type=str, default=None, help="Path to KG TTL file")
    ap.add_argument("--out", type=str, default=None, help="Output runs root directory")
    ap.add_argument("--top_n", type=int, default=None)
    ap.add_argument("--picks_per_beat", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-norm", dest="no_norm", action="store_true",
                    help="Disable embedding normalization")
    ap.add_argument("--provider", type=str, default=None, choices=["ollama", "gemini"])
    ap.add_argument("--system-prompt", type=str, default=None,
                    help="Inline system prompt text")
    ap.add_argument("--system-prompt-file", type=str, default=None,
                    help="Path to a file containing the system prompt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.provider:
        cfg.setdefault("llm", {})["provider"] = args.provider

    # Resolve system prompt (file > inline > config > empty)
    if args.system_prompt_file:
        p = Path(args.system_prompt_file)
        if not p.exists():
            raise FileNotFoundError(f"--system-prompt-file not found: {p}")
        sys_prompt = p.read_text()
    elif args.system_prompt is not None:
        sys_prompt = args.system_prompt
    else:
        sys_prompt = deep_get(cfg, "llm.system_prompt", "")

    # Base persona (length may be overridden per-run)
    base_length = args.length or deep_get(cfg, "persona.length", "short")
    persona_base = {
        "name": args.persona or deep_get(cfg, "persona.name", "Emma"),
        "tone": args.tone or deep_get(cfg, "persona.tone", "educational"),
        "length": base_length,
    }

    # Planner cfg (picks_per_beat may be overridden per-run)
    planner_cfg_base = {
        "top_n": args.top_n if args.top_n is not None else deep_get(cfg, "planner.top_n", 10),
        "picks_per_beat": args.picks_per_beat if args.picks_per_beat is not None else deep_get(cfg, "planner.picks_per_beat", None),
        "seed": args.seed if args.seed is not None else deep_get(cfg, "planner.seed", 42),
        "normalize_embeddings": (False if args.no_norm else deep_get(cfg, "planner.normalize_embeddings", True)),
    }

    kg_ttl = args.kg or deep_get(cfg, "paths.kg_ttl", "data/liveaid_instances.ttl")
    out_root = Path(args.out or deep_get(cfg, "paths.out_dir", "runs"))

    # Determine suite lengths
    if args.lengths:
        lengths = [s.strip().lower() for s in args.lengths.split(",") if s.strip()]
    else:
        lengths = [base_length]

    # Run each length variant
    for ln in lengths:
        persona = dict(persona_base)
        persona["length"] = ln

        planner_cfg = dict(planner_cfg_base)
        # Default picks_per_beat from length unless explicitly given
        if planner_cfg.get("picks_per_beat") is None:
            planner_cfg["picks_per_beat"] = { "short": 1, "medium": 2, "long": 3 }.get(ln, 1)

        # Coverage expectation flows into cfg for enhanced eval
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["coverage_expectation"] = _coverage_expectation_for_length(ln)

        # Separate subfolder per length
        out_root_len = out_root / ln
        out_root_len.mkdir(parents=True, exist_ok=True)

        _, run_dir = run(persona, kg_ttl, str(out_root_len), planner_cfg, cfg, sys_prompt)
        print(f"[INFO] ({ln}) Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
