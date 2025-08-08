import json
from datetime import datetime
from pathlib import Path

from generator.generator import generate
from planner.planner import build_plan_for_persona
from retriever.retriever import retrieve
from utils_llm import ollama_chat

# Optional: import evaluator if available
try:
    from evaluation.evaluate_narrative import evaluate_narrative
except Exception:
    evaluate_narrative = None

def run(persona: dict, kg_ttl="data/liveaid_instances.ttl", out_dir: str = None):
    # Build plan -> retrieve facts -> generate narrative
    my_plan = build_plan_for_persona(persona)
    facts = retrieve(my_plan, kg_ttl)
    prompt, narrative = generate(ollama_chat, my_plan, facts)

    out = {"plan": my_plan, "facts": facts, "narrative": narrative, "prompt": prompt}

    # Create run folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_dir or f"runs/{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    (run_dir / "plan.json").write_text(json.dumps(my_plan, indent=2, ensure_ascii=False))
    (run_dir / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))
    (run_dir / "prompt.txt").write_text(prompt)
    (run_dir / "narrative.md").write_text(narrative)

    # Always save narrative_output.json (so enhanced eval can use it later)
    synth_path = run_dir / "narrative_output.json"
    synth_data = {"plan": my_plan, "facts": facts, "narrative": narrative}
    synth_path.write_text(json.dumps(synth_data, indent=2, ensure_ascii=False))

    # Run baseline evaluation
    eval_results = None
    if evaluate_narrative:
        eval_results = evaluate_narrative(str(synth_path))
        (run_dir / "eval.json").write_text(json.dumps(eval_results, indent=2, ensure_ascii=False))

    # Run enhanced evaluation
    try:
        from evaluation.enhanced_eval import evaluate_enhanced, summarize_markdown
        enhanced_results = evaluate_enhanced(str(synth_path))
        (run_dir / "eval_enhanced.json").write_text(json.dumps(enhanced_results, indent=2, ensure_ascii=False))
        (run_dir / "EVAL_SUMMARY.md").write_text(summarize_markdown(enhanced_results))
    except Exception as e:
        print(f"[WARN] Enhanced evaluation not run: {e}")

    # Markdown summary for README.md
    md = ["# KG-RAG Run Summary\n"]
    md.append(f"**Timestamp:** {ts}")
    md.append(f"\n**Persona:** {persona}\n")
    if eval_results:
        md.append("## Baseline Evaluation\n```")
        md.append(json.dumps(eval_results, indent=2, ensure_ascii=False))
        md.append("```")
    if 'enhanced_results' in locals():
        md.append("\n## Enhanced Evaluation Summary\n")
        md.append("(See `EVAL_SUMMARY.md` for full details)\n")
        md.append("```")
        md.append(json.dumps(enhanced_results.get("summary", {}), indent=2, ensure_ascii=False))
        md.append("```")
    (run_dir / "README.md").write_text("\n".join(md))

    return out, str(run_dir)


if __name__ == "__main__":
    # Minimal example persona; adjust in your CLI or config
    persona = {"name": "Emma", "tone": "educational", "length": "short"}
    out, run_dir = run(persona)
    print(f"[INFO] Run artifacts saved to: {run_dir}")

    # Enhanced evaluation
    try:
        from evaluation.enhanced_eval import evaluate_enhanced, summarize_markdown

        synth_path = Path(run_dir) / "narrative_output.json"
        enhanced = evaluate_enhanced(str(synth_path))
        (Path(run_dir)  / "eval_enhanced.json").write_text(json.dumps(enhanced, indent=2, ensure_ascii=False))
        (Path(run_dir)  / "EVAL_SUMMARY.md").write_text(summarize_markdown(enhanced))
    except Exception as e:
        print(f"[WARN] Enhanced evaluation not run: {e}")


