#!/usr/bin/env python3
# generator_graph.py — dedicated graph-led story generator (no CQ dependency)
#
# Consumes: plan_with_evidence_Graph.json (built from the graph adapter)
# Produces: answers_Graph.jsonl, story_Graph.md, story_Graph_clean.md
#
# Providers: ollama (default) or gemini (requires GOOGLE_API_KEY)

from __future__ import annotations
import argparse, json, os, re, subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------- LLM provider shims -----------------------

def _call_ollama(model: str, prompt: str, num_ctx: int | None = None) -> str:
    args = ["ollama", "run"]
    if num_ctx is not None:
        args += ["--num-ctx", str(num_ctx)]
    args += [model, "--prompt", "-"]
    out = subprocess.check_output(args, input=prompt.encode("utf-8"), text=True)
    return out.strip()

def _call_gemini(model: str, prompt: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        raise RuntimeError("Gemini provider selected but 'google-generativeai' package is not installed.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini provider selected but GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    r = model_obj.generate_content(prompt)
    return (getattr(r, "text", None) or "").strip()

def call_model(provider: str, model: str, prompt: str, **kw) -> str:
    if provider == "ollama":
        return _call_ollama(model, prompt, kw.get("ollama_num_ctx"))
    if provider == "gemini":
        return _call_gemini(model, prompt)
    raise ValueError(f"Unknown provider: {provider}")

# ----------------------- IO helpers -----------------------

def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _strip_meta(text: str) -> str:
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            lines.append("")
            continue
        if s.lower().startswith(("here is", "here's", "section:", "answer:", "story:")):
            continue
        lines.append(raw)
    out = "\n".join(lines).strip()
    out = re.sub(r"```[a-zA-Z]*\n?|```", "", out).strip()
    return out

def _clean_story_text_remove_headings_and_citations(text: str) -> str:
    out = []
    for line in (text or "").splitlines():
        if line.strip().startswith("#"):
            continue
        # strip [1], [2], ... and [CQ-...]
        line = re.sub(r"\[(?:\d+(?:\s*,\s*\d+)*)\]", "", line)
        line = re.sub(r"\[CQ-[^\]]+\]", "", line)
        out.append(line.rstrip())
    return "\n".join(out).strip()

# ----------------------- Prompt building -----------------------

def _fmt_facts(facts: List[str], limit: int) -> str:
    return "\n".join("- " + f for f in facts[:limit])

def build_section_prompt(persona, topic, beat_title, summary, facts, sentences):
    persona_name  = persona.get("name", "")
    persona_level = persona.get("level", "")
    persona_goals = ", ".join(persona.get("goals", []) or [])
    fact_lines = "\n".join(f"- {f}" for f in facts[:60])
    return f"""You are a narrative writer.

Audience:
- name: {persona_name}
- expertise: {persona_level}
- goals: {persona_goals}

Topic: {topic}
Section: {beat_title}

CONTEXT (use only this):
[Graph Summary]
{summary}

[Key Relations: subject | predicate | object]
{fact_lines}

Write a cohesive prose section for this Section using only the context above.
Rules:
- About {sentences} sentences; plain paragraphs only (no bullets, no headings, no lists).
- Use concrete names from the context; do **not** print raw IDs or URIs; use display names only.
- Preserve the direction of relations (subject → predicate → object) in what you write.
- Prefer connective tissue (because, so, therefore, as a result) to link ideas.
- If multiple songs or members appear, narrate them in a clear, logical order.
- Do not mention “graph”, “summary”, “relations”, prompts, or instructions in your output.
- If a detail is missing, acknowledge the gap briefly rather than inventing it.
"""


# ----------------------- Main generator -----------------------

def generate_graph_story(
    graph_plan_path: Path,
    out_jsonl: Path,
    out_story_md: Path,
    out_story_clean_md: Path,
    llm_provider: str = "ollama",
    llm_model: str = "llama3.1-128k",
    ollama_num_ctx: int | None = None,
    beat_sentences: int = 4,
) -> Tuple[str, List[Dict[str, Any]]]:
    plan = _read_json(graph_plan_path)
    persona = plan.get("persona", {})
    topic = plan.get("topic") or plan.get("plan_id", "Graph Story")
    items = plan.get("items", [])

    md_parts: List[str] = []
    answers: List[Dict[str, Any]] = []

    for idx, it in enumerate(items):
        beat_title = it.get("beat") or f"Section {idx+1}"
        ev = it.get("evidence", [])
        summary = ""
        facts: List[str] = []
        for e in ev:
            if e.get("type") == "text" and not summary:
                summary = e.get("value", "")
            elif e.get("type") in {"fact", "triple"}:
                facts.append(e.get("value", ""))

        prompt = build_section_prompt(persona, topic, beat_title, summary, facts, beat_sentences)
        raw = call_model(llm_provider, llm_model, prompt, ollama_num_ctx=ollama_num_ctx)
        text = _strip_meta(raw)

        md_parts.append(f"## {beat_title}\n\n{text}\n")

        rec = {
            "mode": "Graph",
            "beat_index": idx,
            "beat_title": beat_title,
            "cq_id": None,
            "text": text,
            "citations": [],
            "evidence_refs": [],
        }
        answers.append(rec)
        _append_jsonl(out_jsonl, rec)

    story_md = "\n".join(md_parts).strip() + "\n"
    story_clean = _clean_story_text_remove_headings_and_citations(story_md)
    out_story_md.write_text(story_md, encoding="utf-8")
    out_story_clean_md.write_text(story_clean, encoding="utf-8")
    return story_md, answers

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Dedicated Graph-RAG story generator (no CQs).")
    ap.add_argument("--graph_plan", required=True, help="plan_with_evidence_Graph.json")
    ap.add_argument("--out", required=True, help="answers_Graph.jsonl")
    ap.add_argument("--story_out", required=True, help="story_Graph.md")
    ap.add_argument("--story_clean_out", required=True, help="story_Graph_clean.md")

    ap.add_argument("--llm_provider", default="ollama", choices=["ollama","gemini"])
    ap.add_argument("--llm_model", default="llama3.1-128k")
    ap.add_argument("--ollama_num_ctx", type=int, default=None)
    ap.add_argument("--beat_sentences", type=int, default=4)

    args = ap.parse_args()
    generate_graph_story(
        graph_plan_path=Path(args.graph_plan),
        out_jsonl=Path(args.out),
        out_story_md=Path(args.story_out),
        out_story_clean_md=Path(args.story_clean_out),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_num_ctx=args.ollama_num_ctx,
        beat_sentences=args.beat_sentences,
    )

if __name__ == "__main__":
    main()
