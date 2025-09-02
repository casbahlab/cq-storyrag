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

#- Return SSML only inside <speak>; keep wording unchanged; mark intonation by wrapping each clause in <prosody> with pitch=\"+2st\" (rise), \"-2st\" (fall), \"0st\" (neutral), \"+4st\" (question), \"-4st\" (final); insert pauses with <break time=\"200ms\"/>.






# add near the top with your other imports
import re, json

LIT_RX = re.compile(r'\blit::[A-Za-z0-9]+\b')
MULTI_DASH_SPLIT = re.compile(r'\s-\s+')           # " - " separators
HAS_ARROW = re.compile(r'→')
TRIPLE_SHAPE = re.compile(r'^[^→]+→[^→]+→[^→]+$')  # subject → predicate → object

def _explode_graph_context(evidence_items):
    """
    Turn plan evidence into clean atomic context lines:
      - split 'Graph neighborhood: - ... - ...' into separate triples
      - keep {type: "triple" | "fact"} values as-is when usable
      - drop typed-literal crumbs and malformed fragments
    """
    out = []

    def _tidy(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("“","\"").replace("”","\"").replace("’","'")
        s = LIT_RX.sub("", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for e in evidence_items or []:
        et = (e.get("type") or "").lower()
        val = _tidy(e.get("value", ""))

        if not val:
            continue

        # Case 1: a “Graph neighborhood:” blob → split on " - "
        if val.lower().startswith("graph neighborhood:"):
            core = val.split(":", 1)[1].strip()
            for frag in MULTI_DASH_SPLIT.split(core):
                frag = _tidy(frag)
                if not frag:
                    continue
                if not HAS_ARROW.search(frag):
                    continue
                if not TRIPLE_SHAPE.match(frag):
                    # drop half-triples like "→ location → X"
                    continue
                out.append(frag)
            continue

        # Case 2: explicit triples/facts from plan
        if et in {"triple", "fact"}:
            # keep only clean triples or non-empty facts
            if et == "triple":
                if TRIPLE_SHAPE.match(val):
                    out.append(val)
            else:
                out.append(val)
            continue

        # Case 3: plain text summary — keep if short and informative
        if "→" in val and TRIPLE_SHAPE.match(val):
            out.append(val)
        elif len(val) >= 12:
            out.append(val)

    # de-dup, keep shortish lines
    cleaned = []
    seen = set()
    for c in out:
        c = _tidy(c)
        if not c:
            continue
        if len(c) > 180:
            # trim overly long lines at the last space
            c = c[:180].rsplit(" ", 1)[0] + "…"
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
    return cleaned


# in pipeline_graph.py (or wherever generate_graph_story lives)

def _evidence_to_context_lines(evidence):
    """
    Convert plan['items'][i]['evidence'] into plain text lines.
    Keeps 'text' and 'fact' values as-is; 'triple' values are also kept as-is,
    since your plan already stores a readable string.
    """
    lines = []
    for e in evidence or []:
        v = (e.get("value") or "").strip()
        if not v:
            continue
        t = (e.get("type") or "").lower()
        if t in {"text", "fact", "triple"}:
            lines.append(v)
        else:
            # If you later add other kinds, keep them too:
            lines.append(v)
    # de-dupe while preserving order
    seen = set()
    out = []
    for L in lines:
        k = L.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(L)
    return out

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
    import re, json
    from typing import Any, Dict, List

    # --- light cleaners (local, no other deps) ---
    _WS = re.compile(r"\s+")
    _URL_OR_IRI = re.compile(r'https?://\S+|<[^>]+>')

    def _light_clean_line(s: str) -> str:
        if not s:
            return ""
        s = str(s).replace("“", '"').replace("”", '"').replace("’", "'")
        s = _URL_OR_IRI.sub("", s)
        s = _WS.sub(" ", s).strip()
        return s

    def _dedupe_ci(lines: List[str]) -> List[str]:
        out, seen = [], set()
        for ln in lines:
            c = _light_clean_line(ln)
            if not c:
                continue
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out

    def _collect_context_lines_from_item(it: Dict[str, Any]) -> List[str]:
        """
        Pull evidence text, facts, and simple triples from a plan item into 1-line 'anchors'.
        """
        lines: List[str] = []

        # 1) evidence array: prefer value/text/content; include triples if present
        for ev in (it.get("evidence") or []):
            v = ev.get("value") or ev.get("text") or ev.get("content")
            if isinstance(v, str) and v.strip():
                lines.append(v)
            tri = ev.get("triple") or ev.get("spo")
            if isinstance(tri, (list, tuple)) and any(tri):
                parts = [str(x).strip() for x in tri if str(x).strip()]
                if parts:
                    lines.append(" — ".join(parts))

        # 2) any top-level rows/records that look like texty facts or triples
        for r in (it.get("rows") or []):
            if isinstance(r, str) and r.strip():
                lines.append(r.strip())
                continue
            if isinstance(r, dict):
                for k in ("text", "value", "label", "object", "o"):
                    rv = r.get(k)
                    if isinstance(rv, str) and rv.strip():
                        lines.append(rv.strip())
                tri = r.get("triple") or r.get("spo")
                if isinstance(tri, (list, tuple)) and any(tri):
                    parts = [str(x).strip() for x in tri if str(x).strip()]
                    if parts:
                        lines.append(" — ".join(parts))

        # 3) optionally include the summary and explicit facts captured below
        #    (kept short; your prompt will still use summary separately)
        # Note: these will be appended later when we have them.

        return _dedupe_ci(lines)

    # --- main body ---
    plan = _read_json(graph_plan_path)
    persona = plan.get("persona", {})
    topic = plan.get("topic") or plan.get("plan_id", "Graph Story")
    items = plan.get("items", [])

    md_parts: List[str] = []
    answers: List[Dict[str, Any]] = []

    for idx, it in enumerate(items):
        beat_title = it.get("beat") or f"Section {idx+1}"

        # Build evidence summary + facts for the prompt
        ev = it.get("evidence", []) or []
        summary = ""
        facts: List[str] = []
        for e in ev:
            t = e.get("type")
            if t == "text" and not summary:
                summary = e.get("value", "") or e.get("text", "") or e.get("content", "")
            elif t in {"fact", "triple"}:
                val = e.get("value", "")
                if val:
                    facts.append(val)

        prompt = build_section_prompt(persona, topic, beat_title, summary, facts, beat_sentences)
        print(f"prompt : {prompt}")
        raw = call_model(llm_provider, llm_model, prompt, ollama_num_ctx=ollama_num_ctx)
        text = _strip_meta(raw)

        # Compose per-beat context lines (plan-derived) and lightly extend with summary/facts
        ctx_lines = _evidence_to_context_lines(ev)

        # Write JSONL record with context_lines
        rec = {
            "mode": "Graph",
            "beat_index": idx,
            "beat_title": beat_title,
            "cq_id": None,
            "text": text,
            "citations": [],
            "evidence_refs": [],
            "context_lines": ctx_lines,   # <-- NEW: what support_ctx needs
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
