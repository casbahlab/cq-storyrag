from typing import Dict, Any, List
import re

def _flat(rows: List[dict]) -> str:
    if not rows:
        return "No results."
    return "\n".join("; ".join(f"{k}={v}" for k, v in r.items()) for r in rows)

def _apply_labels_to_rows(rows: List[dict], labels: dict) -> List[dict]:
    if not labels:
        return rows
    out = []
    for r in rows:
        out.append({k: labels.get(v, v) for k, v in r.items()})
    return out

def build_prompt(persona: Dict[str, Any], facts: Dict[str, Any]) -> str:
    def sec(cat):
        parts = []
        for item in facts.get(cat, []):
            pretty_rows = _apply_labels_to_rows(item.get("rows", []), item.get("labels", {}))
            parts.append(f"- {item.get('question','')}\n  Facts: {_flat(pretty_rows)}")
        return "\n".join(parts) if parts else "No facts found."

    name = persona.get("name", "Visitor")
    tone = persona.get("tone", "educational")
    length = persona.get("length", "short")

    return f"""
You are a narrative writer for a museum audio guide.

Persona: {name}
Tone: {tone}
Length: {length}

Use only the facts below. If a section has no facts, gracefully move on.

[ENTRY FACTS]
{sec('Entry')}

[CORE FACTS]
{sec('Core')}

[EXIT FACTS]
{sec('Exit')}

Guidelines:
- Use {tone} tone and keep {length} length.
- Prefer human-readable labels over URIs (labels injected where available).
- Compress multiple rows within the same category into short, clear sentences.
- Create smooth transitions: Entry → Core, Core → Exit.
- Do not invent or guess facts. If unsure, omit.

Now write the narrative:
""".strip()

def generate(llm_call, plan: Dict[str, Any], facts: Dict[str, Any]):
    prompt = build_prompt(plan['persona'], facts)
    return prompt, llm_call(prompt)
