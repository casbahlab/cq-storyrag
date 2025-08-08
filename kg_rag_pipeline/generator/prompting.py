# kg_rag_pipeline/generator/prompting.py
from typing import Dict, Any, List, Optional

def _flatten_rows(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No results."
    lines = []
    for r in rows:
        parts = [f"{k}={v}" for k, v in r.items()]
        lines.append("; ".join(parts))
    return "\n".join(lines)

def build_prompt(user_question: str,
                 cq: Dict[str, Any],
                 rows: List[Dict[str, str]],
                 persona: Optional[str],
                 category: Optional[str]) -> str:
    facts = _flatten_rows(rows)
    cq_id = cq.get("cq_id", "UNKNOWN")
    cq_text = cq.get("text", "")
    persona = persona or cq.get("persona", "")
    category = category or cq.get("category", "")
    sparql = (cq.get("sparql") or "").strip()

    return f"""You are generating a factual narrative grounded ONLY in the provided knowledge graph facts.

Persona: {persona}
Narrative Segment Category: {category}

User Question:
{user_question}

Matched CQ:
[{cq_id}] {cq_text}

SPARQL Executed:
{sparql}

Retrieved Facts (tabular):
{facts}

Instructions:
- Write a cohesive paragraph (4–6 sentences) tailored to the persona.
- Use only the retrieved facts; do NOT invent information.
- If facts are sparse, say so explicitly and keep it concise.
- Prefer explicit names for events, artists, venues when present in facts.
"""
