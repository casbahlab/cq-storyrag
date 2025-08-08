from typing import Dict, Any, List

def _flat(rows: List[dict]) -> str:
    if not rows:
        return "No results."
    return "\n".join("; ".join(f"{k}={v}" for k, v in r.items()) for r in rows)


def build_prompt(persona, facts):
    print(f"persona : {persona}")
    return f"""
You are a narrative writer for a museum audio guide.

Persona
- Tone: {['tone']}
- Length: {persona['length']}
- Audience: {persona.get('audience','general')}
- Focus: {', '.join(persona['focus'])}

Goal
Produce ONE cohesive narrative (no headings/bullets). Weave facts naturally.
Do not invent facts. Where multiple rows exist, compress sensibly.

Context (factual notes)
[Entry] {facts['Entry']}
[Core] {facts['Core']}
[Exit] {facts['Exit']}

Style
- Keep {persona['length']} length.
- Maintain {persona['tone']} tone.
- Bridge Entry→Core and Core→Exit with connective sentences.
- Prefer human-readable labels over raw URIs.

Now write the narrative:
""".strip()

def generate(llm_call, plan, facts):
    print(f"plan : {plan['persona']}")
    prompt = build_prompt(plan['persona'], facts)
    return llm_call(prompt)
