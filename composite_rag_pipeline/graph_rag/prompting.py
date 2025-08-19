def format_triples(tris, limit=60):
    return "\n".join(f"- {s} | {p} | {o}" for s,p,o in tris[:limit])

def render_story_from_graph(d: dict) -> str:
    cq = d.get("cq_text","")
    persona = d.get("persona",{})
    graph_summary = d.get("graph_summary","")
    triples = d.get("triples",[])

    prompt = f"""You are a museum storyteller for the British Music Experience.

User persona:
- name: {persona.get('name','')}
- expertise: {persona.get('level','')}
- goals: {', '.join(persona.get('goals', []))}

Competency question:
{cq}

Graph neighborhood (keep relationships intact and coherent):
{graph_summary}

Triples for factual grounding (subject | predicate | object):
{format_triples(triples)}

Write a coherent story that answers the CQ using only the provided knowledge.
Do not use numeric bracket citations or invent facts.
Prefer plain sentences, connect related nodes, and keep it concise for this persona.
"""
    return call_model(prompt)  # your existing LLM wrapper
