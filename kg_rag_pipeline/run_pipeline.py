import json
from planner.planner import build_plan_for_persona
from retriever.retriever import retrieve
from generator.generator import generate
from utils_llm import ollama_chat

def run(persona: dict, kg_ttl="data/liveaid_instances.ttl"):
    my_plan = build_plan_for_persona(persona)
    facts = retrieve(my_plan, kg_ttl)
    narrative = generate(ollama_chat, my_plan, facts)
    return {"plan": my_plan, "facts": facts, "narrative": narrative}


import datetime
import json
import os

if __name__ == "__main__":
    persona = {
        "name": "Emma",
        "tone": "warm, enthusiastic",
        "length": "medium",
        "focus": ["performance", "broadcast", "impact"],
        "audience": "museum visitor"
    }
    out = run(persona)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"narrative_output_{ts}"

    # --- Save JSON ---
    json_file = f"{base_filename}.json"
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(out, jf, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON output saved to {json_file}")

    # --- Save Markdown ---
    md_file = f"{base_filename}.md"
    with open(md_file, "w", encoding="utf-8") as mf:
        mf.write(f"# Narrative Output ({persona['name']})\n\n")
        mf.write(f"**Generated:** {datetime.datetime.now().isoformat()}\n\n")

        mf.write("## Narrative\n\n")
        mf.write(out["narrative"] + "\n\n")

        mf.write("## Plan\n\n")
        mf.write("```json\n")
        mf.write(json.dumps(out["plan"], indent=2, ensure_ascii=False))
        mf.write("\n```\n\n")

        mf.write("## Facts Retrieved\n\n")
        mf.write("```json\n")
        mf.write(json.dumps(out["facts"], indent=2, ensure_ascii=False))
        mf.write("\n```\n")
    print(f"[INFO] Markdown output saved to {md_file}")

