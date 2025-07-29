import json
import pandas as pd
import subprocess
from datetime import datetime
from collections import defaultdict

# ---- CONFIG ----
CQ_TEMPLATE_FILE = "data/WembleyRewindCQs_categories.csv"
CQ_RESULTS_FILE = "embeddings/cq_results.json"
OUTPUT_JSON = "embeddings/persona_full_narratives.json"
OUTPUT_MD = "embeddings/persona_full_narratives.md"
OLLAMA_MODEL = "llama3.1"  # Or any local Ollama model

# Persona prompt templates for full narratives
PROMPT_TEMPLATES = {
    "Emma": """You are a friendly museum storyteller.

Persona: Emma (Curious Novice)
Goal: Cultural awareness, empathy-driven learning, accessible narrative.

We are creating a cohesive story in three sections:
- Entry: Introduce Live Aid for a first-time visitor.
- Core: Share performance highlights and key event details.
- Exit: Describe the impact, legacy, and humanitarian outcomes.

Here are the facts for the {category} section:
{facts}

Task:
Generate a fluent and engaging narrative for this section.
""",
    "Luca": """You are a curated music history narrator.

Persona: Luca (Informed Enthusiast)
Goal: Curated narrative, legacy formation, artist and media focus.

We are creating a cohesive story in three sections:
- Entry: Introduce the event with context for an informed visitor.
- Core: Focus on performances, setlists, and broadcast legacy.
- Exit: Reflect on the cultural impact and symbolic highlights.

Here are the facts for the {category} section:
{facts}

Task:
Generate a fluent, structured narrative for this section, suitable for a knowledgeable audience.
"""
}

def generate_narrative_ollama(persona, category, facts, model=OLLAMA_MODEL):
    """Generate a persona-based narrative for a category using Ollama."""
    facts_str = json.dumps(facts, indent=2, ensure_ascii=False)
    prompt = PROMPT_TEMPLATES[persona].format(category=category, facts=facts_str)

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()

def main():
    # Load CQ template to map CQ_ID → Category
    df_template = pd.read_csv(CQ_TEMPLATE_FILE)
    cq_to_category = {
        str(row["CQ_ID"]).strip(): str(row.get("Category", row.get("Section", "Uncategorized"))).strip()
        for _, row in df_template.iterrows()
        if str(row["CQ_ID"]).strip()
    }
    categories_order = ["Entry", "Core", "Exit"]

    # Load CQ results from KG
    with open(CQ_RESULTS_FILE, "r", encoding="utf-8") as f:
        cq_results = json.load(f)

    # Group facts by persona category
    from collections import defaultdict
    import re

    import re

    facts_by_category = defaultdict(list)

    for item in cq_results:
        cq_id_raw = item["CQ_ID"]
        facts = item.get("Results", [])

        # Normalize CQ_ID
        cq_id_clean = re.sub(r"\s*\(.*?\)", "", cq_id_raw)  # remove (Exit)
        cq_ids = [cid.strip() for cid in cq_id_clean.split("/") if cid.strip()]

        # Remove " Extended" suffix if present
        cq_ids = [cid.replace(" Extended", "") for cid in cq_ids]

        # Map to category
        category = "Uncategorized"
        for cid in cq_ids:
            if cid in cq_to_category:
                category = cq_to_category[cid]
                break

        #print(f"cq_id: {cq_id_raw} -> normalized: {cq_ids} -> category: {category}")

        if facts:
            facts_by_category[category].extend(facts)

    # Generate one narrative per persona with Entry/Core/Exit
    full_persona_narratives = []
    md_lines = []

    for persona in ["Emma", "Luca"]:
        persona_story = {"Persona": persona, "GeneratedAt": datetime.now().isoformat(), "FullNarrative": {}}

        md_lines.append(f"# Persona Narrative: {persona}\n")

        for category in categories_order:
            category_facts = facts_by_category.get(category, [])
            if not category_facts:
                continue

            print(f"Generating {persona} {category} narrative...")
            narrative = generate_narrative_ollama(persona, category, category_facts)
            persona_story["FullNarrative"][category] = narrative

            # Add Markdown section
            md_lines.append(f"## {category} Section\n")
            md_lines.append(narrative + "\n")

        full_persona_narratives.append(persona_story)
        md_lines.append("\n---\n")

    # Save outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_persona_narratives, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nFull persona narratives saved to:\n- {OUTPUT_JSON}\n- {OUTPUT_MD}")

if __name__ == "__main__":
    main()
