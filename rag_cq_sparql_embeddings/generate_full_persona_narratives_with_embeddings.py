# generate_semantic_narratives.py
import json
import numpy as np
import subprocess
from datetime import datetime

INDEX_FILE = "embeddings/fact_index.json"
EMB_FILE = "embeddings/fact_embeddings.npy"
OUTPUT_JSON = "embeddings/persona_full_narratives_semantic.json"
OUTPUT_MD = "embeddings/persona_full_narratives_semantic.md"
OLLAMA_MODEL = "llama3.1"

TOP_K = 8  # facts per category to keep prompts concise

PROMPT_TEMPLATES = {
    "Emma": """You are a friendly museum storyteller.

Persona: Emma (Curious Novice)
Goal: Cultural awareness, empathy-driven learning, accessible narrative.

We are creating a cohesive story in three sections:
1. Entry – Introduce Live Aid for a first-time visitor
2. Core – Share performance highlights and key event details
3. Exit – Describe the impact, legacy, and humanitarian outcomes

Here are the facts for each section:

Entry Facts:
{entry_facts}

Core Facts:
{core_facts}

Exit Facts:
{exit_facts}

Task:
Generate a fluent and engaging 3-part narrative in the format:

## Entry
<entry narrative>

## Core
<core narrative>

## Exit
<exit narrative>
""",
    "Luca": """You are a curated music history narrator.

Persona: Luca (Informed Enthusiast)
Goal: Curated narrative, legacy formation, artist and media focus.

We are creating a cohesive story in three sections:
1. Entry – Introduce the event with context for an informed visitor
2. Core – Focus on performances, setlists, and broadcast legacy
3. Exit – Reflect on the cultural impact and symbolic highlights

Here are the facts for each section:

Entry Facts:
{entry_facts}

Core Facts:
{core_facts}

Exit Facts:
{exit_facts}

Task:
Generate a fluent, structured narrative in the format:

## Entry
<entry narrative>

## Core
<core narrative>

## Exit
<exit narrative>
"""
}

# ---- Load embeddings and metadata ----
with open(INDEX_FILE, "r", encoding="utf-8") as f:
    fact_index = json.load(f)
embeddings = np.load(EMB_FILE)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_facts(query, persona, category, top_k=TOP_K):
    """Semantic search for top-k facts for a persona & category."""
    query_emb = model.encode([query])
    scores = np.dot(embeddings, query_emb.T).squeeze()

    # Filter to persona & category
    candidates = [(i, s) for i, s in enumerate(scores)
                  if fact_index[i]["Persona"] == persona and
                     fact_index[i]["Category"] == category]

    top = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    return [fact_index[i]["Fact"] for i, _ in top]

def generate_narrative_ollama(persona, entry_facts, core_facts, exit_facts):
    prompt = PROMPT_TEMPLATES[persona].format(
        entry_facts=json.dumps(entry_facts, indent=2, ensure_ascii=False),
        core_facts=json.dumps(core_facts, indent=2, ensure_ascii=False),
        exit_facts=json.dumps(exit_facts, indent=2, ensure_ascii=False),
    )
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()

def main():
    personas = ["Emma", "Luca"]
    categories = ["Entry", "Core", "Exit"]

    narratives = []
    md_lines = []

    for persona in personas:
        # Use persona name as semantic query seed for variety
        entry_facts = retrieve_facts("introduction live aid", persona, "Entry")
        core_facts = retrieve_facts("performance highlights live aid", persona, "Core")
        exit_facts = retrieve_facts("legacy impact live aid", persona, "Exit")

        print(f"Generating semantic RAG narrative for {persona}...")
        narrative = generate_narrative_ollama(persona, entry_facts, core_facts, exit_facts)

        persona_story = {
            "Persona": persona,
            "GeneratedAt": datetime.now().isoformat(),
            "EntryFactsUsed": entry_facts,
            "CoreFactsUsed": core_facts,
            "ExitFactsUsed": exit_facts,
            "FullNarrative": narrative
        }
        narratives.append(persona_story)

        md_lines.append(f"# Persona Narrative: {persona}\n")
        md_lines.append(narrative)
        md_lines.append("\n---\n")

    # Save outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Semantic persona narratives saved to:\n- {OUTPUT_JSON}\n- {OUTPUT_MD}")

if __name__ == "__main__":
    main()
