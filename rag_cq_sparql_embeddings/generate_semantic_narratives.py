import json
import faiss
import numpy as np
from datetime import datetime
from ollama import embed, chat

# ===== CONFIG =====
INDEX_FILE = "embeddings/cq_results_faiss.index"
META_FILE = "embeddings/faiss_metadata.json"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.1"
TOP_K = 3

OUTPUT_JSON = f"embeddings/persona_full_narratives_streaming_{TOP_K}.json"
OUTPUT_MD = f"embeddings/persona_full_narratives_streaming_{TOP_K}.md"

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

# ===== LOAD FAISS AND METADATA =====
print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"[INFO] Index vectors: {index.ntotal}, Metadata entries: {len(metadata)}")

# ===== EMBEDDING UTILS =====
def embed_text(text: str):
    resp = embed(model=EMBED_MODEL, input=text)
    return np.array(resp["embeddings"][0], dtype="float32").reshape(1, -1)

def retrieve_facts(query: str, persona: str, category: str, top_k: int = TOP_K):
    """Retrieve top-k facts for a persona/category using FAISS search."""
    vec = embed_text(query)
    distances, indices = index.search(vec, 50)  # search top 50 first

    # Filter results by persona & category
    filtered = [
        (metadata[i], distances[0][j])
        for j, i in enumerate(indices[0])
        if metadata[i].get("Persona") == persona and metadata[i].get("Category") == category
    ]

    top = sorted(filtered, key=lambda x: x[1])[:top_k]  # L2 distance ascending
    return [f"{m['CQ_Text']} | Context: {m.get('ExternalDigest','')}" for m, _ in top]

def stream_narrative(persona: str, entry_facts, core_facts, exit_facts):
    """Stream LLaMA narrative output live while capturing full text."""
    prompt = PROMPT_TEMPLATES[persona].format(
        entry_facts=json.dumps(entry_facts, indent=2, ensure_ascii=False),
        core_facts=json.dumps(core_facts, indent=2, ensure_ascii=False),
        exit_facts=json.dumps(exit_facts, indent=2, ensure_ascii=False),
    )

    print(f"\n[STREAMING NARRATIVE for {persona}] --------------------\n")

    stream = chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n\n[INFO] Streaming complete.\n")
    return full_response

# ===== MAIN PIPELINE =====
def main():
    personas = ["Emma", "Luca"]
    narratives = []
    md_lines = []

    for persona in personas:
        # Retrieve top facts for each narrative section
        entry_facts = retrieve_facts("introduction live aid", persona, "Entry")
        core_facts = retrieve_facts("performance highlights live aid", persona, "Core")
        exit_facts = retrieve_facts("legacy impact live aid", persona, "Exit")

        # Stream narrative
        narrative = stream_narrative(persona, entry_facts, core_facts, exit_facts)

        # Record result
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

    print(f"[INFO] Streaming narratives saved:\n- {OUTPUT_JSON}\n- {OUTPUT_MD}")

if __name__ == "__main__":
    main()
