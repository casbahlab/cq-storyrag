import json
import faiss
import numpy as np
from datetime import datetime
from ollama import embed, chat

# ===== CONFIG =====
INDEX_FILE = "cq_results_faiss.index"
META_FILE = "faiss_metadata.json"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.1"
TOP_K = 8

OUTPUT_JSON = f"embeddings/persona_narratives_clean_{TOP_K}.json"
OUTPUT_MD = f"embeddings/persona_narratives_clean_{TOP_K}.md"

PROMPT_TEMPLATES = {
    "Emma": """You are a friendly museum storyteller.

Persona: Emma (Curious Novice)
Goal: Cultural awareness, empathy-driven learning, accessible narrative.

Here are curated facts for three narrative sections:

Entry:
{entry_facts}

Core:
{core_facts}

Exit:
{exit_facts}

Task:
Write exactly ONE cohesive narrative in three sections. 
Do NOT repeat sections. 
Follow this exact format:

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

Here are curated facts for three narrative sections:

Entry:
{entry_facts}

Core:
{core_facts}

Exit:
{exit_facts}

Task:
Write exactly ONE cohesive narrative in three sections. 
Do NOT repeat sections. 
Follow this exact format:

## Entry
<entry narrative>

## Core
<core narrative>

## Exit
<exit narrative>
"""
}

print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"[INFO] Index vectors: {index.ntotal}, Metadata entries: {len(metadata)}")

# ===== UTILS =====
def concise_fact(entry):
    """Short representation for narrative input."""
    text = entry.get("CQ_Text", "")
    digest = entry.get("ExternalDigest", "").split("|")[0].strip()
    return f"{text} — {digest}" if digest else text

def embed_text(text: str):
    resp = embed(model=EMBED_MODEL, input=text)
    return np.array(resp["embeddings"][0], dtype="float32").reshape(1, -1)

def retrieve_facts(query: str, persona: str, category: str, top_k: int = TOP_K):
    vec = embed_text(query)
    distances, indices = index.search(vec, 50)

    filtered = [
        (metadata[i], distances[0][j])
        for j, i in enumerate(indices[0])
        if metadata[i].get("Persona") == persona and metadata[i].get("Category") == category
    ]

    if not filtered:
        filtered = [
            (metadata[i], distances[0][j])
            for j, i in enumerate(indices[0])
            if metadata[i].get("Category") == category
        ]

    top = sorted(filtered, key=lambda x: x[1])[:top_k]
    facts = [concise_fact(m) for m, _ in top]
    cq_ids = [m["CQ_ID"] for m, _ in top]
    return facts, cq_ids

def clean_narrative(text: str) -> str:
    """Keep only the first instance of each ## Entry/Core/Exit section."""
    sections = {"Entry": None, "Core": None, "Exit": None}
    current = None
    lines = text.splitlines()

    for line in lines:
        if line.strip().startswith("## Entry") and sections["Entry"] is None:
            current = "Entry"; sections["Entry"] = []
            continue
        elif line.strip().startswith("## Core") and sections["Core"] is None:
            current = "Core"; sections["Core"] = []
            continue
        elif line.strip().startswith("## Exit") and sections["Exit"] is None:
            current = "Exit"; sections["Exit"] = []
            continue

        if current and sections[current] is not None:
            sections[current].append(line)

    # Merge into clean narrative
    final_blocks = []
    for key in ["Entry", "Core", "Exit"]:
        if sections[key] is not None:
            final_blocks.append(f"## {key}\n" + "\n".join(sections[key]).strip())

    return "\n\n".join(final_blocks).strip()

def generate_narrative(persona: str, entry_facts, core_facts, exit_facts):
    prompt = PROMPT_TEMPLATES[persona].format(
        entry_facts="\n".join(entry_facts),
        core_facts="\n".join(core_facts),
        exit_facts="\n".join(exit_facts),
    )

    print(f"\n[INFO] Generating narrative for {persona}...\n")
    resp = chat(model=GEN_MODEL, messages=[{"role": "user", "content": prompt}])
    raw_text = resp["message"]["content"]
    return clean_narrative(raw_text)

# ===== MAIN PIPELINE =====
def main():
    personas = ["Emma", "Luca"]
    narratives = []
    md_lines = []

    for persona in personas:
        entry_facts, entry_ids = retrieve_facts("introduction live aid", persona, "Entry")
        core_facts, core_ids = retrieve_facts("performance highlights live aid", persona, "Core")
        exit_facts, exit_ids = retrieve_facts("legacy impact live aid", persona, "Exit")

        narrative = generate_narrative(persona, entry_facts, core_facts, exit_facts)

        persona_story = {
            "Persona": persona,
            "GeneratedAt": datetime.now().isoformat(),
            "EntryFactsUsed": entry_facts,
            "CoreFactsUsed": core_facts,
            "ExitFactsUsed": exit_facts,
            "EntryCQIDs": entry_ids,
            "CoreCQIDs": core_ids,
            "ExitCQIDs": exit_ids,
            "FullNarrative": narrative
        }
        narratives.append(persona_story)

        md_lines.append(f"# Persona Narrative: {persona}\n")
        md_lines.append(narrative)
        md_lines.append("\n---\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[INFO] Clean narratives saved:\n- {OUTPUT_JSON}\n- {OUTPUT_MD}")

if __name__ == "__main__":
    main()
