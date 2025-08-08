import faiss
import json
import numpy as np
import random
from ollama import embed
from datetime import datetime

# --- Config ---
INDEX_FILE = "index_prep/faiss_index/cq_embeddings.index"
META_FILE = "index_prep/faiss_index/metadata.json"
OUTPUT_PLAN = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OLLAMA_MODEL = "nomic-embed-text"

# Parameters
TOP_N = 10       # number of nearest neighbours from FAISS
PICKS_PER_BEAT = 2

# Categories are fixed
CATEGORIES = ["Entry", "Core", "Exit"]

# Example beats — replace with actual beats list from your CSV if needed
BEATS = [
    "Introduction of the Event",
    "Artist Performance Highlight",
    "Behind-the-scenes Insight",
    "Cultural Impact",
    "Closing Sentiment"
]

# --- Load FAISS index & metadata ---
print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "r") as f:
    metadata = json.load(f)

print(f"[INFO] Metadata entries: {len(metadata)}")

# --- Planner Function ---
def build_plan_for_persona(persona: dict):
    plan = []
    seen_cq_ids = set()

    picks_per_category = {}
    for category in CATEGORIES:
        picks_per_beat = []
        for beat in BEATS:
            # Step 1: Create composite search text
            composite_text = f"{category} | {beat} | {persona}"

            # Step 2: Embed composite text
            try:
                emb = embed(model=OLLAMA_MODEL, input=composite_text)["embeddings"][0]
            except Exception as e:
                print(f"[ERROR] Embedding failed for {composite_text}: {e}")
                continue

            # Step 3: Query FAISS index
            D, I = index.search(np.array([emb], dtype="float32"), TOP_N)

            # Step 4: Collect matching metadata
            matches = [metadata[i] for i in I[0] if i < len(metadata)]

            # Step 5: Randomly pick without repeating CQ IDs
            filtered_matches = [m for m in matches if m["cq_id"] not in seen_cq_ids]
            picks = random.sample(filtered_matches, min(PICKS_PER_BEAT, len(filtered_matches)))
            picks_per_beat.append(picks)
        flat_picks = [p for beat_picks in picks_per_beat for p in beat_picks]
        picks_per_category[category] = flat_picks
    return build_plan(persona, metadata, picks_per_category)

def build_plan(persona: dict, cq_catalog: list, picks_per_category: dict):
    """
    persona: str -> Persona name
    cq_catalog: list[dict] -> Metadata list from metadata.json
    picks_per_category: dict -> Number of CQs to pick from each category
    """
    # Build a map for quick lookup
    by_cat = {}
    for c in cq_catalog:
        by_cat.setdefault(c["category"], []).append(c)

    # Categories are fixed: Entry, Core, Exit
    detailed_plan = {}
    execution_plan = {"Entry": [], "Core": [], "Exit": []}

    for category in ["Entry", "Core", "Exit"]:

        category_picks = picks_per_category[category]
        # Add to detailed plan (full CQ info)
        detailed_plan[category] = [
            {
                "cq_id": c["cq_id"],
                "text": c["text"],
                "category": c["category"],
                "beat": c.get("beat", ""),
                "sparql": c.get("sparql", "")
            }
            for c in category_picks
        ]

        # Add to execution plan (only what retriever needs)
        execution_plan[category] = [
            {
                "cq_id": c["cq_id"],
                "sparql": c.get("sparql", ""),
                "question": c.get("text", "")
            }
            for c in category_picks
        ]

    # Final plan object
    plan = {
        "persona": persona,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detailed_plan": detailed_plan,
        "execution": execution_plan
    }

    # Save to file
    out_file = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"[SUCCESS] Plan saved to {out_file}")
    return plan


# --- Run Planner ---
if __name__ == "__main__":
    persona_name = "Emma"  # Example; replace dynamically if needed
    print(f"[INFO] Building plan for persona: {persona_name}")
    plan = build_plan_for_persona(persona_name)

    with open(OUTPUT_PLAN, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"[SUCCESS] Plan saved to {OUTPUT_PLAN}")
