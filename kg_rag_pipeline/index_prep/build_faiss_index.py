import pandas as pd
import numpy as np
import faiss
import json
from ollama import embed
from tqdm import tqdm

# --- Config ---
CSV_FILE = "../data/WembleyRewindCQs_categories_with_beats.csv"
SPARQL_TEMPLATE_FILE = "../data/cqs_queries_template.rq"
SPARQL_OUT_FILE = "../data/resolved_sparql_queries.rq"
INDEX_OUT = "faiss_index/cq_embeddings.index"
META_OUT = "faiss_index/metadata.json"
OLLAMA_MODEL = "nomic-embed-text"

# --- Mappings for SPARQL placeholders ---
placeholder_mapping_sparql = {
    "{event}": "ex:LiveAid1985",
    "{venue}": "ex:WembleyStadium",
    "{singleartist}": "ex:Madonna",
    "{musicgroup}": "ex:Queen"
}

# --- Mappings for text placeholders ---
placeholder_mapping_text = {
    "[Event]": "Live Aid",
    "[Venue]": "Wembley Stadium",
    "[SingleArtist]": "Madonna",
    "[MusicGroup]": "Queen"
}

# --- Step 1: Load CSV ---
df = pd.read_csv(CSV_FILE)
print(f"[INFO] Loaded {len(df)} CQs from CSV")

# --- Step 2: Load SPARQL templates and apply placeholder substitution ---
with open(SPARQL_TEMPLATE_FILE, "r") as f:
    query_blocks = f.read().split("\n\n")

sparql_map = {}
resolved_queries = []

for block in query_blocks:
    lines = block.strip().splitlines()
    cq_id = None
    for line in lines:
        if line.strip().startswith("#CQ-ID:"):
            _, rest = line.strip().split(":", 1)
            cq_id, cq_text = rest.strip().split(" ", 1)
            break
    if not cq_id:
        continue

    # Replace placeholders for SPARQL
    resolved_block = block
    for k, v in placeholder_mapping_sparql.items():
        resolved_block = resolved_block.replace(k, v)

    # Remove first line (CQ-ID comment)
    resolved_lines = resolved_block.strip().splitlines()
    cleaned_sparql = "\n".join(resolved_lines[1:]).strip()

    sparql_map[cq_id] = cleaned_sparql
    resolved_queries.append(cleaned_sparql)

# Save the resolved queries for inspection
with open(SPARQL_OUT_FILE, "w") as f:
    f.write("\n\n".join(resolved_queries))
print(f"[INFO] Resolved SPARQL queries saved to {SPARQL_OUT_FILE}")

# --- Step 3: Generate embeddings and metadata ---
vectors = []
metadata = []

print("[INFO] Generating embeddings via Ollama...")
for i in tqdm(range(len(df)), desc="Embedding CQs"):
    refactored_cq = df["Refactored CQ"].iloc[i].strip()
    cq_id = df["CQ_ID"].iloc[i]
    persona = df["Persona"].iloc[i]
    category = df["Category"].iloc[i]
    beats = df.get("Beats", pd.Series([None]*len(df))).iloc[i]

    # Create human-readable filled CQ text
    cq_text_filled = refactored_cq
    for k, v in placeholder_mapping_text.items():
        cq_text_filled = cq_text_filled.replace(k, v)

    if not refactored_cq:
        print(f"[SKIP] Empty CQ: {cq_id}")
        continue

    try:
        composite_text = f"Category: {category} | Beat: {beats} | CQ: {refactored_cq}"
        response = embed(model=OLLAMA_MODEL, input=refactored_cq)
        vector = response["embeddings"][0]
    except Exception as e:
        print(f"[ERROR] Embedding failed for {cq_id}: {e}")
        continue

    vectors.append(vector)
    metadata.append({
        "cq_id": cq_id,
        "cq_text_templated": refactored_cq,
        "text": cq_text_filled,
        "refactored_cq": refactored_cq,
        "persona": persona,
        "category": category,
        "beats": beats,
        "sparql": sparql_map.get(cq_id, None)
    })

# --- Step 4: Build and save FAISS index ---
vec_np = np.array(vectors).astype("float32")
index = faiss.IndexFlatL2(vec_np.shape[1])
index.add(vec_np)
faiss.write_index(index, INDEX_OUT)

# --- Step 5: Save metadata ---
with open(META_OUT, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[SUCCESS] Saved FAISS index to {INDEX_OUT} and metadata to {META_OUT}")
