import json
import numpy as np
import faiss
from tqdm import tqdm
from ollama import embed  # Make sure Ollama Python client is installed

# ---- CONFIG ----
INPUT_FILE = "embeddings/cq_results_with_enhanced_digests.json"
OUTPUT_FILE = "embeddings/cq_results_with_vectors.json"
FAISS_INDEX_FILE = "embeddings/cq_results_faiss.index"
OLLAMA_MODEL = "nomic-embed-text"  # or any embedding model available in Ollama

# ---- LOAD DATA ----
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = json.load(f)

print(f"[INFO] Loaded {len(entries)} entries for embedding")

vectors = []
valid_entries = []

# ---- GENERATE EMBEDDINGS ----
for entry in tqdm(entries, desc="Generating embeddings"):
    embedding_input = entry.get("EmbeddingInput", "").strip()

    if not embedding_input or embedding_input.endswith("External Summary:\nNone"):
        print(f"[SKIP] {entry['CQ_ID']} has no meaningful content for embedding")
        continue

    resp = embed(model=OLLAMA_MODEL, input=embedding_input)
    vector = resp["embeddings"][0]  # FIX: Use 'embeddings'

    entry["EmbeddingVector"] = vector
    vectors.append(vector)
    valid_entries.append(entry)


# Convert to NumPy array
vectors_np = np.array(vectors).astype("float32")

# ---- BUILD FAISS INDEX ----
dim = len(vectors_np[0])
index = faiss.IndexFlatL2(dim)
index.add(vectors_np)

# Save FAISS index and JSON with vectors
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"[INFO] FAISS index saved to {FAISS_INDEX_FILE} with {len(vectors)} vectors")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(valid_entries, f, indent=2, ensure_ascii=False)

print(f"[INFO] JSON with embeddings saved to {OUTPUT_FILE}")
