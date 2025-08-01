import json

INPUT_FILE = "embeddings/cq_results_with_vectors.json"
OUTPUT_FILE = "embeddings/faiss_metadata.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

metadata = []
for entry in data:
    if "EmbeddingVector" in entry:  # Only include valid embedded entries
        metadata.append({
            "CQ_ID": entry.get("CQ_ID"),
            "CQ_Text": entry.get("CQ_Text"),
            "Persona": entry.get("Persona", ""),
            "Category": entry.get("Category", ""),
            "ExternalLinks": entry.get("ExternalLinks", []),
            "ExternalDigest": entry.get("ExternalDigest", "")
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"[INFO] Metadata JSON generated with {len(metadata)} entries → {OUTPUT_FILE}")
