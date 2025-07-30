# generate_fact_embeddings.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

CQ_RESULTS_FILE = "embeddings/cq_results.json"
INDEX_FILE = "embeddings/fact_index.json"
EMB_FILE = "embeddings/fact_embeddings.npy"

model = SentenceTransformer('all-MiniLM-L6-v2')

def main():
    with open(CQ_RESULTS_FILE, "r", encoding="utf-8") as f:
        cq_results = json.load(f)

    entries, texts = [], []

    for item in cq_results:
        cq_id = item["CQ_ID"]
        persona = item.get("Persona", "")
        category = item.get("Category", "")
        for fact in item.get("Results", []):
            fact_text = json.dumps(fact, ensure_ascii=False)
            entries.append({
                "CQ_ID": cq_id,
                "Persona": persona,
                "Category": category,
                "Fact": fact_text
            })
            texts.append(fact_text)

    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(EMB_FILE, embeddings)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(entries)} embeddings to {EMB_FILE}")
    print(f"Metadata saved to {INDEX_FILE}")

if __name__ == "__main__":
    main()
