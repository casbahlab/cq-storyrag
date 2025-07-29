import json
import numpy as np
from sentence_transformers import SentenceTransformer

CSV_FILE = "cq_sparql_full_mapping.csv"
INDEX_FILE = "embeddings/cq_index.json"
EMB_FILE = "embeddings/cq_embeddings.npy"

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_embedding_index():
    import csv
    cqs = []
    texts = []
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cqs.append({"CQ_ID": row["CQ_ID"], "CQ_Text": row["CQ_Text"], "SPARQL_Query": row["SPARQL_Query"]})
            texts.append(row["CQ_Text"])
    embeddings = model.encode(texts)
    np.save(EMB_FILE, embeddings)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(cqs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(cqs)} embeddings to {EMB_FILE}")

if __name__ == "__main__":
    build_embedding_index()
