import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_FILE = "embeddings/cq_index.json"
EMB_FILE = "embeddings/cq_embeddings.npy"

model = SentenceTransformer('all-MiniLM-L6-v2')

def find_best_cq(user_question: str, top_k: int = 1):
    # Load index and embeddings
    with open(INDEX_FILE, encoding="utf-8") as f:
        cq_index = json.load(f)
    embeddings = np.load(EMB_FILE)

    # Encode query
    query_emb = model.encode([user_question])
    scores = cosine_similarity(query_emb, embeddings)[0]

    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        cq = cq_index[idx]
        cq["score"] = float(scores[idx])
        results.append(cq)
    return results[0] if top_k == 1 else results
