import faiss
import json
import numpy as np
from ollama import embed, chat

# === CONFIG ===
INDEX_FILE = "embeddings/cq_results_faiss.index"       # FAISS vector index
META_FILE = "embeddings/faiss_metadata.json"           # Metadata for retrieved results
EMBED_MODEL = "nomic-embed-text"            # Embedding model
GEN_MODEL = "llama3.1"                      # LLaMA 3.1 for generation
TOP_K = 3                                   # Number of results to retrieve

# === LOAD INDEX AND METADATA ===
print("[INFO] Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

if len(metadata) != index.ntotal:
    print(f"[WARN] Metadata count {len(metadata)} != FAISS index {index.ntotal}")

print(f"[INFO] Loaded FAISS index with {index.ntotal} vectors.")

# === HELPER FUNCTIONS ===
def embed_text(text: str):
    """Generate embedding vector using Ollama embed API."""
    resp = embed(model=EMBED_MODEL, input=text)
    return np.array(resp["embeddings"][0], dtype="float32").reshape(1, -1)

def retrieve_context(query: str, k=TOP_K):
    """Return top-k metadata entries for the query."""
    vec = embed_text(query)
    distances, indices = index.search(vec, k)
    return [metadata[i] for i in indices[0]], distances[0]

def build_prompt(query: str, retrieved: list):
    """Combine query + retrieved CQ digests into a generation prompt."""
    context_blocks = []
    for entry in retrieved:
        digest = entry.get("ExternalDigest", "")
        cq_text = entry.get("CQ_Text", "")
        persona = entry.get("Persona", "")
        block = f"- CQ: {cq_text}\n- Persona: {persona}\n- Context: {digest}"
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)
    return (
        f"You are WembleyRewind, answering based on curated Live Aid knowledge.\n"
        f"User question: {query}\n\n"
        f"Relevant knowledge:\n{context_text}\n\n"
        f"Answer in a factual and narrative style, grounded in the context above."
    )

def answer_query_stream(query: str):
    """End-to-end RAG query with streaming output."""
    print(f"\n[USER] {query}")
    retrieved, distances = retrieve_context(query, k=TOP_K)

    prompt = build_prompt(query, retrieved)
    print("\n[WEMBLEY REWIND ANSWER] (streaming):\n")

    # Streaming generation
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
    print("\n")  # final newline

    return full_response

# === MAIN INTERACTIVE LOOP ===
if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        answer_query_stream(query)
