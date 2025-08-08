import faiss
import json
import numpy as np
import random
from ollama import embed
from datetime import datetime
import pandas as pd
from typing import Dict, List

# --- Config (keep defaults; allow overrides via function args) ---
INDEX_FILE = "index_prep/faiss_index/cq_embeddings.index"
META_FILE = "index_prep/faiss_index/metadata.json"
OLLAMA_MODEL = "nomic-embed-text"

TOP_N = 10            # nearest neighbours from FAISS
PICKS_PER_BEAT = 2    # how many to pick per beat
SEED = 42             # set None for nondeterministic
NORMALIZE_EMB = True  # set False if your index expects L2

# Categories are fixed
CATEGORIES = ["Entry", "Core", "Exit"]
CSV_FILE = "data/WembleyRewindCQs_categories_with_beats.csv"

# Preload CSV
df = pd.read_csv(CSV_FILE)
BEATS = df["Beats"].dropna().unique().tolist()

# Load FAISS index & metadata once
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r") as f:
    metadata = json.load(f)

if SEED is not None:
    random.seed(SEED)

def _maybe_normalize(v: np.ndarray) -> np.ndarray:
    if not NORMALIZE_EMB:
        return v
    n = np.linalg.norm(v) + 1e-8
    return v / n

def _embed(text: str) -> np.ndarray:
    out = embed(model=OLLAMA_MODEL, input=text)["embeddings"][0]
    v = np.array(out, dtype="float32")
    v = _maybe_normalize(v)
    return v.reshape(1, -1)

def build_plan_for_persona(persona: Dict, *, top_n:int=TOP_N, picks_per_beat:int=PICKS_PER_BEAT) -> Dict:
    """
    Persona: dict with persona settings (tone/length/etc.).
    Returns a plan dict with detailed/execution sections and diagnostics.
    """
    seen_cq_ids = set()
    picks_per_category = {}
    diagnostics = {"skipped": {}, "empty": []}

    k = min(int(top_n), max(index.ntotal, 1))

    for category in CATEGORIES:
        picks = []
        for beat in BEATS:
            composite_text = f"{category} | {beat} | {persona.get('name', persona)}"
            try:
                q = _embed(composite_text)
                D, I = index.search(q, k)
            except Exception as e:
                diagnostics["skipped"].setdefault(category, []).append({"beat": beat, "reason": str(e)})
                continue

            # Build (score, item) pairs, filter invalid ids
            pairs = []
            for dist, idx in zip(D[0], I[0]):
                if 0 <= idx < len(metadata):
                    m = metadata[idx]
                    score = float(dist)  # sort asc for L2 / adjust if IP
                    pairs.append((score, m))

            # Sort deterministically by score (ascending typical for L2)
            pairs.sort(key=lambda x: x[0])
            ranked = [m for _, m in pairs if m.get("cq_id") not in seen_cq_ids]

            if not ranked:
                diagnostics["empty"].append({"category": category, "beat": beat})
                continue

            # Deterministic take
            take = ranked[:picks_per_beat]
            for t in take:
                if cid := t.get("cq_id"):
                    seen_cq_ids.add(cid)
            picks.extend(take)

        picks_per_category[category] = picks

    return build_plan(persona, picks_per_category, diagnostics)

def build_plan(persona: Dict, picks_per_category: Dict, diagnostics: Dict) -> Dict:
    detailed_plan = {}
    execution_plan = {}

    for category in CATEGORIES:
        category_picks = picks_per_category.get(category, [])
        detailed_plan[category] = [
            {
                "cq_id": c.get("cq_id"),
                "text": c.get("text"),
                "category": c.get("category"),
                "beat": c.get("beat", ""),
                "sparql": c.get("sparql", "")
            }
            for c in category_picks
        ]
        execution_plan[category] = [
            {
                "cq_id": c.get("cq_id"),
                "sparql": c.get("sparql", ""),
                "question": c.get("text", "")
            }
            for c in category_picks
        ]

    plan = {
        "persona": persona,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detailed_plan": detailed_plan,
        "execution": execution_plan,
        "diagnostics": diagnostics
    }

    # The caller (run_pipeline) is responsible for saving artifacts into a run folder.
    return plan

if __name__ == "__main__":
    persona = {"name": "Emma", "tone": "educational", "length": "short"}
    plan = build_plan_for_persona(persona)
    print(json.dumps({k: (v if k!="detailed_plan" else {kk: len(vv) for kk, vv in v.items()}) for k,v in plan.items()}, indent=2))
