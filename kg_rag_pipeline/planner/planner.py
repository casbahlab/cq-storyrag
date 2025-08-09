import faiss
import json
import numpy as np
import random
from ollama import embed
from datetime import datetime
import pandas as pd
from typing import Dict, Optional, Any, List

# --- Defaults (can be overridden by run_pipeline) ---
INDEX_FILE = "index_prep/faiss_index/cq_embeddings.index"
META_FILE = "index_prep/faiss_index/metadata.json"
OLLAMA_MODEL = "nomic-embed-text"

TOP_N = 10
PICKS_PER_BEAT = 2
SEED: Optional[int] = 42
NORMALIZE_EMB = True

CATEGORIES = ["Entry", "Core", "Exit"]
CSV_FILE = "data/WembleyRewindCQs_categories_with_beats.csv"

# Preload CSV / BEATS
df = pd.read_csv(CSV_FILE)
BEATS: List[str] = df["Beats"].dropna().unique().tolist()

# Load FAISS index & metadata
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "r") as f:
    metadata = json.load(f)

# -------- Length-aware budgeting --------
# You can tweak these safely.
LENGTH_TO_STYLE = {
    "short": "brief",
    "medium": "moderate",
    "long": "elaborate",
}

# For each style, decide how many beats per category to consider,
# and how many picks per beat we’ll keep in the plan.
STYLE_BUDGETS = {
    "brief":     {"beats_per_cat": 2, "picks_per_beat": 1},
    "moderate":  {"beats_per_cat": 3, "picks_per_beat": 2},
    "elaborate": {"beats_per_cat": 4, "picks_per_beat": 3},
}


def _maybe_normalize(v: np.ndarray, do_norm: bool) -> np.ndarray:
    if not do_norm:
        return v
    n = float(np.linalg.norm(v)) + 1e-8
    return v / n


def _embed(text: str, normalize: bool) -> np.ndarray:
    out = embed(model=OLLAMA_MODEL, input=text)["embeddings"][0]
    v = np.array(out, dtype="float32")
    v = _maybe_normalize(v, normalize)
    return v.reshape(1, -1)


def _derive_style(persona: Dict[str, Any], explicit_style: Optional[str]) -> str:
    """
    Decide the planning style:
      - if explicit_style provided (brief/moderate/elaborate), use it
      - else derive from persona['length'] (short/medium/long)
      - fallback to 'moderate'
    """
    if explicit_style:
        return explicit_style
    length = ((persona or {}).get("length", "") or "").lower()
    return LENGTH_TO_STYLE.get(length, "moderate")


def build_plan_for_persona(
    persona: Dict[str, Any],
    *,
    top_n: int = TOP_N,
    picks_per_beat: Optional[int] = None,      # if None, will come from style budget
    seed: Optional[int] = SEED,
    normalize_embeddings: bool = NORMALIZE_EMB,
    length_style: Optional[str] = None         # 'brief' | 'moderate' | 'elaborate'
) -> Dict[str, Any]:
    """
    Length-aware planner:
    - Derives a style from persona.length (short/medium/long) unless you pass length_style explicitly.
    - Applies style budgets: slice BEATS per category and adjust picks_per_beat accordingly.
    """
    if seed is not None:
        random.seed(seed)

    style = _derive_style(persona, length_style)
    budget = STYLE_BUDGETS.get(style, STYLE_BUDGETS["moderate"])
    beats_per_cat = int(budget["beats_per_cat"])
    picks_per_beat_eff = int(picks_per_beat) if picks_per_beat is not None else int(budget["picks_per_beat"])

    seen_cq_ids = set()
    picks_per_category: Dict[str, Any] = {}
    diagnostics = {"skipped": {}, "empty": [], "style": style, "beats_per_cat": beats_per_cat, "picks_per_beat": picks_per_beat_eff}

    k = min(int(top_n), max(index.ntotal, 1))

    # We’ll take a stable prefix of BEATS to keep determinism; if you prefer
    # dynamic ranking of beats by similarity, we can add that later.
    beats_slice = BEATS[:beats_per_cat] if beats_per_cat > 0 else []

    for category in CATEGORIES:
        picks: List[Dict[str, Any]] = []
        for beat in beats_slice:
            composite_text = f"{category} | {beat} | {persona.get('name', persona)}"
            try:
                q = _embed(composite_text, normalize_embeddings)
                D, I = index.search(q, k)
            except Exception as e:
                diagnostics["skipped"].setdefault(category, []).append({"beat": beat, "reason": str(e)})
                continue

            # Build (score, metadata) tuples and sort by score (ascending for L2)
            pairs = []
            for dist, idx in zip(D[0], I[0]):
                idx = int(idx)
                if 0 <= idx < len(metadata):
                    m = metadata[idx]
                    pairs.append((float(dist), m))
            pairs.sort(key=lambda x: x[0])

            # Dedup by cq_id and take top picks_per_beat_eff
            ranked = [m for _, m in pairs if m.get("cq_id") not in seen_cq_ids]
            if not ranked:
                diagnostics["empty"].append({"category": category, "beat": beat})
                continue

            take = ranked[:picks_per_beat_eff]
            for t in take:
                cid = t.get("cq_id")
                if cid:
                    seen_cq_ids.add(cid)
            picks.extend(take)

        picks_per_category[category] = picks

    return build_plan(persona, picks_per_category, diagnostics)


def build_plan(
    persona: Dict[str, Any],
    picks_per_category: Dict[str, Any],
    diagnostics: Dict[str, Any]
) -> Dict[str, Any]:
    detailed_plan: Dict[str, Any] = {}
    execution_plan: Dict[str, Any] = {}

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
    return plan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a KG-RAG plan for a persona")
    parser.add_argument(
        "--persona-name", type=str, default="Emma",
        help="Name of the persona (for logging only)"
    )
    parser.add_argument(
        "--tone", type=str, default="educational",
        help="Narrative tone (optional)"
    )
    parser.add_argument(
        "--length", type=str, choices=["short", "medium", "long"], default="short",
        help="Persona length hint (used if --length-style not provided)"
    )
    parser.add_argument(
        "--length-style", type=str, choices=["brief", "moderate", "elaborate"],
        help="Override style: brief/moderate/elaborate"
    )
    parser.add_argument(
        "--picks-per-beat", type=int,
        help="Force picks per beat (overrides style default)"
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N,
        help="Number of nearest neighbours to fetch from FAISS index"
    )

    args = parser.parse_args()

    persona = {
        "name": args.persona_name,
        "tone": args.tone,
        "length": args.length
    }

    plan = build_plan_for_persona(
        persona,
        top_n=args.top_n,
        picks_per_beat=args.picks_per_beat,
        length_style=args.length_style
    )

    # Summary output for quick review
    summary = {
        "style": plan["diagnostics"]["style"],
        "beats_per_cat": plan["diagnostics"]["beats_per_cat"],
        "picks_per_beat": plan["diagnostics"]["picks_per_beat"],
        "counts": {k: len(v) for k, v in plan["detailed_plan"].items()}
    }
    print(json.dumps(summary, indent=2))

