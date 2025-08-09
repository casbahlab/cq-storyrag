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
LENGTH_TO_STYLE = {
    "short": "brief",
    "medium": "moderate",
    "long": "elaborate",
}

# (Not used right now, but kept for future tuning.)
STYLE_BUDGETS = {
    "brief":     {"beats_per_cat": 2, "picks_per_beat": 1},
    "moderate":  {"beats_per_cat": 3, "picks_per_beat": 2},
    "elaborate": {"beats_per_cat": 4, "picks_per_beat": 3},
}

LENGTH_FACT_BUDGET = {
    "short": 8,      # ~6–8 facts total
    "medium": 14,    # ~12–15 facts total
    "long": 22,      # ~20–25 facts total
}

def _quota_by_category(total_budget: int) -> dict:
    """
    Split a total budget across Entry/Core/Exit with a slight bias toward Core.
    """
    weights = {"Entry": 1.0, "Core": 2.0, "Exit": 1.0}
    Z = sum(weights.values())
    raw = {k: total_budget * (w / Z) for k, w in weights.items()}
    q = {k: int(round(v)) for k, v in raw.items()}
    drift = total_budget - sum(q.values())
    order = ["Core", "Entry", "Exit"] if drift > 0 else ["Exit", "Entry", "Core"]
    i = 0
    while drift != 0:
        k = order[i % 3]
        q[k] += 1 if drift > 0 else -1
        drift += -1 if drift > 0 else 1
        i += 1
    for k in q:
        q[k] = max(q[k], 0)
    return q

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
    if explicit_style:
        return explicit_style
    length = ((persona or {}).get("length", "") or "").lower()
    return LENGTH_TO_STYLE.get(length, "moderate")

# --- Top-up helper ---

def top_up_plan(
    persona: dict,
    plan_obj: dict,
    need_by_cat: dict,
    *,
    top_n: int = TOP_N,
    seed: int = 42,
    normalize_embeddings: bool = True,
):
    """
    Augment the existing plan with more CQs in categories that are under-covered.
    - Respects 'need_by_cat' counts.
    - Avoids duplicate cq_id.
    - Uses the same FAISS search strategy.
    - Mutates plan_obj in place and returns it.
    """
    random.seed(seed)

    existing_by_cat = {c: set() for c in CATEGORIES}
    for c in CATEGORIES:
        for it in plan_obj.get("detailed_plan", {}).get(c, []):
            existing_by_cat[c].add(it["cq_id"])

    def _append_pick(cat: str, m: dict):
        plan_obj["detailed_plan"].setdefault(cat, []).append({
            "cq_id": m["cq_id"],
            "text": m["text"],
            "category": m["category"],
            "beat": m.get("beat", ""),
            "sparql": m.get("sparql", "")
        })
        plan_obj["execution"].setdefault(cat, []).append({
            "cq_id": m["cq_id"],
            "sparql": m.get("sparql", ""),
            "question": m.get("text", "")
        })

    for cat, need in (need_by_cat or {}).items():
        if need <= 0:
            continue
        filled = 0
        for beat in BEATS:
            if filled >= need:
                break
            composite_text = f"{cat} | {beat} | {persona}"
            try:
                emb = embed(model=OLLAMA_MODEL, input=composite_text)["embeddings"][0]
            except Exception as e:
                print(f"[ERROR] Embedding failed for {composite_text}: {e}")
                continue

            D, I = index.search(np.array([emb], dtype="float32"), top_n)
            matches = [metadata[i] for i in I[0] if i < len(metadata)]
            candidates = [m for m in matches
                          if m["category"] == cat and m["cq_id"] not in existing_by_cat[cat]]

            take = min(len(candidates), need - filled)
            if take <= 0:
                continue

            picks = random.sample(candidates, take)
            for p in picks:
                existing_by_cat[cat].add(p["cq_id"])
                _append_pick(cat, p)
                filled += 1
                if filled >= need:
                    break

    diag = plan_obj.setdefault("diagnostics", {})
    counts = {k: len(plan_obj.get("detailed_plan", {}).get(k, [])) for k in CATEGORIES}
    diag["counts"] = counts
    return plan_obj

# --- Build plan ---

def build_plan_for_persona(
    persona: dict,
    top_n: int = TOP_N,
    picks_per_beat: int = None,
    seed: int = 42,
    normalize_embeddings: bool = True,
    length_style: str = None,
):
    random.seed(seed)

    # Decide length + budget
    if isinstance(persona, dict):
        length = (length_style or persona.get("length") or "short").lower()
    else:
        length = (length_style or "short").lower()

    total_budget = LENGTH_FACT_BUDGET.get(length, LENGTH_FACT_BUDGET["short"])
    cat_quota = _quota_by_category(total_budget)

    local_picks_per_beat = picks_per_beat
    seen_cq_ids = set()
    picks_per_category = {c: [] for c in CATEGORIES}

    for category in CATEGORIES:
        allocated = 0
        if cat_quota.get(category, 0) <= 0:
            continue

        for beat in BEATS:
            if allocated >= cat_quota[category]:
                break

            composite_text = f"{category} | {beat} | {persona}"
            try:
                emb = embed(model=OLLAMA_MODEL, input=composite_text)["embeddings"][0]
            except Exception as e:
                print(f"[ERROR] Embedding failed for {composite_text}: {e}")
                continue

            D, I = index.search(np.array([emb], dtype="float32"), top_n)
            matches = [metadata[i] for i in I[0] if i < len(metadata)]
            filtered_matches = [m for m in matches if m["cq_id"] not in seen_cq_ids and m["category"] == category]

            remaining = cat_quota[category] - allocated
            take = min(
                local_picks_per_beat if local_picks_per_beat else PICKS_PER_BEAT,
                remaining,
                len(filtered_matches),
            )
            if take <= 0:
                continue

            picks = random.sample(filtered_matches, take)
            for p in picks:
                seen_cq_ids.add(p["cq_id"])
            picks_per_category[category].extend(picks)
            allocated += len(picks)

            if allocated >= cat_quota[category]:
                break

    # Build the final plan object
    if not isinstance(picks_per_category, dict):
        raise TypeError(f"picks_per_category should be dict, got {type(picks_per_category)}")

    plan_obj = build_plan(
        persona=persona,
        cq_catalog=metadata,
        picks_per_category=picks_per_category,
    )

    # Attach diagnostics
    plan_obj.setdefault("diagnostics", {})
    plan_obj["diagnostics"].update({
        "style": {"short":"brief","medium":"moderate","long":"elaborate"}.get(length, "brief"),
        "length": length,
        "budget_total": total_budget,
        "budget_by_category": cat_quota,
        "counts": {k: len(v) for k, v in picks_per_category.items()},
        "beats_per_cat": len(BEATS),
        "picks_per_beat": local_picks_per_beat if local_picks_per_beat else PICKS_PER_BEAT,
        "seed": seed,
    })
    return plan_obj

def build_plan(
    persona: Dict[str, Any],
    cq_catalog: list,
    picks_per_category: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the plan JSON structure from selected CQ metadata.
    (cq_catalog is unused here but kept for signature stability and future use.)
    """
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
    }
    return plan

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a KG-RAG plan for a persona")
    parser.add_argument("--persona-name", type=str, default="Emma")
    parser.add_argument("--tone", type=str, default="educational")
    parser.add_argument("--length", type=str, choices=["short", "medium", "long"], default="short")
    parser.add_argument("--length-style", type=str, choices=["brief", "moderate", "elaborate"])
    parser.add_argument("--picks-per-beat", type=int)
    parser.add_argument("--top-n", type=int, default=TOP_N)

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

    summary = {
        "style": plan["diagnostics"]["style"],
        "beats_per_cat": plan["diagnostics"]["beats_per_cat"],
        "picks_per_beat": plan["diagnostics"]["picks_per_beat"],
        "counts": {k: len(v) for k, v in plan["detailed_plan"].items()}
    }
    print(json.dumps(summary, indent=2))
