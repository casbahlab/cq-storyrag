#!/usr/bin/env python3
# Beat-only planner that selects CQs via embeddings and (optionally) re-ranks per-beat
# lists using SPARQL evidence. Zero changes needed to your indexer/metadata.

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss is required. pip install faiss-cpu") from e

try:
    from ollama import embed as ollama_embed
except Exception as e:
    raise RuntimeError("ollama client is required. pip install ollama") from e

# ---------- text utils ----------
def _normalize_text(s: str) -> str:
    s = (s or "").lower().replace("&","and")
    s = re.sub(r"[^a-z0-9\s]"," ", s)
    return re.sub(r"\s+"," ", s).strip()

_SYNONYMS = {
    "audience reaction": ["audience interaction","audience responses","crowd reaction"],
    "legacy and reflection": ["legacy","reflection","legacy reflection","legacy reflections","legacy & reflection"],
    "historical hook": ["historical context","history hook","context hook"],
    "performance detail": ["performance details","set detail","setlist focus"],
    "cultural impact": ["media reception","impact"],
    "behind the scenes": ["backstage","production","technical"],
    "context setup": ["context","setup","orientation"],
}


# --- add near the top ---
def plan(index_path: str, meta_path: str, plans_path: str,
         persona: str, length: str, limit: Optional[int] = None,
         per_beat_pool: int = 12) -> dict:
    """
    Programmatic API for the embedding planner.
    Returns the same dict the CLI prints (persona, length, beats, total_limit, items[]).
    """
    import json
    from pathlib import Path
    import faiss  # type: ignore
    from ollama import embed as ollama_embed  # type: ignore
    import numpy as np

    # --- paste your existing helper fns here or import them if they’re already in-module ---
    # _normalize_text, _canon, _default_total, _normalize_beats, _embed_query, _round_robin
    # (reuse them exactly as in your file)

    index = faiss.read_index(index_path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    # support new keyed meta and legacy arrays
    if "rows" in meta and "order" in meta:
        order = meta["order"]; rows = meta["rows"]
        ids       = order
        beats     = [rows[c].get("beat","") for c in order]
        texts     = [rows[c].get("text","") for c in order]
        sparqls   = [rows[c].get("sparql","") for c in order]
        questions = [rows[c].get("question","") for c in order]
        answers   = [rows[c].get("answer","") for c in order]
        model     = meta.get("model","nomic-embed-text")
    else:
        ids       = meta["ids"]
        beats     = meta.get("beats", [""]*len(ids))
        texts     = meta.get("texts", [""]*len(ids))
        sparqls   = meta.get("sparqls", [""]*len(ids))
        questions = meta.get("questions", [""]*len(ids))
        answers   = meta.get("answers", [""]*len(ids))
        model     = meta.get("model","nomic-embed-text")

    plans = json.loads(Path(plans_path).read_text(encoding="utf-8"))
    persona_entry = plans.get(persona) or plans.get(persona.title()) or plans.get(persona.lower())
    plan_entry = persona_entry.get(length.title()) if isinstance(persona_entry, dict) else None
    beats_order = _normalize_beats(plan_entry)
    total_limit = int(limit) if limit is not None else \
                  int(plan_entry["target"]) if isinstance(plan_entry, dict) and isinstance(plan_entry.get("target"), int) \
                  else _default_total(length)

    per_beat_lists: list[list[int]] = []
    for beat_label in beats_order:
        qtext = f"[Beat] {beat_label}"
        qv = _embed_query(model, qtext)
        topk = min(12, len(ids)) if per_beat_pool is None else min(per_beat_pool, len(ids))
        scores, idxs = index.search(qv, topk)
        per_beat_lists.append([int(i) for i in idxs[0].tolist()])

    chosen_positions = _round_robin(per_beat_lists, total_limit)

    items = []
    for pos in chosen_positions:
        items.append({
            "id": ids[pos],
            "beat": beats[pos] or "Unspecified",
            "text": texts[pos],
            "question": questions[pos],
            "answer": answers[pos],
            "sparql": sparqls[pos]
        })

    return {
        "persona": persona,
        "length": length,
        "beats": beats_order,
        "total_limit": total_limit,
        "items": items
    }


def _canon(label: str) -> str:
    n = _normalize_text(label or "")
    for canon, vars in _SYNONYMS.items():
        if n == canon or n in [_normalize_text(v) for v in vars]:
            return canon
    return n

def _default_total(length: str) -> int:
    return {"short":3,"medium":6,"long":9}.get((length or "").lower(), 3)

def _normalize_beats(entry: Any) -> List[str]:
    if isinstance(entry, dict) and "beats" in entry:
        beats = entry.get("beats") or []
        beats = [b if isinstance(b, str) else b.get("beat") for b in beats]
        return [b for b in beats if b]
    if isinstance(entry, list) and entry and isinstance(entry[0], dict) and "beat" in entry[0]:
        return [step.get("beat") for step in entry if step.get("beat")]
    if isinstance(entry, list) and entry and isinstance(entry[0], str):
        return entry[:]
    return []

# ---------- embeddings ----------
def _embed_query(model: str, text: str) -> np.ndarray:
    from numpy.linalg import norm
    v = np.array(ollama_embed(model=model, input=[text])["embeddings"][0], dtype="float32")
    n = float(norm(v));  v = v / n if n > 0 else v
    return v.reshape(1, -1)

# ---------- SPARQL re-rank (optional) ----------
def _apply_bindings(sparql: str, bindings: Dict[str,str]) -> str:
    # Replace [Var] placeholders. Values should already include <> if IRIs.
    q = sparql
    for k, v in (bindings or {}).items():
        q = q.replace(f"[{k}]", v)
    return q

def _limit_one(q: str) -> str:
    # Append LIMIT 1 if not present (cheap evidence check)
    if re.search(r"\blimit\s+\d+\b", q, flags=re.I):
        return q
    return q.rstrip() + "\nLIMIT 1"

def _has_results(endpoint: str, query: str, timeout: int = 6) -> bool:
    import requests
    r = requests.post(
        endpoint,
        data=_limit_one(query).encode("utf-8"),
        headers={"Content-Type":"application/sparql-query","Accept":"application/sparql-results+json"},
        timeout=timeout
    )
    r.raise_for_status()
    data = r.json()
    return bool(data.get("results", {}).get("bindings"))

def _rerank_by_kg(
    ids: List[str],
    idxs: List[int],
    sims: List[float],
    sparqls: List[str],
    endpoint: str,
    bindings: Dict[str,str],
    alpha_embed: float,
    alpha_kg: float,
    kg_timeout: int
) -> List[int]:
    # Binary KG evidence: 1 if template returns a row (LIMIT 1), else 0
    import hashlib
    cache: Dict[str,bool] = {}
    scored = []
    for pos, sim in zip(idxs, sims):
        cid = ids[pos]
        tmpl = sparqls[pos] or ""
        if not tmpl:
            kg = 0.0
        else:
            filled = _apply_bindings(tmpl, bindings)
            key = hashlib.md5((endpoint + "\n" + filled).encode("utf-8")).hexdigest()
            if key in cache:
                has = cache[key]
            else:
                try:
                    has = _has_results(endpoint, filled, timeout=kg_timeout)
                except Exception:
                    has = False
                cache[key] = has
            kg = 1.0 if has else 0.0
        score = alpha_embed*sim + alpha_kg*kg
        scored.append((score, pos))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pos for _, pos in scored]

# ---------- round-robin ----------
def _round_robin(lists_of_idx: List[List[int]], total: int) -> List[int]:
    picks, ptrs = [], [0]*len(lists_of_idx)
    while len(picks) < total:
        progressed = False
        for i, lst in enumerate(lists_of_idx):
            if len(picks) >= total: break
            p = ptrs[i]
            if p < len(lst):
                idx = lst[p]; ptrs[i] += 1
                if idx not in picks: picks.append(idx)
                progressed = True
        if not progressed: break
    return picks

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--length", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per_beat_pool", type=int, default=12)

    # SPARQL re-rank flags (optional)
    ap.add_argument("--sparql_endpoint", default=None, help="If set, re-rank by KG evidence")
    ap.add_argument("--bindings", default=None, help="JSON file mapping placeholders (e.g., {'Event':'<ex:LiveAid1985>'})")
    ap.add_argument("--alpha_embed", type=float, default=0.85)
    ap.add_argument("--alpha_kg",    type=float, default=0.15)
    ap.add_argument("--kg_timeout",  type=int,   default=6)
    args = ap.parse_args()

    # load index + meta (supports new keyed format and legacy arrays)
    index = faiss.read_index(args.index)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    if "rows" in meta and "order" in meta:
        order = meta["order"]; rows = meta["rows"]
        ids       = order
        beats     = [rows[cid].get("beat","") for cid in order]
        texts     = [rows[cid].get("text","") for cid in order]
        sparqls   = [rows[cid].get("sparql","") for cid in order]
        questions = [rows[cid].get("question","") for cid in order]
        answers   = [rows[cid].get("answer","") for cid in order]
        model     = meta.get("model","nomic-embed-text")
    else:
        ids       = meta["ids"]
        beats     = meta.get("beats", [""]*len(ids))
        texts     = meta.get("texts", [""]*len(ids))
        sparqls   = meta.get("sparqls", [""]*len(ids))
        questions = meta.get("questions", [""]*len(ids))
        answers   = meta.get("answers", [""]*len(ids))
        model     = meta.get("model","nomic-embed-text")

    plans = json.loads(Path(args.plans).read_text(encoding="utf-8"))
    persona_entry = plans.get(args.persona) or plans.get(args.persona.title()) or plans.get(args.persona.lower())
    if not persona_entry: raise ValueError(f"No persona '{args.persona}' in {args.plans}")
    entry = persona_entry.get(args.length.title()) if isinstance(persona_entry, dict) else None
    if not entry: raise ValueError(f"No plan for persona={args.persona}, length={args.length} in {args.plans}")
    beats_order = _normalize_beats(entry)
    if not beats_order: raise ValueError(f"No beats for persona={args.persona}, length={args.length}")

    total_limit = int(args.limit) if args.limit is not None else \
                  int(entry["target"]) if isinstance(entry, dict) and isinstance(entry.get("target"), int) \
                  else _default_total(args.length)

    # optional bindings
    bindings: Dict[str,str] = {}
    if args.bindings:
        bindings = json.loads(Path(args.bindings).read_text(encoding="utf-8"))

    # per-beat search (collect sims + idxs)
    per_beat_lists: List[List[int]] = []
    for beat_label in beats_order:
        qtext = f"[Beat] {beat_label}"
        qv = _embed_query(model, qtext)
        topk = min(args.per_beat_pool, len(ids))
        sims, idxs = index.search(qv, topk)  # sims: cosine, idxs: positions
        idxs = [int(i) for i in idxs[0].tolist()]
        sims = [float(s) for s in sims[0].tolist()]

        # optional SPARQL re-rank
        if args.sparql_endpoint:
            idxs = _rerank_by_kg(
                ids, idxs, sims, sparqls,
                endpoint=args.sparql_endpoint,
                bindings=bindings,
                alpha_embed=args.alpha_embed,
                alpha_kg=args.alpha_kg,
                kg_timeout=args.kg_timeout
            )
        per_beat_lists.append(idxs)

    chosen = _round_robin(per_beat_lists, total_limit)

    items = []
    for pos in chosen:
        items.append({
            "id": ids[pos],
            "beat": beats[pos] or "Unspecified",
            "text": texts[pos],
            "question": questions[pos],
            "answer": answers[pos],
            "sparql": sparqls[pos]
        })

    print(json.dumps({
        "persona": args.persona,
        "length": args.length,
        "beats": beats_order,
        "total_limit": total_limit,
        "items": items
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
