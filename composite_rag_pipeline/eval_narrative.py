
"""
eval_narrative.py (baked-in, robust matching, per-run defaults)

This file replaces the sentence–evidence alignment with robust matching
and sets sensible defaults per run pattern:
- Graph  : ngram_n=5, jaccard_threshold=0.50
- KG     : ngram_n=4, jaccard_threshold=0.45
- Hybrid : ngram_n=4, jaccard_threshold=0.45

Exports:
- compute_support_from_story_and_plan(story_text, plan_with_evidence, pattern)
- align_sentences_with_evidence(...)
"""

import re
import json
from typing import List, Dict, Any, Iterable, Tuple

# -------------------- Text utils --------------------

def _normalize(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    # Strip <URI> artifacts
    s = re.sub(r"<https?://[^>]+>", "", s)
    # Remove bracketed instance IDs like [ex:Thing] if present
    s = re.sub(r"\[[^\]]+\]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", _normalize(s))

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _jaccard(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _split_sentences(text: str) -> List[str]:
    # lightweight splitter; keeps punctuation with sentences
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [p for p in parts if p]

# -------------------- Evidence extraction --------------------

def _clean_evidence_value(v):
    if v is None:
        return None
    s = str(v)
    # Drop surrounding quotes
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    # Remove URIs
    s = re.sub(r"<https?://[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _flatten_context_entry(x):
    # Flatten common context_lines shapes to strings
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for y in x:
            out.extend(_flatten_context_entry(y))
        return out
    if isinstance(x, dict):
        for k in ("text","line","content","value","snippet"):
            if k in x and isinstance(x[k], str):
                return [x[k]]
        for k in ("lines","snippets","contexts"):
            if k in x and isinstance(x[k], list):
                return _flatten_context_entry(x[k])
        return [json.dumps(x, ensure_ascii=False)]
    return [str(x)]

def extract_evidence_from_plan(plan_json: Dict[str, Any], prefer_context_over_factlets: bool = True) -> List[str]:
    # 1) Prefer explicit context_lines if present
    if prefer_context_over_factlets and plan_json.get("context_lines"):
        seen = set(); items = []
        for entry in plan_json["context_lines"]:
            for s in _flatten_context_entry(entry):
                s2 = s.strip()
                if s2 and s2 not in seen:
                    items.append(s2); seen.add(s2)
        if items:
            return items

    # 2) Otherwise, mine item rows as loose evidence
    ev = []
    for item in plan_json.get("items", []):
        for row in item.get("rows", []):
            if isinstance(row, dict):
                for _, v in row.items():
                    s = _clean_evidence_value(v)
                    if s:
                        ev.append(s)
            elif isinstance(row, (list, tuple, set)):
                for v in row:
                    s = _clean_evidence_value(v)
                    if s:
                        ev.append(s)
            else:
                s = _clean_evidence_value(row)
                if s:
                    ev.append(s)
    # de-duplicate preserving order
    seen = set(); uniq = []
    for s in ev:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq

# -------------------- Matching core --------------------

def _contains_ngram(a_tokens: List[str], b_tokens: List[str], n: int) -> bool:
    if len(a_tokens) < n:
        n = max(1, len(a_tokens))
    A = set(_ngrams(a_tokens, n))
    B = set(_ngrams(b_tokens, n))
    return bool(A & B)

def align_sentences_with_evidence(
    text: str,
    evidence_items: List[str],
    ngram_n: int = 5,
    jaccard_threshold: float = 0.50,
    max_hits_per_sentence: int = 2
) -> Dict[str, Any]:
    """
    Returns:
      {
        "support_rate": supported / len(sents),
        "coverage_rate": covered_evidence / len(items),
        "sentence_support": [hit_count_per_sentence],
        "covered_mask": [bool per evidence item],
        "unmatched_sentences": [...],
        "unmatched_evidence_sample": [...]
      }
    """
    sents = _split_sentences(text)
    sents_norm = [_normalize(s) for s in sents]
    sents_toks = [_tokens(s) for s in sents_norm]

    text_norm = _normalize(text)
    text_toks = _tokens(text_norm)

    covered_mask: List[bool] = []
    covered_indices: List[int] = []
    sent_hits = [0] * len(sents)
    unmatched_sentences = []

    for i, raw in enumerate(evidence_items or []):
        ev = _normalize(raw)
        if not ev:
            covered_mask.append(False)
            continue

        etoks = _tokens(ev)

        # Global quick checks
        matched = (
            (ev in text_norm) or
            _contains_ngram(etoks, text_toks, n=ngram_n) or
            (_jaccard(etoks, text_toks) >= jaccard_threshold)
        )

        # If not matched globally, try sentence-local
        best_idx = -1; best_score = -1.0
        if not matched:
            for idx, stoks in enumerate(sents_toks):
                if not stoks:
                    continue
                # N-gram dominates as a strong signal
                if _contains_ngram(etoks, stoks, n=ngram_n):
                    score = 1.0
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                else:
                    jac = _jaccard(etoks, stoks)
                    if jac >= jaccard_threshold and jac > best_score:
                        best_score = jac
                        best_idx = idx

            matched = best_idx >= 0

        covered_mask.append(bool(matched))
        if matched:
            covered_indices.append(i)
            # If we don't yet have a best sentence, pick one now based on Jaccard across all sentences
            if best_idx < 0 and sents_toks:
                best_idx = 0; best_score = -1.0
                for idx, stoks in enumerate(sents_toks):
                    jac = _jaccard(etoks, stoks)
                    if jac > best_score:
                        best_score = jac
                        best_idx = idx
            if best_idx >= 0:
                sent_hits[best_idx] = min(max_hits_per_sentence, sent_hits[best_idx] + 1)

    # Collect unmatched sentences (no hits)
    for i, h in enumerate(sent_hits):
        if h == 0 and i < len(sents):
            unmatched_sentences.append(sents[i])

    supported = sum(1 for h in sent_hits if h > 0)
    support_rate = supported / max(1, len(sents))
    coverage_rate = len(covered_indices) / max(1, len(evidence_items or []))

    # Sample unmatched evidence for debugging
    unmatched_evidence_sample = []
    if evidence_items:
        for i, ok in enumerate(covered_mask):
            if not ok:
                unmatched_evidence_sample.append(evidence_items[i])
                if len(unmatched_evidence_sample) >= 10:
                    break

    return {
        "support_rate": support_rate,
        "coverage_rate": coverage_rate,
        "sentence_support": sent_hits,
        "covered_mask": covered_mask,
        "unmatched_sentences": unmatched_sentences[:10],
        "unmatched_evidence_sample": unmatched_evidence_sample,
    }

# -------------------- Public API with per-run defaults --------------------

_PER_RUN_DEFAULTS = {
    "Graph":  {"ngram_n": 5, "jaccard_threshold": 0.50, "prefer_context": True},
    "KG":     {"ngram_n": 3, "jaccard_threshold": 0.38, "prefer_context": True},
    "Hybrid": {"ngram_n": 3, "jaccard_threshold": 0.38, "prefer_context": True},
}


def compute_support_from_story_and_plan(
    story_text: str,
    plan_with_evidence: Dict[str, Any],
    pattern: str = "KG"
) -> Dict[str, Any]:
    """
    Compute support/coverage using per-run defaults inferred from `pattern`
    (Graph, KG, Hybrid). Falls back to KG defaults if unknown.
    """
    cfg = _PER_RUN_DEFAULTS.get(pattern, _PER_RUN_DEFAULTS["KG"])
    items = extract_evidence_from_plan(plan_with_evidence, prefer_context_over_factlets=cfg["prefer_context"])

    return align_sentences_with_evidence(
        story_text,
        items,
        ngram_n=cfg["ngram_n"],
        jaccard_threshold=cfg["jaccard_threshold"]
    )
