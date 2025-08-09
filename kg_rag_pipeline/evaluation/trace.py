# evaluation/trace.py
# Python 3.9

import re
from typing import Dict, Any, List
# IMPORTANT: reuse the SAME helpers you put in enhanced_eval.py to keep behavior identical
from evaluation.enhanced_eval import _basic_norm, _mentioned, _values_from_item

_SENT_SPLIT = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9“"])')

def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Simple splitter that keeps punctuation in the sentence
    sents = _SENT_SPLIT.split(text)
    # Fallback: if it failed to split, return whole text
    return [s.strip() for s in sents if s.strip()] or [text]

def trace_fact_alignment(narrative: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each fact item, find narrative sentence(s) that mention at least one of its values.
    Returns a structured trace for easy inspection.
    """
    sents = _split_sentences(narrative)
    nn_sents = [_basic_norm(s) for s in sents]

    traces: List[Dict[str, Any]] = []
    index_hits_per_sentence = {i: 0 for i in range(len(sents))}

    for category in ["Entry", "Core", "Exit"]:
        for it in (facts.get(category, []) or []):
            cq_id = it.get("cq_id")
            question = it.get("question")
            vals = _values_from_item(it)

            matched = []
            for i, (raw_sent, norm_sent) in enumerate(zip(sents, nn_sents)):
                # A sentence "matches" a fact if ANY value is mentioned
                local_hits = [v for v in vals if _mentioned(norm_sent, v)]
                if local_hits:
                    matched.append({
                        "sentence_index": i,
                        "sentence": raw_sent,
                        "matched_values": local_hits[:5]  # keep it tidy
                    })
                    index_hits_per_sentence[i] += 1

            traces.append({
                "category": category,
                "cq_id": cq_id,
                "question": question,
                "matches": matched
            })

    # Also report sentences with no fact matches (for diagnostics)
    unmatched_sentences = [
        {"sentence_index": i, "sentence": sents[i]}
        for i in range(len(sents)) if index_hits_per_sentence[i] == 0
    ]

    return {
        "facts_trace": traces,
        "unmatched_sentences": unmatched_sentences
    }
