# evaluation/metrics.py

from difflib import SequenceMatcher
from typing import List, Tuple

from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Load summarization / NLI pipeline (lazy-load if you plan to replace later)
try:
    nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except:
    nli = None  # If offline, skip NLI checks for now

def evaluate_factuality(narrative: str, triples: List[Tuple[str, str, str]]) -> float:
    """
    Match narrative sentences with KG facts using fuzzy token similarity.
    Returns a factuality score between 0 and 1.
    """
    if not triples:
        return 0.0

    sentences = sent_tokenize(narrative)
    match_count = 0

    for triple in triples:
        s, p, o = map(lambda x: x.lower(), triple)
        for sent in sentences:
            sent_l = sent.lower()
            if s in sent_l and (p in sent_l or p.split("/")[-1] in sent_l) and o in sent_l:
                match_count += 1
                break
            elif SequenceMatcher(None, sent_l, f"{s} {p} {o}").ratio() > 0.6:
                match_count += 0.5  # partial match

    return min(1.0, match_count / max(1, len(triples)))

def evaluate_fluency(narrative: str) -> float:
    """
    Naive proxy: score based on average sentence length and grammatical smoothness.
    Could replace with GPT/LLM scoring later.
    """
    sentences = sent_tokenize(narrative)
    if not sentences:
        return 0.0

    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
    ideal_range = (8, 25)
    fluency = 1.0 if ideal_range[0] <= avg_length <= ideal_range[1] else 0.5
    return fluency


def compute_factual_match_score(narrative: str, facts: List[str]) -> float:
    """
    Wrapper to evaluate factuality using string-based fact matching.
    If facts are RDF triples (3-tuples), it delegates directly.
    If facts are strings, converts to pseudo-triples.
    """
    if not facts:
        return 0.0

    # If facts look like triples, delegate
    if isinstance(facts[0], (list, tuple)) and len(facts[0]) == 3:
        return evaluate_factuality(narrative, facts)

    # Otherwise, assume plain text facts and use fuzzy matching
    match_count = 0
    sentences = sent_tokenize(narrative)

    for fact in facts:
        fact_l = fact.lower()
        for sent in sentences:
            sent_l = sent.lower()
            if fact_l in sent_l or SequenceMatcher(None, sent_l, fact_l).ratio() > 0.6:
                match_count += 1
                break

    return min(1.0, match_count / len(facts))
