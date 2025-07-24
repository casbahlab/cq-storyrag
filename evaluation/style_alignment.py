from typing import List
import re

def compute_keyword_alignment_score(
    narrative: str, persona_keywords: List[str]
) -> float:
    """
    Compute a simple keyword alignment score between the narrative and persona keywords.

    Args:
        narrative: The generated story.
        persona_keywords: A list of keywords representing style/tone goals.

    Returns:
        A float score between 0 and 1 representing keyword alignment.
    """
    normalized_text = narrative.lower()
    matches = sum(1 for kw in persona_keywords if re.search(rf'\b{re.escape(kw.lower())}\b', normalized_text))
    return matches / len(persona_keywords) if persona_keywords else 0.0

def extract_matched_keywords(narrative: str, persona_keywords: List[str]) -> List[str]:
    """
    Return the list of keywords from persona that are actually used in the narrative.

    Args:
        narrative: The generated narrative text.
        persona_keywords: Style/tone keywords for the target persona.

    Returns:
        List of matched keywords.
    """
    normalized_text = narrative.lower()
    return [kw for kw in persona_keywords if re.search(rf'\b{re.escape(kw.lower())}\b', normalized_text)]
