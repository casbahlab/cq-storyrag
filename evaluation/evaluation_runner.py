import json
from typing import Dict

from metrics import compute_factual_match_score
from style_alignment import compute_keyword_alignment_score, extract_matched_keywords

class NarrativeEvaluator:
    def __init__(self, persona_profiles: Dict[str, Dict]):
        """
        Args:
            persona_profiles: A dictionary where each key is a persona name and the value is a dict with 'keywords' list.
        """
        self.persona_profiles = persona_profiles

    def evaluate(
        self,
        narrative: str,
        facts: list,
        persona: str
    ) -> Dict[str, float]:
        """
        Run the full evaluation suite for a given narrative.

        Args:
            narrative: The generated story to evaluate.
            facts: List of ground-truth fact strings.
            persona: Name of the persona (e.g., 'Emma' or 'Luca').

        Returns:
            A dictionary with evaluation scores.
        """
        factual_score = compute_factual_match_score(narrative, facts)

        if persona not in self.persona_profiles:
            raise ValueError(f"Persona '{persona}' not found in profiles.")

        keywords = self.persona_profiles[persona]["keywords"]
        style_score = compute_keyword_alignment_score(narrative, keywords)
        matched_keywords = extract_matched_keywords(narrative, keywords)

        return {
            "factual_score": round(factual_score, 3),
            "style_alignment_score": round(style_score, 3),
            "matched_keywords": matched_keywords
        }

    def save_evaluation(
        self,
        result: Dict[str, float],
        narrative: str,
        facts: list,
        persona: str,
        save_path: str
    ):
        """Save the evaluation output to a file."""
        output = {
            "persona": persona,
            "narrative": narrative,
            "facts": facts,
            "evaluation": result
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)
