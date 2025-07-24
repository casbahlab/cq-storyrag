import argparse
import json
from pathlib import Path

from evaluation_runner import NarrativeEvaluator

# Define persona profiles (can be loaded from JSON/YAML if preferred)
persona_profiles = {
    "Emma": {
        "keywords": [
            "Cultural awareness", "Empathy-driven learning", "Accessible narrative",
            "Structured onboarding", "First-time exposure", "No prior knowledge required",
            "Guided storytelling", "Educational framing", "Memory construction"
        ]
    },
    "Luca": {
        "keywords": [
            "Curated narrative", "Legacy formation", "Media aesthetics",
            "Artist sequencing", "Broadcast design", "Temporal structure",
            "Memory and omission", "Cross-stage performance", "Archival selectivity", "Symbolic moments"
        ]
    }
}

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def main(args):
    # Load inputs
    narrative = Path(args.narrative).read_text()
    facts = load_json(args.facts)
    persona = args.persona
    output_path = args.output or "evaluation_output.json"

    # Run evaluation
    evaluator = NarrativeEvaluator(persona_profiles)
    result = evaluator.evaluate(narrative, facts, persona=persona)
    evaluator.save_evaluation(result, narrative, facts, persona, output_path)

    # Display results
    print(f"\nEvaluation for {persona}:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a generated narrative.")
    parser.add_argument("--narrative", required=True, help="Path to file containing the generated narrative.")
    parser.add_argument("--facts", required=True, help="Path to JSON file with ground-truth fact list.")
    parser.add_argument("--persona", required=True, choices=["Emma", "Luca"], help="Persona to evaluate against.")
    parser.add_argument("--output", help="Optional path to save evaluation results.")

    args = parser.parse_args()
    main(args)
