import argparse
from retriever.kg_retriever import retrieve_triples
from generator.narrative_generator import generate_story_with_ollama
from enrichment.enricher import enrich_graph_with_external_links
import json
from pathlib import Path
from datetime import datetime

def save_for_evaluation(narrative: str, triples: list, persona: str, output_dir: str = "evaluation_inputs") -> dict:
    """
    Save narrative, KG facts, and persona into a timestamped file set for evaluation.
    Returns dictionary with paths to saved files.
    """
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{persona.lower()}_{timestamp}"

    narrative_path = Path(output_dir) / f"{base_name}_narrative.txt"
    facts_path = Path(output_dir) / f"{base_name}_facts.json"

    # Save narrative
    with open(narrative_path, "w") as f:
        f.write(narrative)

    # Save facts
    with open(facts_path, "w") as f:
        json.dump([f"{s} {p} {o}" for s, p, o in triples], f, indent=2)

    print(f"\nEvaluation inputs saved:\n- Narrative: {narrative_path}\n- Facts: {facts_path}")

    return {
        "narrative": str(narrative_path),
        "facts": str(facts_path),
        "persona": persona
    }

def main():
    parser = argparse.ArgumentParser(description="Run persona-driven narrative over KG")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--persona", type=str, default="Emma", help="Persona name (Emma or Luca)")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name")

    args = parser.parse_args()

    # Step 1: Query the KG
    print(f"Querying KG for: {args.query}")
    triples = retrieve_triples(args.query)
    if not triples:
        print("No results found for the query.")
        return

    enriched_triples = enrich_graph_with_external_links(triples)
    triples += enriched_triples

    # Step 2: Generate Narrative with Persona Context
    print(f"\nGenerating narrative for persona: {args.persona}")
    story = generate_story_with_ollama(triples, persona=args.persona, model=args.model)

    # Step 3: Output Result
    print("\nGenerated Narrative:\n")
    print(story)

    return save_for_evaluation(story, triples, args.persona)

if __name__ == "__main__":
    main()
