import json
import subprocess
from datetime import datetime

INPUT_FILE = "embeddings/cq_results.json"
OUTPUT_FILE = "embeddings/persona_narratives.json"
OLLAMA_MODEL = "llama3.1"  # Change to any local model pulled in Ollama

# Persona prompt templates
PROMPT_TEMPLATES = {
    "Emma": """You are a friendly museum storyteller.

Persona: Emma (Curious Novice)
Goal: Cultural awareness, empathy-driven learning, accessible narrative.

Here are some facts from the Live Aid Knowledge Graph:
{facts}

Task:
1. Generate a short, engaging narrative for a first-time visitor.
2. Use simple and friendly language.
3. Highlight the cultural and humanitarian aspects.
4. Avoid technical or overly detailed facts.
""",
    "Luca": """You are a curated music history narrator.

Persona: Luca (Informed Enthusiast)
Goal: Curated narrative, legacy formation, artist and media focus.

Here are some facts from the Live Aid Knowledge Graph:
{facts}

Task:
1. Generate a structured, detailed narrative for an informed visitor.
2. Focus on performance sequences, artist context, and broadcast legacy.
3. Maintain a professional, curated tone.
"""
}


def generate_narrative_ollama(persona, facts, model=OLLAMA_MODEL):
    """Generate a persona-based narrative using a local Ollama model."""
    facts_str = json.dumps(facts, indent=2, ensure_ascii=False)
    prompt = PROMPT_TEMPLATES[persona].format(facts=facts_str)

    # Run ollama in subprocess to generate the narrative
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )

    return result.stdout.decode("utf-8").strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        cq_results = json.load(f)

    all_narratives = []
    for cq in cq_results:
        cq_id = cq["CQ_ID"]
        cq_text = cq.get("CQ_Text", "")
        facts = cq.get("Results", [])

        for persona in ["Emma", "Luca"]:
            print(f"Generating {persona} narrative for {cq_id} ...")
            narrative = generate_narrative_ollama(persona, facts)

            all_narratives.append({
                "CQ_ID": cq_id,
                "CQ_Text": cq_text,
                "Persona": persona,
                "Facts": facts,
                "Narrative": narrative,
                "GeneratedAt": datetime.now().isoformat()
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_narratives, f, indent=2, ensure_ascii=False)

    print(f"\nPersona-based narratives saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
