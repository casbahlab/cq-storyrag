import os
import subprocess
import json
import pandas as pd
from pathlib import Path
import sys


def call_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3.1", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()


def format_prompt(sentence, context):
    return f"""
You are generating ontology-aligned competency questions from a scenario.

Context:
\"\"\"
{context}
\"\"\"

Based on the following sentence from the context, generate a structured competency question.

DO NOT use persona names like "Emma", "Luca", or pronouns like "she/he/they" in the question. Instead, use **neutral, domain-specific references** (e.g., "a user with limited musical knowledge").

Return JSON with these keys:
- "Scenario_Sentence"
- "Generated_CQ"
- "Cleaned_CQ" (if needed, rewrite for clarity)
- "CQ_Type" (Agent-based, Procedural, Causal, Temporal, Fact-based, General Inquiry)

Sentence: "{sentence}"

Respond ONLY with a JSON object — no markdown, no explanation.
"""


def generate_cqs_from_facts(fact_file, scenario_file):
    persona = Path(fact_file).stem.split("_")[0].capitalize()
    with open(fact_file, "r", encoding="utf-8") as f:
        facts = [line.strip() for line in f if
                 line.strip() and not line.startswith("Note") and "factual statements" not in line]

    with open(scenario_file, "r", encoding="utf-8") as f:
        context = f.read()

    results = []
    for fact in facts:
        prompt = format_prompt(fact, context)
        response = call_ollama(prompt)

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            cleaned_json = response[json_start:json_end]
            json_obj = json.loads(cleaned_json)
            json_obj["Persona"] = persona
            results.append(json_obj)
        except Exception as e:
            print(f"Failed to parse response for fact: {fact}")
            print("Response was:\n", response)
            print("Error:", e)

    output_path = f"output/{persona.lower()}_generated_CQs_llama_facts.csv"
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_cqs_from_facts.py <fact_file.txt> <scenario_file.txt>")
        sys.exit(1)

    fact_file = sys.argv[1]
    scenario_file = sys.argv[2]
    generate_cqs_from_facts(fact_file, scenario_file)
