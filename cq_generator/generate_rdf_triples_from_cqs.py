import pandas as pd
import subprocess
import json
import re
import os

MODEL_NAME = "llama3.1"
INPUT_FILE = "output/WembleyRewindCQs - atomic_facts.csv"
OUTPUT_FILE = "output/wembley_cqs_with_rdf_output.csv"
os.makedirs("output", exist_ok=True)

def call_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()

def format_prompt(question, atomic_fact):
    return f"""
You are an ontology assistant. Given a competency question and a supporting atomic fact, generate RDF-style semantic triples following schema.org.

Competency Question:
"{question}"

Atomic Fact:
"{atomic_fact}"

Return a JSON array of triples. Each triple must be an object with keys:
- "subject"
- "predicate"
- "object"

Use schema.org terms where applicable.
ONLY return a valid JSON list of triples. No text or markdown.
"""

def generate_rdf():
    df = pd.read_csv(INPUT_FILE)
    rdf_records = []

    for idx, row in df.iterrows():
        cq = row.get("CQ_Text", "")
        fact = row.get("Atomic_Fact", "")
        prompt = format_prompt(cq, fact)
        response = call_ollama(prompt)

        try:
            match = re.search(r"\[(.*)\]", response, re.DOTALL)
            json_str = "[" + match.group(1).strip() + "]" if match else response
            triples = json.loads(json_str)

            for triple in triples:
                triple.update({
                    "CQ_ID": row.get("CQ_ID", ""),
                    "Question": cq,
                    "Atomic_Fact": fact,
                    "Persona": row.get("Persona", ""),
                    "Narrative_Category": row.get("Narrative_Category", ""),
                    "Original_Sentence": row.get("Original_Sentence", ""),
                    "New Sub Question": row.get("New Sub Question", ""),
                })
                rdf_records.append(triple)

        except Exception as e:
            print(f"[ERROR] Failed for CQ_ID {row.get('CQ_ID', '')}: {e}")
            print("Response:\n", response)

    pd.DataFrame(rdf_records).to_csv(OUTPUT_FILE, index=False)
    print(f"RDF triples saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_rdf()
