import pandas as pd
import subprocess
import json
import os
import re
from datetime import datetime

INPUT_FILE = "output/combined_CQs_with_ids.csv"
SCENARIO_FILES = ["scenarios/emma_scenario.txt", "scenarios/luca_scenario.txt"]
OUTPUT_FILE = f"output/fact_based_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
MODEL_NAME = "llama3"

os.makedirs("output", exist_ok=True)

import re

def clean_scenario_sentence(sentence):
    return re.sub(r'^\d+\.\s*', '', sentence).strip()

def load_combined_context():
    all_texts = []
    for path in SCENARIO_FILES:
        with open(path, "r", encoding="utf-8") as f:
            all_texts.append(f.read())
    return "\n".join(all_texts)

def call_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", MODEL_NAME, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout.strip()

def format_prompt(original_cq, sentence, context):
    return f"""
You are given a procedural, causal, or temporal competency question and its original scenario sentence.
Break it down into **simple fact-based questions**.

Your task:
- Use the combined scenario context to guide your breakdown.
- Each fact-based question should be specific, atomic, and reflect concrete details.
- Do NOT use persona names like "Emma", "Luca", or pronouns like "she/he/they".
- Use **neutral, domain-specific phrasing** (e.g., "a visitor to the Live Aid exhibit").

Return ONLY a JSON list. Each item should have:
- "Fact_CQ": the new fact-based question
- "Source_CQ_ID": original ID
- "Original_Question": original procedural/causal/temporal CQ
- "Scenario_Sentence": original scenario sentence

Context:
\"\"\"
{context}
\"\"\"

Question: "{original_cq}"
Scenario Sentence: "{sentence}"
"""

def decompose_complex_cqs():
    df = pd.read_csv(INPUT_FILE)
    context = load_combined_context()

    # Process Procedural, Causal, and Temporal CQs
    filtered_df = df[df["CQ_Type"].isin(["Procedural", "Causal", "Temporal"])]
    fact_cqs = []

    for _, row in filtered_df.iterrows():
        cq_id = row["CQ_ID"]
        original_cq = row["Generated_CQ"]
        sentence = row["Scenario_Sentence"]
        persona = row.get("Persona", "")

        prompt = format_prompt(original_cq, sentence, context)
        response = call_ollama(prompt)

        try:
            match = re.search(r"\[(.*)\]", response, re.DOTALL)
            if match:
                response_json = json.loads("[" + match.group(1).strip() + "]")
            else:
                response_json = json.loads(response)

            for item in response_json:
                item["Source_CQ_ID"] = cq_id
                item["Original_Question"] = original_cq
                item["Scenario_Sentence"] = sentence
                item["CQ_Type"] = "Fact-based"
                item["Persona"] = persona
                item["Rephrased_CQ"] = item["Fact_CQ"]
                item["Cleaned_CQ"] = item["Fact_CQ"]
                fact_cqs.append(item)

        except Exception as e:
            print(f"\nFailed to parse response for {cq_id}\n{response}\nError: {e}\n")

    # Assign CQ_IDs like CQ-XX.Y
    fact_rows = []
    fact_df = pd.DataFrame(fact_cqs)
    for _, row in fact_df.iterrows():
        source_id = row["Source_CQ_ID"].strip()
        siblings = fact_df[fact_df["Source_CQ_ID"] == source_id]
        index = list(siblings.index).index(row.name) + 1
        row["CQ_ID"] = f"{source_id}.{index}"
        fact_rows.append(row)

    # Merge and save
    final_df = pd.concat([df, pd.DataFrame(fact_rows)], ignore_index=True)
    final_df["Scenario_Sentence"] = final_df["Scenario_Sentence"].apply(clean_scenario_sentence)
    final_df.to_csv("output/combined_CQs_with_expanded_facts.csv", index=False)
    print("Appended all fact-based expansions to: output/combined_CQs_with_expanded_facts.csv")

if __name__ == "__main__":
    decompose_complex_cqs()
