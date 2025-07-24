import os
import sys
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")


# Heuristic categorizer
def categorize_sentences(sentences):
    n = len(sentences)
    entry_end = max(1, int(n * 0.2))  # First 20%
    exit_start = max(n - int(n * 0.2), entry_end + 1)  # Last 20%

    categorized = []

    for i, sent in enumerate(sentences):
        if i < entry_end:
            category = "Entry"
        elif i >= exit_start:
            category = "Exit"
        else:
            category = "Core"
        categorized.append((sent.strip(), category))

    return categorized


def process_scenario(filepath):
    persona = os.path.basename(filepath).split("_")[0].capitalize()

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sentences = sent_tokenize(content)
    categorized_data = categorize_sentences(sentences)

    df = pd.DataFrame(categorized_data, columns=["Scenario_Sentence", "Narrative_Category"])
    df["Persona"] = persona

    output_file = f"output/{persona.lower()}_scenario_categorized.csv"
    os.makedirs("output", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved categorized output to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python categorize_scenario_sentences.py scenarios/emma_scenario.txt")
    else:
        process_scenario(sys.argv[1])
