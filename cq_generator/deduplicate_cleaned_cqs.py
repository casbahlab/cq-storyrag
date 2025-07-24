
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
import os

INPUT_FILE = "output/combined_CQs_with_expanded_facts.csv"
OUTPUT_FILE = "output/unique_cleaned_CQs.csv"
SIMILARITY_THRESHOLD = 0.92  # Adjustable based on quality

def normalize_question(q):
    return q.strip().lower()

def group_similar_questions(questions, threshold=SIMILARITY_THRESHOLD):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()


    grouped = []
    used = set()

    for i, q in enumerate(questions):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(questions)):
            if j not in used and similarity_matrix[i][j] >= threshold:
                group.append(j)
                used.add(j)
        grouped.append(group)
    return grouped

def main():
    os.makedirs("output", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df["Cleaned_CQ"] = df["Cleaned_CQ"].astype(str)

    all_questions = df["Cleaned_CQ"].apply(normalize_question).tolist()
    print(f"Total questions: {len(all_questions)}")

    groups = group_similar_questions(all_questions)

    unique_rows = []
    for group in groups:
        # Take the first question in the group as the representative
        representative_idx = group[0]
        representative_question = df.iloc[representative_idx]["Cleaned_CQ"]
        personas = df.iloc[group]["Persona"].dropna().unique().tolist()
        personas = [p for p in personas if p]  # Remove empty

        row = {
            "Cleaned_CQ": representative_question,
            "Source_CQ_IDs": "; ".join(df.iloc[group]["CQ_ID"].astype(str).tolist()),
            "Personas": "; ".join(personas),
            "CQ_Types": "; ".join(df.iloc[group]["CQ_Type"].dropna().unique().tolist())
        }
        unique_rows.append(row)

    pd.DataFrame(unique_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved unique questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
