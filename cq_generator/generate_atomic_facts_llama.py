import pandas as pd
import ollama

# === CONFIG ===
INPUT_CSV = "output/WembleyRewindCQs - scoped_narrativecategory.csv"
OUTPUT_CSV = "output/llama_atomic_facts_with_cq_text.csv"
MODEL = "llama3.1"

# Load scenario contexts
with open("scenarios/emma_scenario.txt") as f:
    emma_context = f.read()
with open("scenarios/luca_scenario.txt") as f:
    luca_context = f.read()

# Load the input CQ file
df = pd.read_csv(INPUT_CSV)

# Lowercase patterns used to detect meta-content
META_PATTERNS = [
    "here is", "here are", "note:", "this meets the criteria",
    "these sentences", "rephrased", "rewritten", "breakdown",
    "commentary", "clarification", "grammatical", "explanation",
    '"', "“", "”"
]

def is_meta_line(text):
    lower = str(text).lower()
    return any(p in lower for p in META_PATTERNS)

# LLaMA prompt
def build_prompt(context, fact):
    return f"""You are an expert knowledge engineer. Rewrite the following fact as standalone atomic factual sentences. Each sentence must:
- Be grammatically complete
- Start with the subject (e.g., “Live Aid...”)
- Not use vague pronouns like "it" or "they"
- Represent one and only one fact
- Be suitable for RDF triple extraction

Do not include commentary, explanations, or phrases like "here is the breakdown."

Fact:
{fact}

Atomic facts:
1."""

# Run generation
results = []
for _, row in df.iterrows():
    persona = row["Personas"].strip().lower()
    context = emma_context if persona == "emma" else luca_context
    fact = str(row["Scenario_Sentence"]).strip()

    if not fact:
        continue

    prompt = build_prompt(context, fact)
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        output = response["message"]["content"]
    except Exception as e:
        print(f"Error on {row['CQ_ID']}: {e}")
        continue

    # Extract and clean atomic facts
    lines = output.strip().split("\n")
    atomic_facts = [
        line.strip("1234567890. ").strip()
        for line in lines
        if line.strip() and not is_meta_line(line)
    ]

    for atomic in atomic_facts:
        results.append({
            "CQ_ID": row["CQ_ID"],
            "Persona": row["Personas"],
            "Narrative_Category": row["Narrative Category"],
            "Original_Sentence": fact,
            "Atomic_Fact": atomic
        })

# Convert and deduplicate
out_df = pd.DataFrame(results)
out_df = out_df.dropna(subset=["Atomic_Fact"])
out_df = out_df[out_df["Atomic_Fact"].str.strip() != ""]
out_df = out_df.drop_duplicates(subset=["CQ_ID", "Atomic_Fact"]).reset_index(drop=True)

# Merge in original CQ text for traceability
cq_texts = df[["CQ_ID", "Manually validated"]].rename(columns={"Manually validated": "CQ_Text"})
final_df = out_df.merge(cq_texts, on="CQ_ID", how="left")

# Save result
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Atomic facts generated and saved to: {OUTPUT_CSV}")
