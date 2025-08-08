import csv
import json

INPUT_CSV = "data/WembleyRewindCQs_categories.csv"
OUTPUT_JSON = "embeddings/cq_persona_category.json"

cq_mapping = {}

with open(INPUT_CSV, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        cq_id = row["CQ_ID"].strip()
        persona = row.get("Persona", "Generic").strip()
        category = row.get("Category", "Unknown").strip()
        cq_mapping[cq_id] = {
            "persona": persona,
            "category": category
        }

with open(OUTPUT_JSON, "w", encoding="utf-8") as jsonfile:
    json.dump(cq_mapping, jsonfile, indent=2)

print(f"[INFO] Mapping file saved to {OUTPUT_JSON} with {len(cq_mapping)} entries.")
