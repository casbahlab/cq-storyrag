import json

# ---- CONFIG ----
CQ_RESULTS_FILE = "embeddings/cq_results.json"            # From Step 1
CQ_PERSONA_FILE = "embeddings/cq_persona_category.json"   # Derived from CSV
OUTPUT_FILE = "embeddings/cq_results_enriched.json"       # Ready for embeddings

with open(CQ_RESULTS_FILE, "r", encoding="utf-8") as f:
    cq_results = json.load(f)

with open(CQ_PERSONA_FILE, "r", encoding="utf-8") as f:
    cq_persona_map = json.load(f)

enriched_results = []

for cq in cq_results:
    cq_id = cq["CQ_ID"]
    cq_text = cq["CQ_Text"]
    cq_list = cq["CQ_List"]
    results = cq.get("Results", [])

    # Determine persona & category for this CQ
    persona = "Generic"
    category = "Unknown"

    # Priority: first CQ_ID in the list
    for cid in cq_list:
        if cid in cq_persona_map:
            persona = cq_persona_map[cid]["persona"]
            category = cq_persona_map[cid]["category"]
            break

    enriched_results.append({
        "CQ_ID": cq_id,
        "CQ_List": cq_list,
        "CQ_Text": cq_text,
        "Persona": persona,
        "Category": category,
        "Results": results
    })

# Save the enriched file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(enriched_results, f, indent=2, ensure_ascii=False)

print(f"[INFO] Enriched results with persona & category saved to {OUTPUT_FILE}")
