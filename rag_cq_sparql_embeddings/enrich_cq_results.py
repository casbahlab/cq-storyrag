import json
import pandas as pd
import re

CQ_RESULTS_FILE = "embeddings/cq_results.json"
CQ_TEMPLATE_FILE = "data/WembleyRewindCQs_categories.csv"
ENRICHED_FILE = "embeddings/cq_results_enriched.json"

def normalize_cq_id(cq_id_raw: str):
    # Remove parenthetical labels and normalize
    cq_id_clean = re.sub(r"\s*\(.*?\)", "", cq_id_raw)
    cq_ids = [cid.strip().replace(" Extended", "") for cid in cq_id_clean.split("/") if cid.strip()]
    return cq_ids

def main():
    # Load mappings
    df_template = pd.read_csv(CQ_TEMPLATE_FILE)
    cq_to_category = {
        str(row["CQ_ID"]).strip(): str(row.get("Category", row.get("Section", "Uncategorized"))).strip()
        for _, row in df_template.iterrows()
        if str(row["CQ_ID"]).strip()
    }
    cq_to_persona = {
        str(row["CQ_ID"]).strip(): str(row.get("Persona", "")).strip()
        for _, row in df_template.iterrows()
        if str(row["CQ_ID"]).strip()
    }

    # Load CQ results
    with open(CQ_RESULTS_FILE, "r", encoding="utf-8") as f:
        cq_results = json.load(f)

    enriched_results = []

    for item in cq_results:
        cq_id_raw = item["CQ_ID"]
        facts = item.get("Results", [])

        cq_ids = normalize_cq_id(cq_id_raw)

        # Determine category and persona
        category = "Uncategorized"
        persona = ""
        for cid in cq_ids:
            if cid in cq_to_category:
                category = cq_to_category[cid]
                persona = cq_to_persona.get(cid, "")
                break  # First match wins

        enriched_item = {
            "CQ_ID": cq_id_raw,
            "CQ_List": cq_ids,
            "Category": category,
            "Persona": persona,
            "Results": facts
        }
        enriched_results.append(enriched_item)

    # Save enriched file
    with open(ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_results, f, indent=2, ensure_ascii=False)

    print(f"Enriched CQ results saved to {ENRICHED_FILE} ({len(enriched_results)} entries)")

if __name__ == "__main__":
    main()
