import json
import os

INPUT_FILE = "embeddings/cq_results_enriched.json"
OUTPUT_FILE = "embeddings/cq_results_for_embeddings.json"

def collect_external_links(result: dict):
    """Collect all external links from a single result dict"""
    links = []
    for k, v in result.items():
        if isinstance(v, str) and (
            v.startswith("http") or k.endswith("URL") or k in ["sameAs", "seeAlso", "subjectOf"]
        ):
            links.append(v)
    return list(set(links))  # remove duplicates

def build_embedding_text(cq_entry: dict):
    """Build a rich embedding text snippet from a CQ entry"""
    cq_id = cq_entry["CQ_ID"]
    cq_text = cq_entry["CQ_Text"]
    persona = cq_entry.get("Persona", "Generic")
    category = cq_entry.get("Category", "Uncategorized")

    snippets = []
    external_links = []

    # Build text for each result
    for res in cq_entry.get("Results", []):
        labels = [f"{k}: {v}" for k, v in res.items() if "Label" in k and v]
        comment = res.get("comment") or res.get("rdfs:comment") or ""
        if labels or comment:
            snippets.append(" | ".join(labels + ([comment] if comment else [])))

        # Collect external links
        external_links.extend(collect_external_links(res))

    # Final embedding text combines persona context, CQ text, and semantic snippets
    embedding_text = f"[Persona: {persona} | Category: {category} | CQ: {cq_id}]\n"
    embedding_text += f"Question: {cq_text}\n"

    if snippets:
        embedding_text += "Facts:\n" + "\n".join(f"- {s}" for s in snippets)
    else:
        embedding_text += "Facts: None\n"

    return embedding_text.strip(), list(set(external_links))

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        cq_data = json.load(f)

    embedding_entries = []
    for cq in cq_data:
        embedding_text, external_links = build_embedding_text(cq)

        embedding_entries.append({
            "CQ_ID": cq["CQ_ID"],
            "CQ_Text": cq["CQ_Text"],
            "Persona": cq.get("Persona", "Generic"),
            "Category": cq.get("Category", "Uncategorized"),
            "EmbeddingInput": embedding_text,
            "ExternalLinks": external_links,
            "OriginalResults": cq.get("Results", [])
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedding_entries, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Embedding-ready JSON saved to {OUTPUT_FILE} with {len(embedding_entries)} entries.")

if __name__ == "__main__":
    main()
