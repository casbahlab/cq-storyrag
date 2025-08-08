import json
from datetime import datetime

# === INPUT FILES ===
EMMA_FILE = "persona_emma_cleaned.json"
LUCA_FILE = "persona_luca_cleaned.json"

# === OUTPUT FILES ===
OUTPUT_JSON = "cleaned_persona_narratives.json"
OUTPUT_MD = "cleaned_persona_narratives.md"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def deduplicate_facts(facts):
    """Remove duplicates while preserving order."""
    seen = set()
    unique_facts = []
    for f in facts:
        if f not in seen:
            unique_facts.append(f)
            seen.add(f)
    return unique_facts

def merge_narratives(emma, luca):
    """Merge two persona narratives into one canonical JSON structure."""
    combined = {
        "Project": "Wembley Rewind – Persona Narratives",
        "GeneratedAt": datetime.now().isoformat(),
        "Personas": []
    }

    for persona_data in [emma, luca]:
        curated_entry = deduplicate_facts(persona_data.get("CuratedEntryFacts", []))
        curated_core = deduplicate_facts(persona_data.get("CuratedCoreFacts", []))
        curated_exit = deduplicate_facts(persona_data.get("CuratedExitFacts", []))

        combined["Personas"].append({
            "Persona": persona_data["Persona"],
            "GeneratedAt": persona_data.get("GeneratedAt"),
            "CuratedEntryFacts": curated_entry,
            "CuratedCoreFacts": curated_core,
            "CuratedExitFacts": curated_exit,
            "FullNarrative": persona_data["FullNarrative"].strip()
        })

    return combined

def save_md(combined, md_file):
    """Generate a clean Markdown file for the dissertation appendix."""
    lines = ["# Wembley Rewind – Persona Narratives\n"]
    for persona in combined["Personas"]:
        lines.append(f"## Persona: {persona['Persona']}\n")
        lines.append(persona["FullNarrative"].strip())
        lines.append("\n---\n")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    emma = load_json(EMMA_FILE)
    luca = load_json(LUCA_FILE)

    combined = merge_narratives(emma, luca)

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # Save Markdown
    save_md(combined, OUTPUT_MD)

    print(f"[INFO] Final combined narratives saved to:\n- {OUTPUT_JSON}\n- {OUTPUT_MD}")

if __name__ == "__main__":
    main()
