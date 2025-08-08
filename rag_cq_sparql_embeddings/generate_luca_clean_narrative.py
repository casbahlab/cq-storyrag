import json
from datetime import datetime
from ollama import chat

INPUT_JSON = "embeddings/persona_full_narratives_streaming_8.json"
OUTPUT_JSON = "persona_luca_cleaned.json"
OUTPUT_MD = "persona_luca_cleaned.md"

def curate_facts(entry_facts, core_facts, exit_facts, top_entry=7, top_core=10, top_exit=7):
    """Trim and simplify facts for Luca persona, keeping some media richness."""
    def simplify(fact):
        # Keep first part before '|' for clarity; skip overly technical IDs
        return fact.split('|')[0].strip()
    return (
        [simplify(f) for f in entry_facts[:top_entry]],
        [simplify(f) for f in core_facts[:top_core]],
        [simplify(f) for f in exit_facts[:top_exit]]
    )

def clean_narrative(text: str) -> str:
    """Ensure only one 3-part narrative remains."""
    sections = {"Entry": None, "Core": None, "Exit": None}
    current = None
    lines = text.splitlines()

    for line in lines:
        if line.strip().startswith("## Entry") and sections["Entry"] is None:
            current = "Entry"; sections["Entry"] = []
            continue
        elif line.strip().startswith("## Core") and sections["Core"] is None:
            current = "Core"; sections["Core"] = []
            continue
        elif line.strip().startswith("## Exit") and sections["Exit"] is None:
            current = "Exit"; sections["Exit"] = []
            continue
        if current and sections[current] is not None:
            sections[current].append(line)

    final_blocks = []
    for key in ["Entry", "Core", "Exit"]:
        if sections[key]:
            final_blocks.append(f"## {key}\n" + "\n".join(sections[key]).strip())
    return "\n\n".join(final_blocks).strip()

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        narratives = json.load(f)

    luca = next((p for p in narratives if p["Persona"] == "Luca"), None)
    if not luca:
        print("[ERROR] No Luca narrative found!")
        return

    # Curate and simplify Luca's facts
    entry_facts, core_facts, exit_facts = curate_facts(
        luca["EntryFactsUsed"], luca["CoreFactsUsed"], luca["ExitFactsUsed"]
    )

    prompt = f"""
You are a curated music history narrator.

Persona: Luca (Informed Enthusiast)
Goal: Deliver a structured, artist and media-focused narrative with historical context.
Tone: Analytical, curated, like an exhibition guide for an informed visitor.

We are creating ONE cohesive story in 3 sections using these curated facts:

Entry:
{json.dumps(entry_facts, indent=2, ensure_ascii=False)}

Core:
{json.dumps(core_facts, indent=2, ensure_ascii=False)}

Exit:
{json.dumps(exit_facts, indent=2, ensure_ascii=False)}

Task:
Write exactly ONE narrative with 3 sections.
Use the format:

## Entry
<entry narrative>

## Core
<core narrative>

## Exit
<exit narrative>
""".strip()

    print("[INFO] Generating cleaned Luca narrative...")
    resp = chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    raw_narrative = resp["message"]["content"]
    cleaned = clean_narrative(raw_narrative)

    result = {
        "Persona": "Luca",
        "GeneratedAt": datetime.now().isoformat(),
        "CuratedEntryFacts": entry_facts,
        "CuratedCoreFacts": core_facts,
        "CuratedExitFacts": exit_facts,
        "FullNarrative": cleaned
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Persona Narrative: Luca\n\n" + cleaned)

    print(f"[INFO] Cleaned Luca narrative saved:\n- {OUTPUT_JSON}\n- {OUTPUT_MD}")

if __name__ == "__main__":
    main()
