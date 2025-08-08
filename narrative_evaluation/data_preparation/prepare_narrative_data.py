import json

def prepare_narrative_data(narrative_id, persona_id, persona_desc, narrative_text, cqs, facts):
    data = {
        "narrative_id": narrative_id,
        "persona": {
            "id": persona_id,
            "description": persona_desc
        },
        "narrative_text": narrative_text,
        "competency_questions": cqs,
        "retrieved_facts": [{"triple": fact} for fact in facts]
    }
    return data

# Example usage
narrative_example = prepare_narrative_data(
    narrative_id="emma-s1-001",
    persona_id="Emma",
    persona_desc="Curious Novice, seeks accessible educational content.",
    narrative_text="Live Aid was a global charity event held in July 1985...",
    cqs=["CQ-E1", "CQ-E2"],
    facts=[
        "ex:LiveAid1985 schema:startDate '1985-07-13'",
        "ex:LiveAid1985 schema:location ex:Global"
    ]
)

# Save to JSON file
with open('input/sample_narrative.json', 'w') as f:
    json.dump(narrative_example, f, indent=2)
