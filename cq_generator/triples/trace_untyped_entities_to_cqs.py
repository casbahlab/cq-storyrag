import json
from collections import defaultdict

# Load untyped entities
with open("untyped_entities.json") as f:
    untyped_entities = set(json.load(f))

# Load triples
with open("triples_from_gpt.json") as f:
    triples_data = json.load(f)

# Mapping from entity URI to list of (CQ_ID, CQ_Text)
entity_usage_map = defaultdict(list)

# Go through each CQ and its triples
for item in triples_data:
    cq_id = item.get("CQ_ID")
    cq_text = item.get("CQ_Text", "")
    answer = item.get("Answer", "")
    for triple in item.get("Triples", []):
        subj = triple.get("subject")
        obj = triple.get("object")
        for entity in [subj, obj]:
            if entity in untyped_entities:
                entity_usage_map[entity].append({
                    "CQ_ID": cq_id,
                    "CQ_Text": cq_text,
                    "Answer": answer
                })

# Save to file
with open("untyped_entity_usage_trace.json", "w") as f:
    json.dump(entity_usage_map, f, indent=2)

print(f"Saved mapping of untyped entities to CQs in 'untyped_entity_usage_trace.json'")
