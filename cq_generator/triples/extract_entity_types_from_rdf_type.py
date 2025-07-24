import json
from collections import defaultdict

# Load the GPT-generated triples
with open("triples_from_gpt.json", "r") as f:
    data = json.load(f)

# A mapping of entities to their rdf:type (object)
entity_type_map = defaultdict(set)

# Process each triple with rdf:type predicate
for item in data:
    for triple in item.get("Triples", []):
        predicate = triple.get("predicate")
        if predicate == "rdf:type":
            subject = triple.get("subject")
            obj = triple.get("object")

            if isinstance(subject, str) and isinstance(obj, str):
                entity_type_map[subject].add(obj)

# Normalize output
entity_type_output = []
for subject, types in entity_type_map.items():
    entity_type_output.append({
        "entity": subject,
        "types": sorted(types)
    })

# Save to file
with open("entity_type_from_rdf_type.json", "w") as f:
    json.dump(entity_type_output, f, indent=2)

print(f"Inferred {len(entity_type_output)} entities with rdf:type relationships.")
