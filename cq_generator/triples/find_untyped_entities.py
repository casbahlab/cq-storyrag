import json

# Load the full triple data
with open("triples_from_gpt.json") as f:
    triples_data = json.load(f)

# Load rdf:type inferred entity types
with open("entity_type_from_rdf_type.json") as f:
    typed_entities = json.load(f)

# Collect all typed entity URIs
typed_uris = {entry["entity"] for entry in typed_entities}

# Collect all subjects and objects from the triples
all_entities = set()
for entry in triples_data:
    for triple in entry.get("Triples", []):
        subj = triple.get("subject")
        obj = triple.get("object")
        if isinstance(subj, str) and subj.startswith("http"):
            all_entities.add(subj)
        if isinstance(obj, str) and obj.startswith("http"):
            all_entities.add(obj)

# Find untyped entities
untyped_entities = sorted(all_entities - typed_uris)

# Save for later
with open("untyped_entities.json", "w") as f:
    json.dump(untyped_entities, f, indent=2)

# Print summary
print(f"Total unique entities found: {len(all_entities)}")
print(f"Typed entities via rdf:type: {len(typed_uris)}")
print(f"Entities missing rdf:type: {len(untyped_entities)}")
print("Saved to: untyped_entities.json")
