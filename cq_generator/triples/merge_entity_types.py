import json
from collections import defaultdict

# Load input files
with open("raw_files/entity_types_list.json") as f1, open("raw_files/entity_type_from_rdf_type.json") as f2:
    enriched = json.load(f1)
    rdf_types = json.load(f2)

# Create mapping: entity → set of types
entity_map = defaultdict(set)

# Add enriched types
for item in enriched:
    entity_map[item["entity"]].update(item["types"])

# Add rdf:type-derived types
for item in rdf_types:
    entity_map[item["entity"]].update(item["types"])

# Clean up entities marked as "Drop this"
cleaned_output = [
    {
        "entity": entity,
        "types": sorted(list(types))
    }
    for entity, types in entity_map.items()
    if not any("Drop this" in t for t in types)
]

# Save output
with open("raw_files/merged_entity_types.json", "w") as out:
    json.dump(cleaned_output, out, indent=2)

print("Merged entity types saved to: merged_entity_types.json")
