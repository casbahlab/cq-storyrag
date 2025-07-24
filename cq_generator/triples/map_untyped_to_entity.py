import json

# Input and output file paths
input_file = "untyped_entity_usage_trace_suggestions.json"
output_file = "entity_types_list.json"

# Load JSON data
with open(input_file, "r") as f:
    data = json.load(f)

# Extract entity-type pairs
entity_types = []

for entity_uri, records in data.items():
    types = set()
    for record in records:
        suggestion = record.get("Suggested", {})
        if "Type" in suggestion:
            types.add(suggestion["Type"])
    if types:
        entity_types.append({
            "entity": entity_uri,
            "types": list(types)
        })

# Save result
with open(output_file, "w") as f:
    json.dump(entity_types, f, indent=2)

print(f"Saved entity types to: {output_file}")
