import json
from collections import defaultdict

# Input and output file paths
input_file = "raw_files/triples_from_gpt.json"
output_entity_types = "entity_type_from_rdf_type.json"
output_custom_schema = "custom_schema_from_rdf_type.ttl"

# Load the GPT-generated triples
with open(input_file, "r") as f:
    data = json.load(f)

# Mappings
entity_type_map = defaultdict(set)
custom_class_defs = {}

# Process each triple
for item in data:
    for triple in item.get("Triples", []):
        predicate = triple.get("predicate")
        subject = triple.get("subject")
        obj = triple.get("object")

        if predicate == "rdf:type" and isinstance(subject, str) and isinstance(obj, str):
            entity_type_map[subject].add(obj)

            # Track custom types for declaration
            if obj.startswith(":") and obj not in custom_class_defs:
                # Default to schema:Thing unless inferred otherwise
                custom_class_defs[obj] = "schema:Thing"

# Format entity-type output
entity_type_output = []
for subject, types in entity_type_map.items():
    entity_type_output.append({
        "entity": subject,
        "types": sorted(types)
    })

# Write entity-type list
with open(output_entity_types, "w") as f:
    json.dump(entity_type_output, f, indent=2)

print(f"Inferred {len(entity_type_output)} entities with rdf:type relationships.")
print(f"Detected {len(custom_class_defs)} custom classes.")

# TTL output
ttl_lines = [
    "@prefix : <http://example.org/> .",
    "@prefix schema: <http://schema.org/> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
]

for cls, parent in custom_class_defs.items():
    ttl_lines.append(f"{cls} a rdfs:Class ;")
    ttl_lines.append(f"  rdfs:subClassOf {parent} .\n")

with open(output_custom_schema, "w") as f:
    f.write("\n".join(ttl_lines))

print(f"Custom classes written to: {output_custom_schema}")
