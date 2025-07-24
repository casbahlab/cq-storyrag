import json

# Input and output file paths
input_file = "untyped_entity_usage_trace_suggestions.json"
output_entity_types = "entity_types_list.json"
output_custom_schema = "custom_schema_declarations.ttl"

# Load JSON data
with open(input_file, "r") as f:
    data = json.load(f)

# Prepare output structures
entity_types = []
custom_classes = {}
custom_properties = {}

# Extract entity-type pairs and custom ontology info
for entity_uri, records in data.items():
    types = set()
    for record in records:
        suggestion = record.get("Suggested", {})
        custom_type = suggestion.get("Type")
        parent_type = suggestion.get("Parent_Type")
        prop = suggestion.get("Property")
        prop_type = suggestion.get("Property_Type")

        # Capture entity-level types
        if custom_type:
            types.add(custom_type)

        # Record custom class definitions
        if custom_type and custom_type.startswith(":") and parent_type:
            custom_classes[custom_type] = parent_type

        # Record custom property definitions
        if prop and prop.startswith(":") and prop_type:
            custom_properties[prop] = {
                "domain": custom_type,
                "range": prop_type
            }

    if types:
        entity_types.append({
            "entity": entity_uri,
            "types": list(types)
        })

# Save entity types list
with open(output_entity_types, "w") as f:
    json.dump(entity_types, f, indent=2)

print(f"Saved entity types to: {output_entity_types}")

# Create TTL content for custom schema
ttl_lines = [
    "@prefix : <http://example.org/> .",
    "@prefix schema: <http://schema.org/> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
]

# Add custom class declarations
for cls, parent in custom_classes.items():
    ttl_lines.append(f"{cls} a rdfs:Class ;")
    ttl_lines.append(f"  rdfs:subClassOf {parent} .\n")

# Add custom property declarations
for prop, info in custom_properties.items():
    domain = info["domain"]
    range_ = info["range"]
    ttl_lines.append(f"{prop} a rdf:Property ;")
    ttl_lines.append(f"  rdfs:domain {domain} ;")
    ttl_lines.append(f"  rdfs:range {range_} ;")
    ttl_lines.append(f"  rdfs:label \"{prop[1:]}\" .\n")

# Write TTL file
with open(output_custom_schema, "w") as f:
    f.write("\n".join(ttl_lines))

print(f"Saved custom schema declarations to: {output_custom_schema}")
