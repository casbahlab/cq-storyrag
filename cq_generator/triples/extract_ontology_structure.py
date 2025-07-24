import json
from rdflib import Graph, RDFS

# Input and output paths
input_ttl_file = "ontology_template_combined_final.ttl"
class_output = "class_hierarchy.json"
property_output = "property_definitions.json"

# Load ontology
g = Graph()
g.parse(input_ttl_file, format="turtle")

# Extract subclass relationships
subclass_data = []
for s, p, o in g.triples((None, RDFS.subClassOf, None)):
    subclass_data.append({
        "Class": str(s),
        "SubclassOf": str(o)
    })

# Extract property domain and range
property_data = []
for s in g.subjects(RDFS.domain, None):
    domain = g.value(s, RDFS.domain)
    range_ = g.value(s, RDFS.range)
    property_data.append({
        "Property": str(s),
        "Domain": str(domain),
        "Range": str(range_)
    })

# Save as JSON
with open(class_output, "w") as f:
    json.dump(subclass_data, f, indent=2)

with open(property_output, "w") as f:
    json.dump(property_data, f, indent=2)

print(f"Extracted {len(subclass_data)} classes → saved to {class_output}")
print(f"Extracted {len(property_data)} properties → saved to {property_output}")
