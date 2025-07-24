import rdflib

# Input file paths
custom_schema_file = "custom_schema_declarations.ttl"
schema_attributes_file = "schema_class_attributes.ttl"
output_file = "ontology_template_combined_final.ttl"

# Standard prefixes to inject
required_prefixes = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <http://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <http://example.org/> .
"""

# Step 1: Load contents
with open(custom_schema_file, "r") as f1, open(schema_attributes_file, "r") as f2:
    ttl_custom = f1.read()
    ttl_schema = f2.read()

# Step 2: Strip existing prefix declarations
def strip_prefixes(ttl_text):
    lines = ttl_text.splitlines()
    return "\n".join(line for line in lines if not line.strip().startswith("@prefix"))

ttl_clean_custom = strip_prefixes(ttl_custom)
ttl_clean_schema = strip_prefixes(ttl_schema)

# Step 3: Merge TTLs with clean prefix block
merged_ttl = required_prefixes + "\n\n" + ttl_clean_custom.strip() + "\n\n" + ttl_clean_schema.strip()

# Step 4: Save the cleaned and merged TTL
with open(output_file, "w") as f:
    f.write(merged_ttl)

print(f"Merged TTL written to {output_file}")

# Step 5: Validate using rdflib
try:

    g = rdflib.Graph()
    g.parse(output_file, format="turtle")
    print(f"Validation successful: {len(g)} triples loaded.")
except Exception as e:
    print("RDF Validation Error:")
    print(e)
