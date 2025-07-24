from rdflib import Graph, Namespace
import os

# === Input TTL files ===
input_files = [
    "schema_subset_validated.ttl",
    "schema_class_attributes_full.ttl",
    "untyped_suggestions_clean.ttl"
]

# === Output path ===
output_file = "combined_schema_graph.ttl"

# === Namespaces ===
SCHEMA = Namespace("http://schema.org/")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

# === Load and merge all graphs ===
combined_graph = Graph()
combined_graph.bind("schema", SCHEMA)
combined_graph.bind("rdfs", RDFS)
combined_graph.bind("rdf", RDF)

for ttl_file in input_files:
    if not os.path.exists(ttl_file):
        print(f"⚠️ File not found: {ttl_file}")
        continue
    print(f"Loading: {ttl_file}")
    combined_graph.parse(ttl_file, format="ttl")

print(f"\nTotal triples in combined graph: {len(combined_graph)}")

# === Save combined TTL ===
combined_graph.serialize(destination=output_file, format="turtle")
print(f"Combined TTL saved to: {output_file}")
