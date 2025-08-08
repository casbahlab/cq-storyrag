import os
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS

# === Paths ===
schema_file = "schemaorg-full.ttl"
custom_ttls = ["schema_class_attributes_full.ttl", "untyped_suggestions_clean.ttl"]
output_file = "schema_subset_validated.ttl"

# === Namespaces ===
SCHEMA = Namespace("http://schema.org/")

# === Step 1: Load full schema.org graph ===
schema_graph = Graph()
schema_graph.parse(schema_file, format="turtle")
print(f"Loaded schema.org graph with {len(schema_graph)} triples")

# === Step 2: Load your custom TTLs ===
custom_graph = Graph()
for ttl_path in custom_ttls:
    custom_graph.parse(ttl_path, format="ttl")
print(f"Loaded custom TTLs with {len(custom_graph)} triples")

# === Step 3: Collect schema.org URIs used ===
used_terms = set()
for s, p, o in custom_graph:
    for term in (s, p, o):
        if isinstance(term, URIRef) and "schema.org" in str(term):
            uri = str(term).replace("https://", "http://").strip()
            if uri.startswith("http://schema.org/"):
                used_terms.add(URIRef(uri))

print(f"Found {len(used_terms)} unique schema.org terms in custom TTLs.")

# === Step 4: Create subset graph ===
subset_graph = Graph()
subset_graph.bind("schema", SCHEMA)
subset_graph.bind("rdf", RDF)
subset_graph.bind("rdfs", RDFS)

triples_added = 0
unmatched_terms = []

for term in sorted(used_terms):
    found = False

    # Match directly (term as subject)
    for triple in schema_graph.triples((term, None, None)):
        subset_graph.add(triple)
        triples_added += 1
        found = True

    # Match directly (term as object)
    for triple in schema_graph.triples((None, None, term)):
        subset_graph.add(triple)
        triples_added += 1
        found = True

    # Match by fragment (fallback)
    if not found:
        frag = str(term).split("/")[-1]
        for s, p, o in schema_graph.triples((None, None, None)):
            if (
                (isinstance(s, URIRef) and s.split("/")[-1] == frag)
                or (isinstance(o, URIRef) and o.split("/")[-1] == frag)
            ):
                subset_graph.add((s, p, o))
                found = True
                triples_added += 1

    if not found:
        unmatched_terms.append(term)

# === Step 5: Save subset ===
subset_graph.serialize(destination=output_file, format="turtle")
print(f"Subset saved to: {output_file} with {triples_added} triples.")

# === Step 6: Report unmatched ===
if unmatched_terms:
    print("\nThe following terms were NOT found in schema.org:")
    for term in unmatched_terms:
        print(f" - {term}")
