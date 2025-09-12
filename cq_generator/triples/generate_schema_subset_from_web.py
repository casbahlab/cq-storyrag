import os
import requests
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, RDFS

# === Paths ===
schema_url = "https://schema.org/version/latest/schemaorg-current-https.ttl"
schema_local_path = "schemaorg-full.ttl"
custom_ttls = ["schema_class_attributes_full.ttl", "untyped_suggestions_clean.ttl"]
output_file = "schema_subset.ttl"

if not os.path.exists(schema_local_path):
    print("Downloading schema.org TTL...")
    response = requests.get(schema_url)
    response.raise_for_status()
    with open(schema_local_path, "wb") as f:
        f.write(response.content)
    print("Downloaded schemaorg-current-https.ttl")

schema_graph = Graph()
schema_graph.parse(schema_local_path, format="turtle")
print(f"📚 Loaded schema.org graph with {len(schema_graph)} triples")

custom_graph = Graph()
for path in custom_ttls:
    custom_graph.parse(path, format="ttl")
print(f"📦 Loaded custom TTLs with {len(custom_graph)} triples")

schema_ns = Namespace("http://schema.org/")
used_terms = set()

for s, p, o in custom_graph:
    for term in (s, p, o):
        if isinstance(term, URIRef) and "schema.org" in str(term):
            uri = str(term).replace("https://", "http://").strip()
            if uri.startswith("http://schema.org/"):
                used_terms.add(URIRef(uri))

print(f"Found {len(used_terms)} unique schema.org terms in custom TTLs.")

subset_graph = Graph()
subset_graph.bind("schema", schema_ns)
subset_graph.bind("rdfs", RDFS)
subset_graph.bind("rdf", RDF)

triples_added = 0
unmatched_terms = []

for term in sorted(used_terms):
    found = False
    # Direct match
    for triple in schema_graph.triples((term, None, None)):
        subset_graph.add(triple)
        triples_added += 1
        found = True
    for triple in schema_graph.triples((None, None, term)):
        subset_graph.add(triple)
        triples_added += 1
        found = True

    # Fallback: match by fragment (e.g., MusicEvent)
    if not found:
        fragment = str(term).split("/")[-1]
        for s, p, o in schema_graph.triples((None, None, None)):
            if (
                    (isinstance(s, URIRef) and s.split("/")[-1] == fragment)
                    or (isinstance(o, URIRef) and o.split("/")[-1] == fragment)
            ):
                subset_graph.add((s, p, o))
                found = True
                triples_added += 1

    if not found:
        unmatched_terms.append(term)

subset_graph.serialize(destination=output_file, format="turtle")
print(f"Subset saved to: {output_file} with {triples_added} triples.")

# === Optional: Show unmatched terms ===
if unmatched_terms:
    print("\nThe following terms were NOT found in schema.org:")
    for term in unmatched_terms:
        print(f" - {term}")
