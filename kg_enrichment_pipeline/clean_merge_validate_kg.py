import rdflib
from rdflib import Graph, URIRef, Namespace
import os
import json

# === Input files ===
KG_FILE = "output/final_with_inferred.ttl"
SCHEMA_FILE = "data/schemaorg.ttl"
MUSICMETA_FILE = "data/musicmeta.owl"

OUTPUT_TTL = "output/final_cleaned_kg.ttl"
REMOVED_LOG = "output/removed_local_classes.json"

os.makedirs("output", exist_ok=True)

# === Namespaces ===
SCHEMA = Namespace("http://schema.org/")

g = Graph()
g.parse(KG_FILE, format="turtle")

schema_g = Graph()
schema_g.parse(SCHEMA_FILE, format="turtle")

musicmeta_g = Graph()
musicmeta_g.parse(MUSICMETA_FILE, format="xml")  # OWL usually XML/RDF

# Merge all into a master graph for validation
merged = g + schema_g + musicmeta_g

print(f"Parsed KG: {len(g)} triples")
print(f"Merged with ontologies: {len(merged)} triples")

removed_triples = []
to_remove = []

for s, p, o in g.triples((None, None, None)):
    # Match triples like: schema:Something a schema:Thing .
    if str(p) in [str(rdflib.RDF.type)] and str(o) == str(SCHEMA["Thing"]):
        # Check if this class exists in merged ontology
        if (s, None, None) not in schema_g:  # local definition
            to_remove.append((s, p, o))
            removed_triples.append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o)
            })

# Remove from KG
for triple in to_remove:
    g.remove(triple)

print(f"🗑 Removed {len(to_remove)} local class assertions.")

undefined_entities = set()
for s, p, o in g.triples((None, None, None)):
    # Check predicate
    if (p, None, None) not in merged and (None, None, p) not in merged:
        undefined_entities.add(str(p))

    # Check object if it is a URI
    if isinstance(o, rdflib.term.URIRef):
        if (o, None, None) not in merged and (None, None, o) not in merged:
            undefined_entities.add(str(o))

print(f"⚠ Undefined entities after cleaning: {len(undefined_entities)}")
if undefined_entities:
    print("Sample:", list(undefined_entities)[:10])

g.serialize(destination=OUTPUT_TTL, format="turtle")
with open(REMOVED_LOG, "w", encoding="utf-8") as f:
    json.dump(removed_triples, f, indent=2)

print(f"✅ Cleaned KG saved to {OUTPUT_TTL}")
print(f"📝 Removed triples logged to {REMOVED_LOG}")
