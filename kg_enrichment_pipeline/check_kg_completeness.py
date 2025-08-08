import rdflib
from rdflib import Graph, URIRef
import os

# Files
KG_FILE = "output/final_grouped_triples.ttl"
ONT_FILES = ["data/musicmeta.owl", "data/schemaorg.ttl"]

# Step 1: Load KG
kg = Graph()
kg.parse(KG_FILE, format="turtle")
print(f"KG Triples: {len(kg)}")

# Step 2: Load Ontologies
ont_graph = Graph()
for ont in ONT_FILES:
    fmt = "xml" if ont.endswith(".owl") else "turtle"
    ont_graph.parse(ont, format=fmt)
print(f"Ontology Triples: {len(ont_graph)}")

# Step 3: Collect subjects (KG + Ontologies)
kg_subjects = {str(s) for s in kg.subjects()}
ont_entities = {str(s) for s in ont_graph.subjects()}

# Step 4: Check completeness
undefined_objects = []
for o in kg.objects():
    o_str = str(o)
    # Ignore literals
    if not (o_str.startswith("http://") or o_str.startswith("https://") or o_str.startswith("ex:")):
        continue
    if o_str not in kg_subjects and o_str not in ont_entities:
        undefined_objects.append(o_str)

undefined_objects = sorted(set(undefined_objects))

print(f"⚠ Objects not defined anywhere: {len(undefined_objects)}")
for u in undefined_objects[:50]:  # Show first 50
    print(" -", u)

# Optional: Save to JSON
import json
with open("output/undefined_objects.json", "w") as f:
    json.dump(undefined_objects, f, indent=2)

print("Undefined objects saved to output/undefined_objects.json")
