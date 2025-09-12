import rdflib
from rdflib.namespace import RDF, RDFS
from collections import defaultdict
import os
import json

# =============================
# CONFIG
# =============================
INPUT_FILE = "output/final_cleaned_kg.ttl"
OUTPUT_SCHEMA = "output/liveaid_schema.ttl"
OUTPUT_INSTANCES = "output/liveaid_instances.ttl"
REPORT_FILE = "output/split_report.json"

os.makedirs("output", exist_ok=True)

# =============================
# LOAD GRAPH
# =============================
print(f"🔹 Loading KG: {INPUT_FILE}")
g = rdflib.Graph()
g.parse(INPUT_FILE, format="turtle")
print(f"✅ Loaded {len(g)} triples.")

# =============================
# SEPARATE SCHEMA VS INSTANCES
# =============================
schema_graph = rdflib.Graph()
instance_graph = rdflib.Graph()

# Prefixes (retain existing)
for prefix, ns in g.namespaces():
    schema_graph.bind(prefix, ns)
    instance_graph.bind(prefix, ns)

schema_triples = set()
instance_triples = set()

# Treat anything that is:
# - a Class or Property definition
# - or involves rdf:type Class/Property
# as schema
schema_classes = set()
schema_predicates = set()

for s, p, o in g:
    # Mark rdf:type Class/Property as schema
    if p == RDF.type and (
        o in [RDFS.Class, RDF.Property] or
        isinstance(o, rdflib.term.URIRef) and "ontology" in str(o).lower()
    ):
        schema_triples.add((s, p, o))
        schema_classes.add(s)
        continue

    # Capture explicit schema definitions
    if p in [RDFS.domain, RDFS.range, RDFS.subClassOf]:
        schema_triples.add((s, p, o))
        schema_classes.add(s)
        continue

    # Otherwise, assume instance triple
    instance_triples.add((s, p, o))
    schema_predicates.add(p)

for t in schema_triples:
    schema_graph.add(t)
for t in instance_triples:
    instance_graph.add(t)

# =============================
# SAVE OUTPUT
# =============================
schema_graph.serialize(destination=OUTPUT_SCHEMA, format="turtle")
instance_graph.serialize(destination=OUTPUT_INSTANCES, format="turtle")

print(f"✅ Schema TTL: {OUTPUT_SCHEMA} ({len(schema_graph)} triples)")
print(f"✅ Instances TTL: {OUTPUT_INSTANCES} ({len(instance_graph)} triples)")

# =============================
# REPORT
# =============================
report = {
    "total_triples": len(g),
    "schema_triples": len(schema_graph),
    "instance_triples": len(instance_graph),
    "sum_check": len(schema_graph) + len(instance_graph),
    "schema_classes_detected": len(schema_classes),
    "unique_predicates_in_instances": len(schema_predicates),
    "missing_types": []
}

# Identify any instance subjects without rdf:type
instance_subjects = set(s for s, _, _ in instance_triples)
typed_subjects = set(s for s, p, _ in instance_triples if p == RDF.type)
report["missing_types"] = list(str(s) for s in instance_subjects - typed_subjects)

with open(REPORT_FILE, "w") as f:
    json.dump(report, f, indent=2)

print(f"📄 Split report written to {REPORT_FILE}")
