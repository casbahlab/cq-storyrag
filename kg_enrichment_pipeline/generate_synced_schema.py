import rdflib
from rdflib.namespace import RDF, RDFS
import json

INPUT_FILE = "output/liveaid_instances.ttl"
OUTPUT_SCHEMA = "output/liveaid_schema_synced.ttl"
UNTYPED_LOG = "output/untyped_ex_entities.json"

g = rdflib.Graph()
g.parse(INPUT_FILE, format="turtle")

schema_graph = rdflib.Graph()

EX = rdflib.Namespace("http://wembrewind.live/ex#")

ex_classes = set()
ex_properties = set()
ex_entities = set()
untyped_entities = []

for s, p, o in g:
    # Track all ex: entities
    if isinstance(s, rdflib.URIRef) and str(s).startswith(str(EX)):
        ex_entities.add(s)
    if isinstance(o, rdflib.URIRef) and str(o).startswith(str(EX)):
        ex_entities.add(o)

    # Detect ex: properties
    if isinstance(p, rdflib.URIRef) and str(p).startswith(str(EX)):
        ex_properties.add(p)

    # Detect ex: classes from rdf:type
    if p == RDF.type and isinstance(o, rdflib.URIRef) and str(o).startswith(str(EX)):
        ex_classes.add(o)

# Identify untyped ex: entities
for entity in sorted(ex_entities):
    has_type = any(
        p == RDF.type for s, p, o in g.triples((entity, None, None))
    )
    if not has_type:
        untyped_entities.append(str(entity))

# Add classes to schema
for c in sorted(ex_classes):
    schema_graph.add((c, RDF.type, RDFS.Class))
    schema_graph.add((c, RDFS.label, rdflib.Literal(c.split("#")[-1])))

# Add properties to schema
for p in sorted(ex_properties):
    schema_graph.add((p, RDF.type, RDF.Property))
    schema_graph.add((p, RDFS.label, rdflib.Literal(p.split("#")[-1])))

# Save schema
schema_graph.serialize(OUTPUT_SCHEMA, format="turtle")
print(f"✅ Synced schema written to {OUTPUT_SCHEMA}")
print(f"Classes: {len(ex_classes)}, Properties: {len(ex_properties)}")

# Save untyped entities log
if untyped_entities:
    with open(UNTYPED_LOG, "w", encoding="utf-8") as log_file:
        json.dump(untyped_entities, log_file, indent=2)
    print(f"⚠ Untyped ex: entities: {len(untyped_entities)} (saved to {UNTYPED_LOG})")
else:
    print("✅ All ex: entities have types.")
