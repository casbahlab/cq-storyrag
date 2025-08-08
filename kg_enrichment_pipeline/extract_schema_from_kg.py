import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef, Literal
from collections import defaultdict
import os

INPUT_KG = "output/final_cleaned_kg.ttl"
OUTPUT_SCHEMA = "output/extracted_schema.ttl"

os.makedirs("output", exist_ok=True)

g = Graph()
g.parse(INPUT_KG, format="turtle")

# Namespaces
EX = Namespace("http://wembrewind.live/ex#")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
SCHEMA = Namespace("http://schema.org/")

schema_graph = Graph()
schema_graph.bind("ex", EX)
schema_graph.bind("mm", MM)
schema_graph.bind("schema", SCHEMA)
schema_graph.bind("rdfs", RDFS)
schema_graph.bind("owl", OWL)
schema_graph.bind("xsd", XSD)

# Collect classes and properties
classes = set()
object_properties = defaultdict(lambda: {"domain": set(), "range": set()})
datatype_properties = defaultdict(lambda: {"domain": set(), "range": set()})

for s, p, o in g:
    # Capture classes
    if p == RDF.type:
        classes.add(o)

    # Determine if property is object or data property
    if isinstance(o, Literal):
        datatype_properties[p]["domain"].add(s)
        # Determine literal type
        if o.datatype:
            datatype_properties[p]["range"].add(o.datatype)
        else:
            datatype_properties[p]["range"].add(XSD.string)
    else:
        object_properties[p]["domain"].add(s)
        object_properties[p]["range"].add(o)

# Declare classes
for c in classes:
    schema_graph.add((c, RDF.type, OWL.Class))

# Declare properties
for prop, info in object_properties.items():
    schema_graph.add((prop, RDF.type, OWL.ObjectProperty))
    for domain in info["domain"]:
        schema_graph.add((prop, RDFS.domain, domain))
    for rng in info["range"]:
        schema_graph.add((prop, RDFS.range, rng))

for prop, info in datatype_properties.items():
    schema_graph.add((prop, RDF.type, OWL.DatatypeProperty))
    for domain in info["domain"]:
        schema_graph.add((prop, RDFS.domain, domain))
    for rng in info["range"]:
        schema_graph.add((prop, RDFS.range, rng))

# Save schema
schema_graph.serialize(destination=OUTPUT_SCHEMA, format="turtle")
print(f"✅ Schema extracted and saved to {OUTPUT_SCHEMA}")
print(f"Total Classes: {len(classes)}")
print(f"Object Properties: {len(object_properties)}")
print(f"Datatype Properties: {len(datatype_properties)}")
