from rdflib import Graph, URIRef, RDF, RDFS, Namespace

# === Paths ===
INPUT_TTL = "cleaned_file_with_schema.ttl"
OUTPUT_TTL = "minimal_schema_template.ttl"

BASE = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")

# === Load full graph ===
g = Graph()
g.parse(INPUT_TTL, format="ttl")

# === Output graph ===
minimal = Graph()
minimal.bind("ex", BASE)
minimal.bind("rdfs", RDFS)
minimal.bind("rdf", RDF)
minimal.bind("schema", SCHEMA)

custom_classes = set()
for s, p, o in g.triples((None, RDF.type, RDFS.Class)):
    if isinstance(s, URIRef) and str(s).startswith("http://example.org/"):
        custom_classes.add(s)

for cls in custom_classes:
    for triple in g.triples((cls, None, None)):
        minimal.add(triple)
    for triple in g.triples((None, None, cls)):
        minimal.add(triple)

for prop in g.subjects(RDFS.domain, None):
    domain = g.value(prop, RDFS.domain)
    range_ = g.value(prop, RDFS.range)
    if domain in custom_classes or range_ in custom_classes:
        for triple in g.triples((prop, None, None)):
            minimal.add(triple)

referenced_schema = set()
for cls in custom_classes:
    for _, _, parent in g.triples((cls, RDFS.subClassOf, None)):
        if isinstance(parent, URIRef) and "schema.org" in str(parent):
            referenced_schema.add(parent)

for prop in minimal.subjects(RDFS.domain, None):
    for obj in g.objects(prop, RDFS.range):
        if isinstance(obj, URIRef) and "schema.org" in str(obj):
            referenced_schema.add(obj)

for schema_cls in referenced_schema:
    for triple in g.triples((schema_cls, None, None)):
        minimal.add(triple)

# === Save result ===
minimal.serialize(destination=OUTPUT_TTL, format="turtle")
print(f"✅ Saved minimal schema-only template to: {OUTPUT_TTL}")
