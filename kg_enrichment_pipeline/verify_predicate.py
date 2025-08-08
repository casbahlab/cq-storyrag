from rdflib import Graph

# Load KG
kg = Graph()
kg.parse("output/final_with_inferred.ttl", format="turtle")

# Load ontologies
ont = Graph()
ont.parse("data/musicmeta.owl")
ont.parse("data/schemaorg.ttl")

# Collect all URIs in ontologies
ontology_uris = set(str(x) for x in ont.all_nodes() if x.startswith("http"))
print(f"Total ontology URIs: {len(ontology_uris)}")

# Namespaces
SCHEMA = "http://schema.org/"
MM = "https://w3id.org/polifonia/ontology/music-meta/"
EX = "http://wembrewind.live/ex#"

undefined_predicates = set()
for _, p, _ in kg:
    ps = str(p)
    if ps.startswith(EX):
        continue
    if ps in ontology_uris or ps.startswith(SCHEMA) or ps.startswith(MM):
        continue
    undefined_predicates.add(ps)

print(f"Total unique predicates: {len(set(kg.predicates()))}")
print(f"⚠ Truly undefined predicates: {len(undefined_predicates)}")
for up in sorted(undefined_predicates):
    print(" -", up)
