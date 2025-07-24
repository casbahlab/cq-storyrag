from rdflib import Graph, RDF, RDFS, URIRef
from collections import defaultdict

ttl_path = "cleaned_file_with_schema.ttl"  # your schema file
g = Graph()
g.parse(ttl_path, format="turtle")

# Collect all declared classes and properties
declared_classes = set(g.subjects(RDF.type, RDFS.Class))
declared_properties = set(g.subjects(RDF.type, RDF.Property))

# Collect all used classes and properties
used_classes = set()
used_properties = set()
undefined_terms = defaultdict(list)

for s, p, o in g:
    if p == RDF.type and isinstance(o, URIRef):
        used_classes.add(o)
        if o not in declared_classes and not str(o).startswith("http://www.w3.org"):
            undefined_terms["Class"].append(o)

    if isinstance(p, URIRef):
        used_properties.add(p)
        if p not in declared_properties and not str(p).startswith("http://www.w3.org"):
            undefined_terms["Property"].append(p)

# Report
print("Declared Classes:", len(declared_classes))
print("Declared Properties:", len(declared_properties))

if undefined_terms:
    print("\nUndefined Terms Found:")
    for kind, uris in undefined_terms.items():
        print(f"\n{kind}:")
        for uri in set(uris):
            print(f" - {uri}")
else:
    print("No undefined classes or properties!")
