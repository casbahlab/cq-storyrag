#!/usr/bin/env python3
import rdflib
import argparse

def list_classes_and_properties(ttl_file):
    g = rdflib.Graph()
    g.parse(ttl_file, format="turtle")

    # Find classes
    classes = set()
    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.Class):
        classes.add(s)
    for s in g.subjects(rdflib.RDF.type, rdflib.RDFS.Class):
        classes.add(s)
    # Also from usage as rdf:type
    for o in g.objects(None, rdflib.RDF.type):
        if isinstance(o, rdflib.URIRef):
            classes.add(o)

    # Find properties
    properties = set()
    for s in g.subjects(rdflib.RDF.type, rdflib.RDF.Property):
        properties.add(s)
    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.ObjectProperty):
        properties.add(s)
    for s in g.subjects(rdflib.RDF.type, rdflib.OWL.DatatypeProperty):
        properties.add(s)
    # Also from usage in triples
    for p in g.predicates(None, None):
        if isinstance(p, rdflib.URIRef):
            properties.add(p)

    # Sort & print
    print("\n=== CLASSES ===")
    for c in sorted(classes):
        print(c)

    print("\n=== PROPERTIES ===")
    for p in sorted(properties):
        print(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List all classes and properties in a TTL file")
    parser.add_argument("--ttl", required=True, help="Path to TTL file")
    args = parser.parse_args()

    list_classes_and_properties(args.ttl)
