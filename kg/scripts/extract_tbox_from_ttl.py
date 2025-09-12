from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
import argparse

# Namespaces
SCHEMA = Namespace("http://schema.org/")
EX = Namespace("http://wembrewind.live/ex#")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
SH = Namespace("http://www.w3.org/ns/shacl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

# TBox categories
TBOX_TYPES = {
    OWL.Class,
    RDF.Property,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.Ontology,
    SH.NodeShape,
    SH.PropertyShape
}

TBOX_PREDICATES = {
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    RDFS.domain,
    RDFS.range,
    OWL.equivalentClass,
    OWL.equivalentProperty,
    RDFS.label,
    RDFS.comment
}

def extract_tbox_from_ttl(input_file, schema_out, data_out=None, in_place=False):
    g = Graph()
    g.parse(input_file, format="turtle")

    schema_graph = Graph()
    schema_graph.bind("ex", EX)
    schema_graph.bind("schema", SCHEMA)
    schema_graph.bind("mm", MM)
    schema_graph.bind("rdfs", RDFS)
    schema_graph.bind("owl", OWL)
    schema_graph.bind("sh", SH)
    schema_graph.bind("xsd", XSD)

    data_graph = Graph()
    data_graph.bind("ex", EX)
    data_graph.bind("schema", SCHEMA)
    data_graph.bind("mm", MM)
    data_graph.bind("rdfs", RDFS)
    data_graph.bind("owl", OWL)
    data_graph.bind("sh", SH)
    data_graph.bind("xsd", XSD)

    for s, p, o in g:
        if (
            (p == RDF.type and o in TBOX_TYPES) or
            (p in TBOX_PREDICATES) or
            (str(s).startswith(str(EX)) and (p in TBOX_PREDICATES))
        ):
            schema_graph.add((s, p, o))
        else:
            data_graph.add((s, p, o))

    schema_graph.serialize(schema_out, format="turtle")
    if in_place:
        data_graph.serialize(input_file, format="turtle")
    elif data_out:
        data_graph.serialize(data_out, format="turtle")

    print(f"[extract] Schema triples: {len(schema_graph)}")
    print(f"[extract] Data triples: {len(data_graph)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract TBox (schema) triples from a TTL file")
    parser.add_argument("--in", dest="input_file", required=True)
    parser.add_argument("--schema-out", dest="schema_out", required=True)
    parser.add_argument("--data-out", dest="data_out")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    extract_tbox_from_ttl(args.input_file, args.schema_out, args.data_out, args.in_place)
