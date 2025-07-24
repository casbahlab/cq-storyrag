from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL
import sys

# Load graphs
def load_graph(path, label):
    g = Graph()
    try:
        g.parse(path, format="turtle")
        print(f"✅ Loaded {label} from: {path} with {len(g)} triples")
        return g
    except Exception as e:
        print(f"❌ Failed to parse {label}: {e}")
        sys.exit(1)

# Check if URI is declared in schema
def get_declared_terms(schema_graph):
    declared = set()

    for s in schema_graph.subjects(RDF.type, RDFS.Class):
        declared.add(s)
    for s in schema_graph.subjects(RDF.type, RDF.Property):
        declared.add(s)
    return declared

# Extract used terms from instance graph
def get_used_terms(instance_graph):
    used = set()

    for s, p, o in instance_graph:
        used.add(p)
        if isinstance(o, URIRef):
            used.add(o)
    return used

# Compare usage and declarations
def validate_schema(instance_graph, schema_graph):
    declared = get_declared_terms(schema_graph)
    used = get_used_terms(instance_graph)

    undeclared = used - declared

    # Exclude core RDF/OWL/RDFS/Schema.org namespaces
    ignore_ns = ("http://www.w3.org", "https://schema.org")

    issues = [u for u in undeclared if not any(str(u).startswith(ns) for ns in ignore_ns)]

    if issues:
        print("\n⚠️ Undeclared terms used in instance TTL:")
        for term in sorted(issues):
            print(f" - {term}")
    else:
        print("\n✅ All terms used in instance TTL are declared in schema TTL or external vocabularies.")

# ----------- MAIN -----------
if __name__ == "__main__":
    instance_path = "minimal_instance.ttl"
    schema_path = "minimal_schema_template_manual_updated.ttl"

    instance_graph = load_graph(instance_path, "Instance Graph")
    schema_graph = load_graph(schema_path, "Schema Graph")

    validate_schema(instance_graph, schema_graph)
