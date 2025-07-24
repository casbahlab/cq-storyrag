import json
from rdflib import Graph, Namespace, RDF, RDFS, URIRef
from urllib.parse import urlparse

# File paths
input_file = "raw_files/untyped_entity_usage_trace_suggestions.json"
output_file = "untyped_suggestions_clean.ttl"

# Load JSON
with open(input_file) as f:
    data = json.load(f)

# Namespaces
EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

# Initialize graph
g = Graph()
g.bind("schema", SCHEMA)
g.bind("xsd", XSD)
g.bind("", EX)

# Helper to check if URI is valid
def is_valid_uri(uri):
    if not isinstance(uri, str): return False
    parsed = urlparse(uri)
    return bool(parsed.scheme and parsed.netloc)

# Internal registries
declared_classes = set()
declared_properties = set()

def safe_ref(s):
    return URIRef(s) if is_valid_uri(s) else None

def local_ref(term):
    if term.startswith("schema:"):
        return SCHEMA[term.replace("schema:", "")]
    elif term.startswith("http://") or term.startswith("https://"):
        return URIRef(term)
    elif term.startswith(":"):
        return EX[term[1:]]
    else:
        return EX[term]

def resolve_term(term):
    if term.startswith("schema:"):
        return SCHEMA[term[7:]]
    elif term.startswith("rdfs:"):
        return RDFS[term[5:]]
    elif term.startswith("rdf:"):
        return RDF[term[4:]]
    elif term.startswith(":"):
        return EX[term[1:]]
    elif term.startswith("http"):
        return URIRef(term)
    else:
        return EX[term]


# Main logic
for entity_uri, entries in data.items():
    subj = safe_ref(entity_uri)
    if not subj:
        continue

    for record in entries:
        suggestion = record.get("Suggested", {})
        type_ = suggestion.get("Type")
        parent_type = suggestion.get("Parent_Type")
        prop = suggestion.get("Property")
        prop_type = suggestion.get("Property_Type")

        # Class Declaration
        if type_:
            class_uri = local_ref(type_)
            if class_uri:
                if type_ not in declared_classes:
                    g.add((class_uri, RDF.type, RDFS.Class))
                    if parent_type:
                        parent_uri = (
                            SCHEMA[parent_type.replace("schema:", "")]
                            if parent_type.startswith("schema:")
                            else local_ref(parent_type)
                        )
                        if parent_uri:
                            g.add((class_uri, RDFS.subClassOf, resolve_term(parent_type)))
                    declared_classes.add(type_)
                g.add((subj, RDF.type, class_uri))

        # Property Declaration
        if prop and prop_type:
            prop_uri = local_ref(prop)
            range_uri = (
                SCHEMA[prop_type.replace("schema:", "")] if prop_type.startswith("schema:") else
                XSD[prop_type.replace("xsd:", "")] if prop_type.startswith("xsd:") else
                local_ref(prop_type)
            )
            if prop_uri and range_uri and prop not in declared_properties:
                g.add((prop_uri, RDF.type, RDF.Property))
                if type_:
                    g.add((prop_uri, RDFS.domain, local_ref(type_)))
                g.add((prop_uri, RDFS.range, range_uri))
                declared_properties.add(prop)

# Save output
g.serialize(destination=output_file, format="turtle")
print(f"Clean TTL written to: {output_file}")
