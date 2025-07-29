from rdflib import Graph, RDF, RDFS, OWL, URIRef
from collections import defaultdict
import pickle

# Load both ontology graphs
musicmeta_graph = Graph()
musicmeta_graph.parse("musicmeta.owl")

schema_graph = Graph()
schema_graph.parse("schemaorg.ttl", format="ttl")

# Prefix mapping
PREFIX_MAP = {
    "https://w3id.org/polifonia/ontology/music-meta/": "mm:",
    "https://w3id.org/polifonia/ontology/core/": "mmcore:",
    "http://schema.org/": "schema:"
}

def apply_prefix(uri: str) -> str:
    for base, prefix in PREFIX_MAP.items():
        if uri.startswith(base):
            return uri.replace(base, prefix)
    return uri


def get_prefixed_ontology_maps(ontology_graph: Graph):
    subject_predicate_map = defaultdict(set)
    predicate_object_map = {}
    subclass_map = defaultdict(set)
    class_restrictions = defaultdict(set)

    # Step 1: Direct domain/range property mapping
    for s, p, o in ontology_graph.triples((None, RDF.type, RDF.Property)):
        for domain in ontology_graph.objects(subject=s, predicate=RDFS.domain):
            subject_predicate_map[apply_prefix(str(domain))].add(str(s))  # key prefixed, value full URI
        for range_ in ontology_graph.objects(subject=s, predicate=RDFS.range):
            pred_key = apply_prefix(str(s))  # prefixed key
            if pred_key not in predicate_object_map:
                predicate_object_map[pred_key] = set()
            predicate_object_map[pred_key].add(str(range_))  # value full URI

    # Step 2: Subclass mapping
    for subclass, _, superclass in ontology_graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(superclass, URIRef):
            subclass_map[apply_prefix(str(subclass))].add(apply_prefix(str(superclass)))

    # Step 3: Restriction-based property capture
    for class_uri, _, restriction in ontology_graph.triples((None, RDFS.subClassOf, None)):
        if (restriction, RDF.type, OWL.Restriction) in ontology_graph:
            for prop in ontology_graph.objects(restriction, OWL.onProperty):
                class_restrictions[apply_prefix(str(class_uri))].add(str(prop))  # full URI as value

    # Step 4: Inherited properties recursively
    def get_all_properties(cls, memo={}):
        if cls in memo:
            return memo[cls]
        props = set(subject_predicate_map.get(cls, set())) | class_restrictions.get(cls, set())
        for parent in subclass_map.get(cls, []):
            props |= get_all_properties(parent, memo)
        memo[cls] = props
        return props

    # Step 5: Final expanded property map
    full_class_properties = {}
    all_classes = set(subject_predicate_map.keys()) | set(class_restrictions.keys()) | set(subclass_map.keys())
    for cls in all_classes:
        full_class_properties[cls] = sorted(get_all_properties(cls))

    return full_class_properties, predicate_object_map


# Build for both ontologies
mm_class_props, mm_pred_map = get_prefixed_ontology_maps(musicmeta_graph)
schema_class_props, schema_pred_map = get_prefixed_ontology_maps(schema_graph)

# Combine maps with prefixed keys but full URI values
combined = {
    "subject_predicate_map": {**mm_class_props, **schema_class_props},
    "predicate_object_map": {**mm_pred_map, **schema_pred_map}
}

# Save to file
with open("ontology_maps_prefixed_keys_only.pkl", "wb") as f:
    pickle.dump(combined, f)

print("✅ Saved ontology maps with prefixed keys and full URI values.")
