# ontology_loader.py
from __future__ import annotations

from rdflib import Graph, RDF, RDFS, OWL, URIRef
from typing import Dict, Set
import os

MUSIC_META_URL = "https://raw.githubusercontent.com/polifonia-project/music-meta-ontology/main/ontology/musicmeta.owl"
SCHEMA_ORG_TTL = "https://schema.org/version/latest/schemaorg-current-https.ttl"


def merge_ontologies(*ontologies):
    combined = {
        "classes": {},
        "properties": {}
    }

    for onto in ontologies:
        classes = onto.get("classes", {})
        properties = onto.get("properties", {})

        # Convert sets to dicts with dummy values if needed
        if isinstance(classes, set):
            classes = {cls: {} for cls in classes}
        if isinstance(properties, set):
            properties = {prop: {} for prop in properties}

        for k, v in classes.items():
            if k not in combined["classes"]:
                combined["classes"][k] = v
            else:
                combined["classes"][k].update(v)

        for k, v in properties.items():
            if k not in combined["properties"]:
                combined["properties"][k] = v
            else:
                combined["properties"][k].update(v)

    return combined


def load_graph(source_url: str, local_path: str) -> Graph:
    if not os.path.exists(local_path):
        import requests
        print(f"Downloading {source_url}")
        with open(local_path, 'wb') as f:
            f.write(requests.get(source_url).content)
    g = Graph()
    fmt = 'xml' if local_path.endswith(".owl") else 'turtle'
    g.parse(local_path, format=fmt)
    return g

def prefixed_label(uri: str) -> str:
    if "polifonia/ontology/music-meta" in uri:
        return "mm:" + uri.split("/")[-1]
    elif "schema.org" in uri:
        return "schema:" + uri.split("/")[-1]
    elif "polifonia/ontology/core" in uri:
        return "mmcore:" + uri.split("/")[-1]
    else:
        print(f"odd schema alert!!! : {uri}" )
        return uri.split("/")[-1]


def extract_classes(graph: Graph) -> Dict[str, str]:
    class_map = {}
    for class_type in [OWL.Class, RDFS.Class]:
        for s in graph.subjects(RDF.type, class_type):
            uri = str(s)
            if "/" in uri:
                class_map[uri] = prefixed_label(uri)
    return class_map



def extract_properties(graph: Graph) -> Dict[str, Dict[str, Set[str] | str]]:
    props = {}


    for prop in graph.subjects(RDF.type, RDF.Property):
        uri = str(prop)
        if uri.startswith("_:") or uri.startswith("N") or "/" not in uri:
            continue  # Skip anonymous or malformed props

        label_str = prefixed_label(uri)

        props[uri] = {
            "label": label_str,
            "domain": set(),
            "range": set()
        }

    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        uri = str(prop)
        if uri.startswith("_:") or uri.startswith("N") or "/" not in uri:
            continue  # Skip anonymous or malformed props

        label_str = prefixed_label(uri)



        props[uri] = {
            "label": label_str,
            "domain": set(),
            "range": set()
        }

    for uri in props:
        prop_uri = URIRef(uri)

        for d in graph.objects(prop_uri, RDFS.domain):
            props[uri]["domain"].add(prefixed_label(d))
        for r in graph.objects(prop_uri, RDFS.range):
            props[uri]["range"].add(prefixed_label(r))

    return props



def load_music_meta_ontology() -> dict:
    music_meta_path = "musicmeta.owl"  # or full path if needed
    g = Graph()
    g.parse(music_meta_path)

    classes = {}
    properties = {}

    for s in g.subjects(RDF.type, OWL.Class):
        label = g.value(s, RDFS.label)
        classes[str(s)] = str(label) if label else s.split("/")[-1]

    for s in g.subjects(RDF.type, OWL.ObjectProperty):
        label = g.value(s, RDFS.label)
        properties[str(s)] = str(label) if label else s.split("/")[-1]

    return {
        "graph": g,
        "classes": extract_classes(g),
        "properties": extract_properties(g)
    }


def load_schema_org_ontology(local_file: str = "schemaorg.ttl") -> Dict:
    g = load_graph(SCHEMA_ORG_TTL, local_file)
    return {
        "graph": g,
        "classes": extract_classes(g),
        "properties": extract_properties(g)
    }


def combined_graph():
    mm = load_music_meta_ontology()
    print(f"Music Meta: {len(mm['classes'])} classes, {len(mm['properties'])} props")

    schema = load_schema_org_ontology()
    print(f"Schema.org: {len(schema['classes'])} classes, {len(schema['properties'])} props")

    # Merge
    combined_ontology = merge_ontologies(mm, schema)
    classes = combined_ontology["classes"]
    properties = combined_ontology["properties"]

    # Readable dropdowns
    class_labels = sorted({
        label for label in classes.values()
        if isinstance(label, str)
    })

    property_labels = sorted({
        label
        for value in properties.values()
        for label in (
            [value] if isinstance(value, str)
            else value.get("label", []) if isinstance(value.get("label", []), list)
            else [value.get("label")] if isinstance(value.get("label"), str)
            else []
        )
    })

    # 🔁 Build Subject → Predicate Map
    subject_predicate_map = {}
    for prop_uri, prop_info in properties.items():
        for domain in prop_info.get("domain", []):
            subject_predicate_map.setdefault(domain, set()).add(prop_uri)

    # 🔁 Build Predicate → Object Type Map
    predicate_object_map = {}
    for prop_uri, prop_info in properties.items():
        predicate_object_map[prop_uri] = prop_info.get("range", set())

    # 🔁 Build Label → URI maps for dropdown usage
    label_to_class = {
        label: label for uri, label in classes.items()
        if isinstance(label, str)
    }

    label_to_property = {}
    for uri, prop in properties.items():
        if isinstance(prop, str):
            label_to_property[prop] = uri
        elif isinstance(prop, dict):
            if isinstance(prop.get("label"), list):
                for lbl in prop["label"]:
                    label_to_property[lbl] = uri
            elif isinstance(prop.get("label"), str):
                label_to_property[prop["label"]] = uri

    return {
        "classes": classes,
        "properties": properties,
        "class_labels": class_labels,
        "property_labels": property_labels,
        "subject_predicate_map": subject_predicate_map,
        "predicate_object_map": predicate_object_map,
        "label_to_class": label_to_class,
        "label_to_property": label_to_property,
    }



g = combined_graph()
#print(f'graph classes : {g["classes"]}')
# 'https://w3id.org/polifonia/ontology/music-meta/AbstractScore': 'mm:AbstractScore'

#print(f'graph subject_predicate_map : {g["subject_predicate_map"]}')
# 'mm:CreativeProcess': {'https://w3id.org/polifonia/ontology/music-meta/involvesCreativeAction', 'https://w3id.org/polifonia/ontology/music-meta/creates', 'https://w3id.org/polifonia/ontology/music-meta/hasAgentRole'}

#print(f'graph properties : {g["properties"]}')
# 'https://w3id.org/polifonia/ontology/core/hasSource': {'label': 'mmcore:hasSource', 'domain': set(), 'range': set()}

#print(f'graph classes : {g["predicate_object_map"]}')

# 'https://w3id.org/polifonia/ontology/music-meta/hasCollaboratedWith': {'mm:MusicArtist'}
# 'https://w3id.org/polifonia/ontology/core/hasSource': set()

#print(f'graph label_to_class : {g["label_to_class"]}')
# 'mm:AbstractScore': 'https://w3id.org/polifonia/ontology/music-meta/AbstractScore'

#print(f'graph label_to_property : {g["label_to_property"]}')
#'mmcore:hasSource': 'https://w3id.org/polifonia/ontology/core/hasSource'