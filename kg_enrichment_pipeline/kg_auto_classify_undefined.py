import rdflib
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal
import json
import os

# ==============================
# CONFIG
# ==============================
KG_FILE = "output/final_grouped_triples.ttl"
MUSICMETA_FILE = "data/musicmeta.owl"
SCHEMAORG_FILE = "data/schemaorg.ttl"
OUTPUT_TTL = "output/final_with_inferred.ttl"
INFER_LOG = "output/inferred_entities.json"

os.makedirs("output", exist_ok=True)

EX = Namespace("http://wembrewind.live/ex#")
SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

# ==============================
# LOAD KG + Ontologies
# ==============================
g = Graph()
g.parse(KG_FILE, format="turtle")

ont = Graph()
ont.parse(MUSICMETA_FILE, format="xml")
ont.parse(SCHEMAORG_FILE, format="turtle")

# ==============================
# Find undefined entities
# ==============================
subjects = {str(s) for s in g.subjects()}
objects = {str(o) for o in g.objects() if isinstance(o, URIRef)}

ontology_classes = {str(s) for s in ont.subjects(RDF.type, RDFS.Class)}
ontology_instances = {str(s) for s in ont.subjects()}

undefined_entities = [
    o for o in objects
    if o not in subjects
    and o not in ontology_classes
    and o not in ontology_instances
]

# ==============================
# Heuristic classification rules
# ==============================
def infer_type_for_entity(entity: URIRef):
    """Infer a class for an undefined entity based on its usage"""
    preds_as_obj = list(g.predicates(None, entity))
    preds_as_subj = list(g.predicates(entity, None))

    name_hint = entity.split("#")[-1]

    # Rule-based inference
    if any("recordingOf" in str(p) for p in preds_as_obj):
        return [MM.Album, SCHEMA.MusicAlbum]
    if any("hasPerformance" in str(p) for p in preds_as_obj):
        return [MM.LivePerformance]
    if any("formedBand" in str(p) for p in preds_as_obj):
        return [MM.MusicGroup]
    if any("location" in str(p) for p in preds_as_obj):
        return [SCHEMA.Place]
    if any("lyrics" in str(p) for p in preds_as_subj):
        return [MM.Song, SCHEMA.MusicRecording]

    # Default fallbacks based on name
    if "Magic" in name_hint or "Album" in name_hint:
        return [MM.Album]
    if "Song" in name_hint or "Track" in name_hint:
        return [SCHEMA.MusicRecording]
    if "Stadium" in name_hint or "Stage" in name_hint or "City" in name_hint:
        return [SCHEMA.Place]

    # Default to Thing if unknown
    return [SCHEMA.Thing]

# ==============================
# Infer and append
# ==============================
inferred = []
for ent in undefined_entities:
    ent_uri = URIRef(ent)
    inferred_types = infer_type_for_entity(ent_uri)
    for t in inferred_types:
        g.add((ent_uri, RDF.type, t))
    inferred.append({
        "entity": ent,
        "types": [str(t) for t in inferred_types],
        "context_predicates_as_object": [str(p) for p in g.predicates(None, ent_uri)][:5],
        "context_predicates_as_subject": [str(p) for p in g.predicates(ent_uri, None)][:5]
    })

# ==============================
# Save outputs
# ==============================
g.serialize(destination=OUTPUT_TTL, format="turtle")
with open(INFER_LOG, "w", encoding="utf-8") as f:
    json.dump(inferred, f, indent=2)

print(f"✅ Inferred {len(inferred)} undefined entities.")
print(f"Updated TTL: {OUTPUT_TTL}")
print(f"Inference log: {INFER_LOG}")
