#!/usr/bin/env python3
"""
Apply song external IDs using your existing verified_song_mappings.json format.
- JSON format (current): { "Song Title": {"wikidata": URL, "wikipedia": URL, "musicbrainz": "MBID"}, ... }

Emits:
- owl:sameAs for Wikidata URLs
- schema:sameAs for Wikipedia URLs
- schema:identifier as schema:PropertyValue for MBIDs (propertyID="musicbrainz")

Writes to: kg/83_external_ids_songs.ttl
"""
import json, os
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF

KG_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_TTL = os.path.join(KG_DIR, "liveaid_instances_master.ttl")
OUTPUT_TTL = os.path.join(KG_DIR, "83_external_ids_songs.ttl")
MAPPINGS_FILE = os.path.join(KG_DIR, "mappings", "verified_song_mappings.json")

SCHEMA = Namespace("http://schema.org/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

def add_identifier(gx, subj, system, value):
    pv = BNode()
    gx.add((subj, SCHEMA.identifier, pv))
    gx.add((pv, RDF.type, SCHEMA.PropertyValue))
    gx.add((pv, SCHEMA.propertyID, Literal(system)))
    gx.add((pv, SCHEMA.value, Literal(value)))

def main():
    ref = Graph()
    # Load modules where songs/names may live
    for fname in ["liveaid_instances_master.ttl"]:
        ref.parse(os.path.join(KG_DIR, fname), format="turtle")

    # Build label->URI map
    label_to_uri = {}
    for s, p, o in ref.triples((None, SCHEMA.name, None)):
        label_to_uri[str(o)] = s

    with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
        verified = json.load(f)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("owl", OWL)
    added = 0

    for title, ids in verified.items():
        subj = label_to_uri.get(title)
        if not subj:
            continue
        wikidata = ids.get("wikidata")
        wikipedia = ids.get("wikipedia")
        mbid = ids.get("musicbrainz")
        if wikidata:
            out.add((subj, OWL.sameAs, URIRef(wikidata))); added += 1
        if wikipedia:
            out.add((subj, SCHEMA.sameAs, URIRef(wikipedia))); added += 1
        if mbid:
            add_identifier(out, subj, "musicbrainz", mbid); added += 1

    out.serialize(OUTPUT_TTL, format="turtle")
    print(f"[songs] wrote {added} triples → {OUTPUT_TTL}")

if __name__ == "__main__":
    main()
