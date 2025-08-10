#!/usr/bin/env python3
"""
Apply artist external IDs using your existing verified_artist_mappings.json format.
- JSON format (current): { "Artist Name": "https://www.wikidata.org/wiki/Q15862", ... }
- Also supports richer entries: { "Artist Name": {"wikidata": "...", "wikipedia": "...", "musicbrainz": "MBID"} }

Emits:
- owl:sameAs for Wikidata URLs
- schema:sameAs for Wikipedia URLs
- schema:identifier as schema:PropertyValue for MBIDs (propertyID="musicbrainz")

Writes to: kg/82_external_ids_artists.ttl
"""
import json, os
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF

KG_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_TTL = os.path.join(KG_DIR, "liveaid_instances_master.ttl")
OUTPUT_TTL = os.path.join(KG_DIR, "82_external_ids_artists.ttl")
MAPPINGS_FILE = os.path.join(KG_DIR, "mappings", "verified_artist_mappings.json")

SCHEMA = Namespace("http://schema.org/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

def add_identifier(gx, subj, system, value):
    pv = BNode()
    gx.add((subj, SCHEMA.identifier, pv))
    gx.add((pv, RDF.type, SCHEMA.PropertyValue))
    gx.add((pv, SCHEMA.propertyID, Literal(system)))
    gx.add((pv, SCHEMA.value, Literal(value)))

def main():
    ref = Graph().parse(INPUT_TTL, format="turtle")

    # Build label->URI map for artists
    label_to_uri = {}
    for s, p, o in ref.triples((None, SCHEMA.name, None)):
        label_to_uri[str(o)] = s

    with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("owl", OWL)
    added = 0

    for label, payload in raw.items():
        subj = label_to_uri.get(label)
        if not subj:
            continue

        wikidata = None; wikipedia = None; mbid = None

        if isinstance(payload, str) and "wikidata.org" in payload:
            wikidata = payload
        elif isinstance(payload, dict):
            wikidata = payload.get("wikidata")
            wikipedia = payload.get("wikipedia")
            mbid = payload.get("musicbrainz")

        if wikidata:
            out.add((subj, OWL.sameAs, URIRef(wikidata))); added += 1
        if wikipedia:
            out.add((subj, SCHEMA.sameAs, URIRef(wikipedia))); added += 1
        if mbid:
            add_identifier(out, subj, "musicbrainz", mbid); added += 1

    out.serialize(OUTPUT_TTL, format="turtle")
    print(f"[artists] wrote {added} triples → {OUTPUT_TTL}")

if __name__ == "__main__":
    main()
