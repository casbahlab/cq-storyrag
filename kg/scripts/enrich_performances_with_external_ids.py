#!/usr/bin/env python3
"""
Apply performance-related external links using your existing verified_performance_mappings.json format.
- JSON format (current) examples:
  {
    "Queen Live Aid Performance 1": {
      "wikidata_event": "https://www.wikidata.org/wiki/Q193740",
      "wikipedia_event": "https://en.wikipedia.org/wiki/Live_Aid",
      "wikidata_artist": "https://www.wikidata.org/wiki/Q15862",
      "event_iri": "http://wembrewind.live/ex#LiveAid1985",
      "artist_iri": "http://wembrewind.live/ex#Queen",
      "venue_iri": "http://wembrewind.live/ex#WembleyStadium"
    },
    ...
  }

What we emit (best-effort, respecting semantics):
- If there is a dedicated Wikidata/Wikipedia page *for the performance entity itself*, accept keys:
    "wikidata_performance": URL  → add owl:sameAs
    "wikipedia_performance": URL → add schema:sameAs
- For event/artist links in the mapping:
    "wikipedia_event": URL       → add schema:isBasedOn (source for performance facts)
    "wikidata_event": URL        → (we DO NOT set owl:sameAs to the event; they are different entities)
    "wikidata_artist": URL       → add schema:mentions (reference to the artist's external entity)
- If no label match for the performance:
    fall back to logic matching using (event_iri, artist_iri, venue_iri). If exactly one candidate is found,
    apply the links to that node.

Writes to: kg/84_external_links_performances.ttl
"""
import json, os
from rdflib import Graph, Namespace, URIRef

KG_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_TTL = os.path.join(KG_DIR, "liveaid_instances_master.ttl")
OUTPUT_TTL = os.path.join(KG_DIR, "84_external_links_performances.ttl")
MAPPINGS_FILE = os.path.join(KG_DIR, "mappings", "verified_performance_mappings.json")

SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")

def main():
    ref = Graph().parse(INPUT_TTL, format="turtle")

    # Label -> URI map
    label_to_uri = {}
    for s, p, o in ref.triples((None, SCHEMA.name, None)):
        label_to_uri[str(o)] = s

    with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
        verified = json.load(f)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("owl", OWL)
    added = 0

    for perf_label, ids in verified.items():
        subj = label_to_uri.get(perf_label)

        if not subj:
            # try logic match
            ev = ids.get("event_iri"); ar = ids.get("artist_iri"); ve = ids.get("venue_iri")
            candidates = set()
            if ev and ar and ve:
                for s, _, _ in ref.triples((None, None, None)):
                    pass  # iteration trigger
                for s, _, _ in ref.triples((None, SCHEMA.isPartOf, URIRef(ev))):
                    if (s, MM.isCreatedBy, URIRef(ar)) in ref and (s, SCHEMA.location, URIRef(ve)) in ref:
                        candidates.add(s)
            if len(candidates) == 1:
                subj = next(iter(candidates))

        if not subj:
            continue

        # performance-level external identity (if present)
        if ids.get("wikidata_performance"):
            out.add((subj, OWL.sameAs, URIRef(ids["wikidata_performance"]))); added += 1
        if ids.get("wikipedia_performance"):
            out.add((subj, SCHEMA.sameAs, URIRef(ids["wikipedia_performance"]))); added += 1

        # event/artist references as provenance/context
        if ids.get("wikipedia_event"):
            out.add((subj, SCHEMA.isBasedOn, URIRef(ids["wikipedia_event"]))); added += 1
        if ids.get("wikidata_artist"):
            out.add((subj, SCHEMA.mentions, URIRef(ids["wikidata_artist"]))); added += 1

    out.serialize(OUTPUT_TTL, format="turtle")
    print(f"[performances] wrote {added} triples → {OUTPUT_TTL}")

if __name__ == "__main__":
    main()
