import json
import csv
from rdflib import Graph, Namespace, URIRef, RDFS
from rdflib.namespace import RDF

# ==== CONFIG ====
INPUT_KG = "data/enriched_kg_songs.ttl"
OUTPUT_KG = "data/enriched_kg_performances.ttl"
LOG_FILE = "logs/performance_enrichment_log.csv"
CACHE_FILE = "logs/verified_performance_mappings.json"

SCHEMA = Namespace("https://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

# Verified artist cache (from previous enrichment)
ARTIST_CACHE_FILE = "logs/verified_artist_mappings.json"
with open(ARTIST_CACHE_FILE, "r", encoding="utf-8") as f:
    artist_cache = json.load(f)

# Synthetic event: Live Aid
LIVE_AID_WIKIDATA = "http://www.wikidata.org/entity/Q184199"
LIVE_AID_WIKIPEDIA = "https://en.wikipedia.org/wiki/Live_Aid"

# ---- Load cache if exists ----
try:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        verified_cache = json.load(f)
except FileNotFoundError:
    verified_cache = {}

# ---- Extract Performances from KG ----
def extract_performances(graph):
    """Extract all live performances and broadcast events from the KG."""
    performances = []
    seen = set()

    # mm:LivePerformance
    for s in graph.subjects(RDF.type, MM.LivePerformance):
        label = graph.value(s, RDFS.label)
        if label and s not in seen:
            performances.append({"uri": str(s), "label": str(label)})
            seen.add(s)

    return performances

# ---- Synthetic Enrichment ----
def enrich_synthetic_performance(graph, perf):
    subj = URIRef(perf["uri"])
    label = perf["label"]

    # Extract artist name
    artist_name = label.split("Live Aid")[0].strip()
    artist_entry = artist_cache.get(artist_name)

    if not artist_entry:
        print(f"⚠ No artist mapping for {artist_name}, skipping synthetic enrichment.")
        return None

    # Normalize to dict
    if isinstance(artist_entry, str):
        artist_entry = {"wikidata": artist_entry, "musicbrainz": "", "wikipedia": ""}

    wikidata_artist = artist_entry.get("wikidata", "")
    wikipedia_artist = artist_entry.get("wikipedia", "")

    # Add triples
    if wikidata_artist:
        graph.add((subj, SCHEMA.performer, URIRef(wikidata_artist)))
        graph.add((subj, SCHEMA.sameAs, URIRef(wikidata_artist)))

    # Link to Live Aid event
    graph.add((subj, SCHEMA.inEvent, URIRef(LIVE_AID_WIKIDATA)))
    graph.add((subj, SCHEMA.subjectOf, URIRef(LIVE_AID_WIKIPEDIA)))

    if wikipedia_artist:
        graph.add((subj, SCHEMA.subjectOf, URIRef(wikipedia_artist)))

    return {
        "artist": artist_name,
        "wikidata_artist": wikidata_artist,
        "wikidata_event": LIVE_AID_WIKIDATA,
        "wikipedia_event": LIVE_AID_WIKIPEDIA,
        "wikipedia_artist": wikipedia_artist
    }


# ---- Main Pipeline ----
def main():
    g = Graph()
    g.parse(INPUT_KG, format="ttl")
    performances = extract_performances(g)
    print(f"Found {len(performances)} performances to enrich (synthetic mode).")

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "PerformanceLabel","EntityURI",
            "WikidataArtist","WikidataEvent","WikipediaEvent","WikipediaArtist"
        ])
        writer.writeheader()

        for perf in performances:
            label = perf["label"]

            if label in verified_cache:
                print(f"✅ Using cached enrichment for {label}")
                continue

            enriched_entry = enrich_synthetic_performance(g, perf)
            if enriched_entry:
                verified_cache[label] = enriched_entry
                writer.writerow({
                    "PerformanceLabel": label,
                    "EntityURI": perf["uri"],
                    "WikidataArtist": enriched_entry["wikidata_artist"],
                    "WikidataEvent": enriched_entry["wikidata_event"],
                    "WikipediaEvent": enriched_entry["wikipedia_event"],
                    "WikipediaArtist": enriched_entry["wikipedia_artist"]
                })
                print(f"Enriched {label} with synthetic links")

    # Save cache + KG
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(verified_cache, f, indent=2)

    g.serialize(OUTPUT_KG, format="ttl")
    print(f"\nEnriched TTL saved to {OUTPUT_KG}")
    print(f"Log saved to {LOG_FILE}")
    print(f"Cache updated at {CACHE_FILE}")

if __name__ == "__main__":
    main()
