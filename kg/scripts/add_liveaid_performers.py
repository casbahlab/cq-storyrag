#!/usr/bin/env python3
from rdflib import Graph, Namespace, URIRef
from pathlib import Path

# Namespaces
SCHEMA = Namespace("http://schema.org/")
EX = Namespace("http://wembrewind.live/ex#")

# Files
TTL_PATH = Path("kg/20_artists.ttl")  # adjust if needed
OUTPUT_PATH = TTL_PATH  # overwrite, or change to Path("kg/20_artists_updated.ttl")

# Load graph
g = Graph()
g.parse(TTL_PATH, format="turtle")
print(f"[load] {len(g)} triples from {TTL_PATH}")

# Target event
live_aid = EX.LiveAid1985

# Find all performers
performers = set(
    list(g.subjects(predicate=None, object=SCHEMA.Person)) +
    list(g.subjects(predicate=None, object=SCHEMA.MusicGroup))
)
# Also capture mm:MusicArtist if present
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
performers |= set(g.subjects(predicate=None, object=MM.MusicArtist))

print(f"[scan] found {len(performers)} performer entities")

# Add schema:performer triples
added = 0
for perf in performers:
    triple = (live_aid, SCHEMA.performer, perf)
    if triple not in g:
        g.add(triple)
        added += 1

print(f"[add] added {added} new schema:performer triples")

# Save
g.serialize(destination=OUTPUT_PATH, format="turtle")
print(f"[write] saved to {OUTPUT_PATH}")
