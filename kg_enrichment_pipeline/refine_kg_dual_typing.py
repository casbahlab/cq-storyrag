import rdflib
from rdflib import Graph, Namespace, URIRef, RDF
import json
import os

# ---------------- CONFIG ----------------
INPUT_FILE = "output/liveaid_instances.ttl"
OUTPUT_FILE = "output/final_refined_liveaid.ttl"
LOG_FILE = "output/refinement_log.json"

SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX = Namespace("http://wembrewind.live/ex#")

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load the graph
g = Graph()
g.parse(INPUT_FILE, format="turtle")
print(f"✅ Parsed KG with {len(g)} triples")

# Prepare refinement log
refinement_log = {}

# Helper function to add dual typing if not already present
def add_dual_type(entity, *types):
    added = []
    for t in types:
        if (entity, RDF.type, t) not in g:
            g.add((entity, RDF.type, t))
            added.append(str(t))
    return added

# Detect and classify entities
for s in set(g.subjects()):
    s_str = str(s)
    existing_types = list(g.objects(s, RDF.type))
    new_types = []

    # --- 1. Performances / Events ---
    if (s, SCHEMA.location, None) in g or (s, MM.performedAt, None) in g:
        new_types += add_dual_type(s, MM.LivePerformance, SCHEMA.MusicEvent)
        # Custom narrative class
        new_types += add_dual_type(s, EX.BenefitConcert)

    # --- 2. Musical Works / Songs ---
    if (s, SCHEMA.byArtist, None) in g or (s, MM.performedBy, None) in g:
        new_types += add_dual_type(s, MM.MusicalWork, SCHEMA.MusicRecording)

    # --- 3. Recordings / Broadcasts ---
    if (s, MM.recordedAs, None) in g or (s, SCHEMA.duration, None) in g:
        new_types += add_dual_type(s, MM.Recording, SCHEMA.MediaObject)

    # --- 4. Stadiums / Stages / Venues ---
    if (s, SCHEMA.address, None) in g or (s, SCHEMA.addressLocality, None) in g:
        new_types += add_dual_type(
            s, SCHEMA.Place, SCHEMA.MusicVenue, SCHEMA.StadiumOrArena, EX.LiveAidVenue
        )

    # Log refinements
    if new_types:
        refinement_log[s_str] = {
            "existing_types": [str(t) for t in existing_types],
            "new_types": new_types,
        }

# Serialize refined TTL
g.serialize(OUTPUT_FILE, format="turtle")
print(f"✅ Refined KG written to {OUTPUT_FILE}")

# Save refinement log
with open(LOG_FILE, "w", encoding="utf-8") as log:
    json.dump(refinement_log, log, indent=2)
print(f"📝 Refinement log saved to {LOG_FILE}")
print(f"🔹 Entities refined: {len(refinement_log)}")
