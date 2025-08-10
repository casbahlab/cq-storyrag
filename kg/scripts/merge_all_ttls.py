#!/usr/bin/env python3
"""
Merge all KG module TTLs into a single master file.
- Reads known module files in order (if present)
- Writes: kg/liveaid_instances_master.ttl
- Requires: rdflib

Usage:
  python kg/scripts/merge_all_ttls.py
"""

import os
from rdflib import Graph

KG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASTER_OUT = os.path.join(KG_DIR, "liveaid_instances_master.ttl")

MODULES_IN_ORDER = [
    "10_core_entities.ttl",
    "20_artists.ttl",
    "30_performances.ttl",
    "40_setlists_songs.ttl",
    "50_instruments.ttl",
    "60_reviews.ttl",
    "70_conditions.ttl",
    "80_provenance.ttl",
    "81_links_sameAs.ttl",
    "82_external_ids_artists.ttl",
    "83_external_ids_songs.ttl",
    "84_external_links_performances.ttl",
    "85_artist_mbids.ttl",
]

def main():
    g = Graph()
    count_before = 0
    for fname in MODULES_IN_ORDER:
        path = os.path.join(KG_DIR, fname)
        if os.path.exists(path):
            g.parse(path, format="turtle")
            print(f"[merge] parsed: {fname}  (triples now: {len(g)})")
        else:
            print(f"[merge] missing: {fname} (skipped)")
    g.serialize(MASTER_OUT, format="turtle")
    print(f"[merge] wrote master: {MASTER_OUT}  (total triples: {len(g)})")

if __name__ == "__main__":
    main()
