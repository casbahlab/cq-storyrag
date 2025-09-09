#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tidy_memberships_isMemberOf.py

- Map mm:MusicEnsembleMembership → (artist core:isMemberOf ensemble)
- Remove instrument assertions (membership-level by default)
- Optional: also drop role-level / artist-level instruments
- Optional: prune unreferenced instrument individuals
- No extra rdf:type assertions are added.

Usage:
  python tidy_memberships_isMemberOf.py \
    --in  /path/in.ttl \
    --out /path/out.ttl \
    --drop-membership-instruments \
    --drop-role-instruments \
    --drop-artist-instruments \
    --prune-unreferenced-instruments
"""

import argparse
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def drop_triples(g: Graph, s=None, p=None, o=None):
    for t in list(g.triples((s, p, o))):
        g.remove(t)

def main():
    ap = argparse.ArgumentParser(description="Map memberships to core:isMemberOf and clean instrument assertions.")
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--drop-membership-instruments", action="store_true",
                    help="Remove schema:instrument from membership nodes")
    ap.add_argument("--drop-role-instruments", action="store_true",
                    help="Remove schema:instrument from role nodes linked via core:involvesRole")
    ap.add_argument("--drop-artist-instruments", action="store_true",
                    help="Remove schema:instrument from artist (Musician) nodes")
    ap.add_argument("--prune-unreferenced-instruments", action="store_true",
                    help="Delete instrument individuals that are no longer referenced by any triple")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_path, format="turtle")
    g.bind("schema", SCHEMA)
    g.bind("core", CORE)
    g.bind("mm", MM)

    memberships = set(g.subjects(RDF.type, MM.MusicEnsembleMembership))
    mapped = 0
    mem_instr_removed = 0
    role_instr_removed = 0
    artist_instr_removed = 0
    pruned_instruments = 0

    for m in memberships:
        # Map memberships → isMemberOf
        artists   = list(g.objects(m, MM.involvesMemberOfMusicEnsemble))
        ensembles = list(g.objects(m, MM.involvesMusicEnsemble))

        for a in artists:
            for e in ensembles:
                g.add((a, CORE.isMemberOf, e))
                mapped += 1

        # Remove instruments on membership
        if args.drop_membership_instruments:
            for inst in list(g.objects(m, SCHEMA.instrument)):
                g.remove((m, SCHEMA.instrument, inst))
                mem_instr_removed += 1

        # Remove instruments on role nodes linked via core:involvesRole
        if args.drop_role_instruments:
            for role in g.objects(m, CORE.involvesRole):
                for inst in list(g.objects(role, SCHEMA.instrument)):
                    g.remove((role, SCHEMA.instrument, inst))
                    role_instr_removed += 1

    # Remove instruments on artist nodes (optional)
    if args.drop_artist_instruments:
        # Heuristic: subjects of core:isMemberOf are your artist/musician nodes
        for artist in set(g.subjects(predicate=CORE.isMemberOf)):
            for inst in list(g.objects(artist, SCHEMA.instrument)):
                g.remove((artist, SCHEMA.instrument, inst))
                artist_instr_removed += 1

    # Prune unreferenced instrument individuals (optional)
    if args.prune_unreferenced_instruments:
        # Find nodes that *look* like instrument individuals: appear as object of schema:instrument
        instrument_nodes = set(g.objects(None, SCHEMA.instrument))
        for inst_node in instrument_nodes:
            # If no one points to it anymore as object, and it has no outgoing triples that matter, prune.
            incoming = any(True for _ in g.subject_predicates(inst_node))
            if not incoming:
                # delete all triples where it is subject (labels, types, sameAs, etc.)
                for t in list(g.triples((inst_node, None, None))):
                    g.remove(t)
                pruned_instruments += 1

    g.serialize(destination=args.out_path, format="turtle")

    print(f"Memberships: {len(memberships)}")
    print(f"core:isMemberOf triples added: {mapped}")
    print(f"Removed membership instruments: {mem_instr_removed}")
    print(f"Removed role instruments: {role_instr_removed}")
    print(f"Removed artist instruments: {artist_instr_removed}")
    print(f"Pruned instrument individuals: {pruned_instruments}")
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
