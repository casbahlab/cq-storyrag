#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tidy_memberships_add_hasMember.py

- For every mm:MusicEnsembleMembership:
    * Add (artist core:isMemberOf ensemble)
    * Add (ensemble core:hasMember artist)
- Instrument cleanup (optional flags):
    * --drop-membership-instruments : remove schema:instrument on membership nodes
    * --drop-role-instruments       : remove schema:instrument on role nodes linked via core:involvesRole
    * --drop-artist-instruments     : remove schema:instrument on artist nodes (subjects of core:isMemberOf)
    * --prune-unreferenced-instruments : delete instrument individuals that are no longer referenced by any triple

No extra rdf:type assertions are added. Idempotent (safe to run multiple times).

Usage:
  python tidy_memberships_add_hasMember.py \
    --in  /path/in.ttl \
    --out /path/out.ttl \
    --drop-membership-instruments \
    --drop-role-instruments \
    --drop-artist-instruments \
    --prune-unreferenced-instruments
"""

import argparse
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def main():
    ap = argparse.ArgumentParser(description="Map memberships to isMemberOf/hasMember and clean instrument assertions.")
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--drop-membership-instruments", action="store_true",
                    help="Remove schema:instrument from membership nodes")
    ap.add_argument("--drop-role-instruments", action="store_true",
                    help="Remove schema:instrument from role nodes linked via core:involvesRole")
    ap.add_argument("--drop-artist-instruments", action="store_true",
                    help="Remove schema:instrument from artist nodes (subjects of core:isMemberOf)")
    ap.add_argument("--prune-unreferenced-instruments", action="store_true",
                    help="Delete instrument individuals that are no longer referenced by any triple")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_path, format="turtle")
    g.bind("schema", SCHEMA)
    g.bind("core", CORE)
    g.bind("mm", MM)

    memberships = set(g.subjects(RDF.type, MM.MusicEnsembleMembership))

    mapped_isMemberOf = 0
    mapped_hasMember  = 0
    removed_mem_instr = 0
    removed_role_instr = 0
    removed_artist_instr = 0
    pruned_instruments = 0

    for m in memberships:
        # artists linked via involvesMemberOfMusicEnsemble
        artists   = list(g.objects(m, MM.involvesMemberOfMusicEnsemble))
        # ensembles linked via involvesMusicEnsemble
        ensembles = list(g.objects(m, MM.involvesMusicEnsemble))

        for a in artists:
            for e in ensembles:
                if (a, CORE.isMemberOf, e) not in g:
                    g.add((a, CORE.isMemberOf, e))
                    mapped_isMemberOf += 1
                if (e, CORE.hasMember, a) not in g:
                    g.add((e, CORE.hasMember, a))
                    mapped_hasMember += 1

        # Drop instruments on membership (optional)
        if args.drop_membership_instruments:
            for inst in list(g.objects(m, SCHEMA.instrument)):
                g.remove((m, SCHEMA.instrument, inst))
                removed_mem_instr += 1

        # Drop instruments on roles attached to this membership (optional)
        if args.drop_role_instruments:
            for role in g.objects(m, CORE.involvesRole):
                for inst in list(g.objects(role, SCHEMA.instrument)):
                    g.remove((role, SCHEMA.instrument, inst))
                    removed_role_instr += 1

    # Drop instruments on artist nodes (optional)
    if args.drop_artist_instruments:
        for artist in set(g.subjects(predicate=CORE.isMemberOf)):
            for inst in list(g.objects(artist, SCHEMA.instrument)):
                g.remove((artist, SCHEMA.instrument, inst))
                removed_artist_instr += 1

    # Prune unreferenced instrument individuals (optional)
    if args.prune_unreferenced_instruments:
        # Recompute current instrument objects and prune those with no incoming edges
        instrument_nodes = set(g.objects(None, SCHEMA.instrument))
        for inst_node in list(instrument_nodes):
            still_referenced = any(True for _ in g.subjects(None, inst_node))
            if not still_referenced:
                # remove all outgoing triples (labels, types, sameAs, etc.)
                for t in list(g.triples((inst_node, None, None))):
                    g.remove(t)
                # ensure no lingering schema:instrument links
                for t in list(g.triples((None, SCHEMA.instrument, inst_node))):
                    g.remove(t)
                pruned_instruments += 1

    g.serialize(destination=args.out_path, format="turtle")

    print(f"Memberships processed: {len(memberships)}")
    print(f"Added core:isMemberOf: {mapped_isMemberOf}")
    print(f"Added core:hasMember:  {mapped_hasMember}")
    print(f"Removed membership instruments: {removed_mem_instr}")
    print(f"Removed role instruments:       {removed_role_instr}")
    print(f"Removed artist instruments:     {removed_artist_instr}")
    print(f"Pruned instrument individuals:  {pruned_instruments}")
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
