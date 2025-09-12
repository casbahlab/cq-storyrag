#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_instruments_to_artist.py

For every mm:MusicEnsembleMembership:
  - Collect instruments from:
      * membership: schema:instrument
      * roles linked via core:involvesRole: schema:instrument
  - Add (artist schema:instrument instrument) for each artist linked via
    mm:involvesMemberOfMusicEnsemble
  - By default, REMOVE those instrument triples from the membership/roles (i.e., "move")
    Use --copy-only to keep originals.
Optional:
  - --prune-unreferenced-instruments : delete instrument individuals that nobody references anymore

No extra rdf:type assertions. Idempotent.

Usage:
  python move_instruments_to_artist.py \
    --in /path/in.ttl --out /path/out.ttl
  # copy without removing originals:
  python move_instruments_to_artist.py \
    --in /path/in.ttl --out /path/out.ttl --copy-only
  # also prune now-unreferenced instrument nodes:
  python move_instruments_to_artist.py \
    --in /path/in.ttl --out /path/out.ttl --prune-unreferenced-instruments
"""

import argparse
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def main():
    ap = argparse.ArgumentParser(description="Move instruments from memberships/roles to the artist.")
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--copy-only", action="store_true",
                    help="Copy instruments to artist but do NOT remove from membership/roles")
    ap.add_argument("--prune-unreferenced-instruments", action="store_true",
                    help="Delete instrument individuals that are no longer referenced by any triple")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_path, format="turtle")
    g.bind("schema", SCHEMA)
    g.bind("core", CORE)
    g.bind("mm", MM)

    memberships = set(g.subjects(RDF.type, MM.MusicEnsembleMembership))

    moved_to_artists = 0
    removed_from_memberships = 0
    removed_from_roles = 0
    pruned_instruments = 0

    for m in memberships:
        # artists linked via involvesMemberOfMusicEnsemble
        artists = list(g.objects(m, MM.involvesMemberOfMusicEnsemble))
        if not artists:
            continue

        # collect membership-level instruments
        mem_instrs = list(g.objects(m, SCHEMA.instrument))

        # collect role-level instruments from roles linked to the membership
        role_instrs = []
        roles = list(g.objects(m, CORE.involvesRole))
        for r in roles:
            role_instrs.extend(list(g.objects(r, SCHEMA.instrument)))

        all_instrs = set(mem_instrs + role_instrs)

        # attach to each artist
        for a in artists:
            for inst in all_instrs:
                if (a, SCHEMA.instrument, inst) not in g:
                    g.add((a, SCHEMA.instrument, inst))
                    moved_to_artists += 1

        # remove originals unless copying only
        if not args.copy_only:
            for inst in mem_instrs:
                g.remove((m, SCHEMA.instrument, inst))
                removed_from_memberships += 1
            for r in roles:
                for inst in list(g.objects(r, SCHEMA.instrument)):
                    g.remove((r, SCHEMA.instrument, inst))
                    removed_from_roles += 1

    # prune unreferenced instrument nodes (optional)
    if args.prune_unreferenced_instruments:
        # Any node still used as object of schema:instrument is referenced
        referenced = set(g.objects(None, SCHEMA.instrument))
        # Build a set of all objects in the graph
        all_objects = set(o for (_, _, o) in g)
        # Remove subjects that have no incoming edges at all
        subjects = set(s for (s, _, _) in g)
        for node in list(subjects):
            if node in referenced:
                continue
            # If node is not used as any object anywhere, drop its outgoing triples
            if node not in all_objects:
                for t in list(g.triples((node, None, None))):
                    g.remove(t)
                pruned_instruments += 1

    g.serialize(destination=args.out_path, format="turtle")

    print(f"Memberships processed: {len(memberships)}")
    print(f"Instruments attached to artists: {moved_to_artists}")
    print(f"Removed membership-level instruments: {removed_from_memberships}")
    print(f"Removed role-level instruments: {removed_from_roles}")
    print(f"Pruned unreferenced nodes: {pruned_instruments}")
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()

