#!/usr/bin/env python3
# diff_musicians.py
# Find mm:Musician individuals present in the OLD file but not in the NEW file,
# and write a TTL with just those subjects' triples (copied from OLD).
#
# Usage:
#   python diff_musicians.py --old 20_artists.ttl --new new.ttl --out missing_musicians.ttl
# Options:
#   --presence subject|type
#       subject (default): consider an artist "present in NEW" if it appears as a subject of any triple in NEW.
#
import argparse
from rdflib import Graph, Namespace, RDF, URIRef

MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def load_graph(path, forced_format=None):
    g = Graph()
    if forced_format:
        g.parse(path, format=forced_format)
        return g
    for fmt in ["turtle", "xml", "n3", "nt", "trig", "json-ld"]:
        try:
            g.parse(path, format=fmt)
            return g
        except Exception:
            continue
    raise RuntimeError(f"Failed to parse {path} in common RDF formats. Try --format ttl.")

def subject_exists(g: Graph, s: URIRef) -> bool:
    # True if subject s appears as a subject in any triple
    for _ in g.triples((s, None, None)):
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Old RDF file with full set (TTL/RDF/JSON-LD).")
    ap.add_argument("--new", required=True, help="New RDF file missing some artists.")
    ap.add_argument("--out", required=True, help="Output TTL with missing mm:Musician subjects from OLD.")
    ap.add_argument("--format", default=None, help="Force parse format for both files (e.g., ttl, xml, nt, json-ld).")
    ap.add_argument("--presence", choices=["subject","type"], default="subject",
                    help="How to decide presence in NEW. Default: subject.")
    args = ap.parse_args()

    old_g = load_graph(args.old, forced_format=args.format)
    new_g = load_graph(args.new, forced_format=args.format)

    # Collect mm:Musician subjects in OLD
    old_musicians = set(s for s in old_g.subjects(RDF.type, MM.Musician) if isinstance(s, URIRef))

    # Decide which are present in NEW
    if args.presence == "subject":
        present_new = set(s for s in old_musicians if subject_exists(new_g, s))
    else:  # presence == "type"
        present_new = set(s for s in old_musicians if (s, RDF.type, MM.Musician) in new_g)

    missing = sorted(old_musicians - present_new)

    # Build output graph with only missing subjects' triples from OLD
    out_g = Graph()
    out_g.bind("mm", MM)
    # copy all triples with subject in missing from OLD
    added = 0
    for s in missing:
        for p,o in old_g.predicate_objects(s):
            out_g.add((s,p,o))
            added += 1

    out_g.serialize(destination=args.out, format="turtle")

    print(f"[diff] mm:Musician in OLD: {len(old_musicians)}")
    print(f"[diff] present in NEW (mode={args.presence}): {len(present_new)}")
    print(f"[diff] missing in NEW: {len(missing)}")
    print(f"[write] {args.out} with {added} triples")

if __name__ == "__main__":
    main()
