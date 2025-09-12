#!/usr/bin/env python3
# diff_entities.py
# Find individuals of given classes present in OLD but not in NEW,
# and write a TTL containing just those subjects' triples from OLD.
#
# Usage:
#   python diff_entities.py --old 20_artists.ttl --new new.ttl \
#       --classes mm:Musician,mm:MusicEnsembleMembership \
#       --out missing_entities.ttl
#
# Options:
#   --presence subject|type
#       subject (default): consider an entity "present in NEW" if the URI appears as a subject in NEW.
#       type: consider "present in NEW" only if it has rdf:type <class> in NEW.
#   --split-per-class
#       Also writes one TTL per class: <out_prefix>_<LocalName>.ttl
#
import argparse
from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import split_uri

# Default prefixes (extend as needed)
PREFIXES = {
    "mm":     "https://w3id.org/polifonia/ontology/music-meta/",
    "core":   "https://w3id.org/polifonia/ontology/core/",
    "schema": "http://schema.org/",
    "ex":     "http://example.org/",
}

def resolve_iri(token: str) -> URIRef:
    token = token.strip()
    if token.startswith("http://") or token.startswith("https://"):
        return URIRef(token)
    if ":" in token:
        pref, local = token.split(":", 1)
        base = PREFIXES.get(pref)
        if base:
            return URIRef(base + local)
    raise ValueError(f"Cannot resolve class IRI from '{token}'. Use full IRI or a known prefix: {', '.join(PREFIXES)}")

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
    for _ in g.triples((s, None, None)):
        return True
    return False

def localname(u: URIRef) -> str:
    try:
        _ns, ln = split_uri(u)
        return ln
    except Exception:
        s = str(u)
        for sep in ['#','/']:
            if sep in s:
                return s.rsplit(sep, 1)[-1]
        return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Old RDF file (TTL/RDF/JSON-LD).")
    ap.add_argument("--new", required=True, help="New RDF file after update.")
    ap.add_argument("--classes", required=True, help="Comma-separated list of class CURIEs or IRIs (e.g., mm:Musician,mm:MusicEnsembleMembership).")
    ap.add_argument("--out", required=True, help="Output TTL with all missing entities from OLD.")
    ap.add_argument("--format", default=None, help="Force parse format for both files (ttl, xml, nt, json-ld).")
    ap.add_argument("--presence", choices=["subject","type"], default="subject", help="Presence criterion in NEW. Default: subject.")
    ap.add_argument("--split-per-class", action="store_true", help="Also emit one TTL per class.")
    args = ap.parse_args()

    # Resolve classes
    class_tokens = [t for t in args.classes.split(",") if t.strip()]
    class_iris = [resolve_iri(t) for t in class_tokens]

    old_g = load_graph(args.old, forced_format=args.format)
    new_g = load_graph(args.new, forced_format=args.format)

    # Prepare combined and per-class graphs
    out_combined = Graph()
    for pfx, ns in PREFIXES.items():
        out_combined.bind(pfx, Namespace(ns))

    per_class = {}
    if args.split_per_class:
        for c in class_iris:
            g = Graph()
            for pfx, ns in PREFIXES.items():
                g.bind(pfx, Namespace(ns))
            per_class[c] = g

    total_old = 0
    total_present = 0
    total_missing = 0
    total_triples = 0

    for C in class_iris:
        # gather subjects of class C in OLD (URIs only)
        olds = set(s for s in old_g.subjects(RDF.type, C) if isinstance(s, URIRef))
        total_old += len(olds)

        # determine presence in NEW
        if args.presence == "subject":
            present = set(s for s in olds if subject_exists(new_g, s))
        else:  # presence == "type"
            present = set(s for s in olds if (s, RDF.type, C) in new_g)
        total_present += len(present)

        missing = sorted(olds - present)
        total_missing += len(missing)

        # copy triples for missing subjects from OLD
        for s in missing:
            for p,o in old_g.predicate_objects(s):
                out_combined.add((s,p,o))
                total_triples += 1
                if args.split_per_class:
                    per_class[C].add((s,p,o))

        # If split, write file per class
    out_combined.serialize(destination=args.out, format="turtle")

    if args.split_per_class:
        prefix = args.out.rsplit(".",1)[0]
        for C, g in per_class.items():
            path = f"{prefix}_{localname(C)}.ttl"
            g.serialize(destination=path, format="turtle")
            print(f"[write] {path} ({len(g)})")

    print(f"[diff] classes: {', '.join(str(c) for c in class_iris)}")
    print(f"[diff] total in OLD: {total_old}")
    print(f"[diff] present in NEW (mode={args.presence}): {total_present}")
    print(f"[diff] missing in NEW: {total_missing}")
    print(f"[write] {args.out} with {total_triples} triples")

if __name__ == "__main__":
    main()
