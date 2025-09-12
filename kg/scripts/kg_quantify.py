#!/usr/bin/env python3
"""
kg_quantify.py

Quantify a KG by reporting:
1) Unique rdf:type combinations per subject + counts
2) Per-class instance counts
3) Predicate triple counts
4) Total triple count

Usage examples:

# Single file
python kg_quantify.py \
  --in /path/to/liveaid_instances_master.ttl \
  --out-combos /path/to/kg_full_combos.csv \
  --out-classes /path/to/kg_full_classes.csv \
  --out-preds /path/to/kg_full_predicates.csv \
  --print-summary

# Multiple files combined
python kg_quantify.py \
  --in /path/to/20_artists.ttl \
  --in /path/to/40_setlists_songs.ttl \
  --out-combos /path/to/kg_combos.csv \
  --out-classes /path/to/kg_classes.csv \
  --out-preds /path/to/kg_predicates.csv \
  --print-summary
"""

import argparse
import csv
from collections import Counter, defaultdict
from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal

SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")

def lname(u: URIRef) -> str:
    s = str(u)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def get_label(g: Graph, node: URIRef) -> str:
    # Prefer schema:name or rdfs:label for class label printing
    for p in (SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label):
        val = g.value(node, p)
        if isinstance(val, Literal):
            return str(val)
    return lname(node)

def parse_any(g: Graph, path: str):
    # Try Turtle first then let rdflib guess
    try:
        g.parse(path, format="turtle")
    except Exception:
        g.parse(path)

def main():
    ap = argparse.ArgumentParser(
        description="Quantify a KG: type combinations, class counts, predicate counts, triple totals."
    )
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="Input RDF file(s). Can be repeated.")
    ap.add_argument("--out-combos", required=True, help="CSV path for type-combination counts.")
    ap.add_argument("--out-classes", required=True, help="CSV path for per-class instance counts.")
    ap.add_argument("--out-preds", required=False, help="CSV path for predicate triple counts.")
    ap.add_argument("--print-summary", action="store_true", help="Print a human-readable summary.")
    args = ap.parse_args()

    g = Graph()
    for f in args.inputs:
        parse_any(g, f)

    # Total triples
    triple_count = len(g)

    # Predicate counts
    pred_counts = Counter()
    for _, p, _ in g:
        pred_counts[p] += 1

    # Per-class instance counts and subject->types
    class_counts = Counter()
    subject_types = defaultdict(set)
    for s, _, o in g.triples((None, RDF.type, None)):
        subject_types[s].add(o)
        class_counts[o] += 1

    # Unique type combinations per subject
    combo_counts = Counter()
    for s, types in subject_types.items():
        combo_counts[frozenset(types)] += 1

    # Write type-combination counts
    with open(args.out_combos, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo_index", "count", "n_types", "types_uris", "types_labels"])
        # Sort by count desc then by type local names for stability
        for i, (combo, cnt) in enumerate(
            sorted(combo_counts.items(),
                   key=lambda x: (-x[1], sorted(lname(t) for t in x[0])))
        ):
            types_uris = ";".join(sorted(str(t) for t in combo))
            types_labels = ";".join(sorted(get_label(g, t) for t in combo))
            w.writerow([i, cnt, len(combo), types_uris, types_labels])

    # Write per-class instance counts
    with open(args.out_classes, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_uri", "class_label", "count"])
        for cls, cnt in sorted(class_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
            w.writerow([str(cls), get_label(g, cls), cnt])

    # Write predicate triple counts
    if args.out_preds:
        with open(args.out_preds, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["predicate_uri", "predicate_label", "triple_count"])
            for p, cnt in sorted(pred_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
                w.writerow([str(p), lname(p), cnt])

    # Optional console summary
    if args.print_summary:
        print("=== KG Quantification Summary ===")
        print(f"Total triples: {triple_count}")
        print(f"Typed subjects: {len(subject_types)}")
        print(f"Distinct rdf:type classes: {len(class_counts)}")
        print(f"Distinct type combinations: {len(combo_counts)}")
        top5_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
        print("Top classes:")
        for cls, cnt in top5_classes:
            print(f"  - {get_label(g, cls)} ({cls}): {cnt}")
        top5_preds = sorted(pred_counts.items(), key=lambda x: -x[1])[:5]
        print("Top predicates:")
        for p, cnt in top5_preds:
            print(f"  - {p} → {cnt}")

if __name__ == "__main__":
    main()
