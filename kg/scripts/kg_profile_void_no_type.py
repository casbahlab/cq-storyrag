#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_profile_void.py

Profile an RDF KG and produce:
- CSVs: per-class counts, predicate counts, type-combination counts, co-typing matrix
- Figures: top classes, top predicates, top type combinations, co-typing heatmap
- VoID TTL: standards-based dataset description (triples, classes, properties, partitions)
- Optional coverage CSV for selected classes/properties

Usage:
python kg_profile_void.py \
  --in /path/to/liveaid_instances_master.ttl \
  --out-dir /path/to/out \
  --dataset-uri http://example.org/dataset/liveaid \
  --top-classes 15 --top-preds 15 --top-combos 12 \
  --coverage-class https://w3id.org/polifonia/ontology/music-meta/Musician \
  --coverage-class http://schema.org/MusicGroup \
  --coverage-prop http://schema.org/name \
  --coverage-prop http://www.w3.org/2002/07/owl#sameAs \
  --coverage-prop http://schema.org/sameAs
"""

import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, Set, Tuple

from rdflib import Graph, RDF, RDFS, Namespace, URIRef, BNode, Literal, XSD

# Plotting (matplotlib only; no seaborn/colors for portability)
import matplotlib.pyplot as plt

# Namespaces
SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
VOID = Namespace("http://rdfs.org/ns/void#")
DCT = Namespace("http://purl.org/dc/terms/")
XSD_NS = Namespace(str(XSD))

def lname(u: URIRef) -> str:
    s = str(u)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def get_label(g: Graph, node: URIRef) -> str:
    for p in (SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label):
        val = g.value(node, p)
        if isinstance(val, Literal):
            return str(val)
    return lname(node)

def parse_any(g: Graph, path: str):
    try:
        g.parse(path, format="turtle")
    except Exception:
        g.parse(path)

def quant_compute(g: Graph):
    triple_count = len(g)

    pred_counts = Counter()
    # Exclude rdf:type from predicate counts
    for _, p, _ in g:
        if p == RDF.type:
            continue
        pred_counts[p] += 1

    class_counts = Counter()
    subject_types: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for s, _, o in g.triples((None, RDF.type, None)):
        subject_types[s].add(o)
        class_counts[o] += 1

    combo_counts = Counter()
    for s, types in subject_types.items():
        combo_counts[frozenset(types)] += 1

    # Distincts for VoID
    distinct_subjects = set(s for s, _, _ in g)
    distinct_objects = set(o for _, _, o in g)
    distinct_predicates = set(p for _, p, _ in g if p != RDF.type)

    return (triple_count, pred_counts, class_counts, subject_types, combo_counts,
            distinct_subjects, distinct_objects, distinct_predicates)

def write_tables(out_dir: Path, g: Graph,
                 pred_counts: Counter, class_counts: Counter, combo_counts: Counter,
                 subject_types: Dict[URIRef, Set[URIRef]],
                 prefix="kg_full") -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Predicates
    preds_csv = out_dir / f"{prefix}_predicates.csv"
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predicate_uri", "predicate_label", "triple_count"])
        for p, cnt in sorted(pred_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
            w.writerow([str(p), lname(p), cnt])
    paths["predicates"] = str(preds_csv)

    # Classes
    classes_csv = out_dir / f"{prefix}_classes.csv"
    with classes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_uri", "class_label", "count"])
        for cls, cnt in sorted(class_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
            w.writerow([str(cls), get_label(g, cls), cnt])
    paths["classes"] = str(classes_csv)

    # Type combinations
    combos_csv = out_dir / f"{prefix}_combos.csv"
    with combos_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo_index", "count", "n_types", "types_uris", "types_labels"])
        for i, (combo, cnt) in enumerate(sorted(
            combo_counts.items(),
            key=lambda x: (-x[1], sorted(lname(t) for t in x[0]))
        )):
            types_uris = ";".join(sorted(str(t) for t in combo))
            types_labels = ";".join(sorted(get_label(g, t) for t in combo))
            w.writerow([i, cnt, len(combo), types_uris, types_labels])
    paths["combos"] = str(combos_csv)

    # Co-typing matrix (pairwise class co-occurrence across subjects)
    all_classes = sorted(class_counts.keys(), key=lambda c: get_label(g, c).lower())
    idx = {c: i for i, c in enumerate(all_classes)}
    # matrix[i][j] = co-occurrence count (s has both class i and class j)
    size = len(all_classes)
    matrix = [[0]*size for _ in range(size)]
    for s, types in subject_types.items():
        tlist = list(types)
        for a, b in combinations(tlist, 2):
            i, j = idx[a], idx[b]
            matrix[i][j] += 1
            matrix[j][i] += 1
        # diagonal as number of subjects with that class
        for t in tlist:
            matrix[idx[t]][idx[t]] += 1

    cot_csv = out_dir / f"{prefix}_co_typing_matrix.csv"
    with cot_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["class_uri"] + [str(c) for c in all_classes]
        w.writerow(header)
        for c in all_classes:
            row = [str(c)] + matrix[idx[c]]
            w.writerow(row)
    paths["cot_matrix"] = str(cot_csv)

    return paths, all_classes, matrix

def plot_barh(labels, values, title, xlabel, ylabel, png_path: Path, pdf_path: Path):
    plt.figure()
    plt.barh(labels[::-1], values[::-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

def plot_figures(out_dir: Path,
                 g: Graph,
                 class_counts: Counter,
                 pred_counts: Counter,
                 combo_counts: Counter,
                 all_classes: Iterable[URIRef],
                 matrix,
                 top_classes: int,
                 top_preds: int,
                 top_combos: int):
    # Top classes
    top_c = sorted(class_counts.items(), key=lambda x: -x[1])[:top_classes]
    labels_c = [get_label(g, c) for c, _ in top_c]
    vals_c = [cnt for _, cnt in top_c]
    plot_barh(labels_c, vals_c, f"Top {top_classes} Classes by Instance Count",
              "Instance count", "Class",
              out_dir / "fig_top_classes.png", out_dir / "fig_top_classes.pdf")

    # Top predicates
    top_p = sorted(pred_counts.items(), key=lambda x: -x[1])[:top_preds]
    labels_p = [lname(p) for p, _ in top_p]
    vals_p = [cnt for _, cnt in top_p]
    plot_barh(labels_p, vals_p, f"Top {top_preds} Predicates by Triple Count",
              "Triple count", "Predicate",
              out_dir / "fig_top_predicates.png", out_dir / "fig_top_predicates.pdf")

    # Top type combos
    top_k = sorted(combo_counts.items(), key=lambda x: -x[1])[:top_combos]
    labels_k = []
    vals_k = []
    for combo, cnt in top_k:
        lab = " + ".join(sorted(get_label(g, t) for t in combo))
        if len(lab) > 60:
            lab = lab[:57] + "..."
        labels_k.append(lab)
        vals_k.append(cnt)
    plot_barh(labels_k, vals_k, f"Top {top_combos} Type Combinations by Subject Count",
              "Subjects", "Type combination",
              out_dir / "fig_top_combos.png", out_dir / "fig_top_combos.pdf")

    # Co-typing heatmap
    import numpy as np
    arr = np.array(matrix)
    plt.figure()
    plt.imshow(arr, aspect="auto")  # default colormap, no explicit colors
    plt.xticks(range(len(all_classes)), [get_label(g, c) for c in all_classes], rotation=90)
    plt.yticks(range(len(all_classes)), [get_label(g, c) for c in all_classes])
    plt.title("Class Co-typing Heatmap (counts)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_cotyping_heatmap.png", dpi=300)
    plt.savefig(out_dir / "fig_cotyping_heatmap.pdf")
    plt.close()

def write_void(out_dir: Path,
               dataset_uri: str,
               triple_count: int,
               distinct_subjects: Set,
               distinct_objects: Set,
               distinct_predicates: Set,
               class_counts: Counter,
               pred_counts: Counter):
    vg = Graph()
    vg.bind("void", VOID)
    vg.bind("dct", DCT)

    dataset = URIRef(dataset_uri) if dataset_uri else BNode()
    vg.add((dataset, RDF.type, VOID.Dataset))

    # Core VoID stats
    vg.add((dataset, VOID.triples, Literal(triple_count, datatype=XSD.integer)))
    vg.add((dataset, VOID.entities, Literal(len([s for s in distinct_subjects if isinstance(s, URIRef)]), datatype=XSD.integer)))
    vg.add((dataset, VOID.classes, Literal(len(class_counts), datatype=XSD.integer)))
    vg.add((dataset, VOID.properties, Literal(len(distinct_predicates), datatype=XSD.integer)))
    vg.add((dataset, VOID.distinctSubjects, Literal(len(distinct_subjects), datatype=XSD.integer)))
    vg.add((dataset, VOID.distinctObjects, Literal(len(distinct_objects), datatype=XSD.integer)))

    # Class partitions
    for cls, cnt in class_counts.items():
        part = BNode()
        vg.add((dataset, VOID.classPartition, part))
        vg.add((part, VOID["class"], cls))
        vg.add((part, VOID.entities, Literal(cnt, datatype=XSD.integer)))

    # Property partitions
    for p, cnt in pred_counts.items():
        part = BNode()
        vg.add((dataset, VOID.propertyPartition, part))
        vg.add((part, VOID.property, p))
        vg.add((part, VOID.triples, Literal(cnt, datatype=XSD.integer)))

    path = out_dir / "void.ttl"
    vg.serialize(destination=str(path), format="turtle")
    return str(path)

def compute_coverage(g: Graph,
                     classes: Iterable[str],
                     props: Iterable[str],
                     out_csv: Path):
    # % of instances of each class that have each property present at least once
    classes_uris = [URIRef(c) for c in classes]
    props_uris = [URIRef(p) for p in props]

    # Collect instances per class
    instances_by_class: Dict[URIRef, Set[URIRef]] = {c: set(g.subjects(RDF.type, c)) for c in classes_uris}

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["class_uri", "class_label", "instances"] + [p for p in props]
        w.writerow(header)
        for cls in classes_uris:
            insts = instances_by_class[cls]
            row = [str(cls), get_label(g, cls), len(insts)]
            for p in props_uris:
                covered = sum(1 for s in insts if any(True for _ in g.triples((s, p, None))))
                pct = 0.0 if len(insts) == 0 else (100.0 * covered / len(insts))
                row.append(f"{pct:.1f}")
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Profile an RDF KG with CSVs, figures, and a VoID TTL.")
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Input RDF file(s). Repeatable.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--dataset-uri", default="http://example.org/dataset/unknown", help="URI to use as the VoID dataset IRI.")
    ap.add_argument("--top-classes", type=int, default=15)
    ap.add_argument("--top-preds", type=int, default=15)
    ap.add_argument("--top-combos", type=int, default=12)
    ap.add_argument("--coverage-class", action="append", default=[], help="IRI of a class to include in coverage table. Repeatable.")
    ap.add_argument("--coverage-prop", action="append", default=[], help="IRI of a property to compute coverage for. Repeatable.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Build combined graph
    g = Graph()
    for f in args.inputs:
        parse_any(g, f)

    # Compute stats
    (triple_count, pred_counts, class_counts, subject_types, combo_counts,
     distinct_subjects, distinct_objects, distinct_predicates) = quant_compute(g)

    # Tables + co-typing matrix
    paths, all_classes, matrix = write_tables(out_dir, g, pred_counts, class_counts, combo_counts, subject_types)

    # Figures
    plot_figures(out_dir, g, class_counts, pred_counts, combo_counts, all_classes, matrix,
                 args.top_classes, args.top_preds, args.top_combos)

    # VoID profile
    void_path = write_void(out_dir, args.dataset_uri, triple_count,
                           distinct_subjects, distinct_objects, distinct_predicates,
                           class_counts, pred_counts)

    # Optional coverage
    if args.coverage_class and args.coverage_prop:
        cov_csv = out_dir / "coverage_table.csv"
        compute_coverage(g, args.coverage_class, args.coverage_prop, cov_csv)

    # Console summary
    print("=== KG Profile Summary ===")
    print(f"Triples: {triple_count}")
    print(f"Distinct classes: {len(class_counts)}  |  Distinct predicates: {len(pred_counts)}")
    print(f"Typed subjects: {len(subject_types)}   |  Type combinations: {len(combo_counts)}")
    print(f"Tables: {paths}")
    print(f"VoID TTL: {void_path}")
    print(f"Figures: fig_top_classes.*, fig_top_predicates.*, fig_top_combos.*, fig_cotyping_heatmap.*")

if __name__ == "__main__":
    main()
