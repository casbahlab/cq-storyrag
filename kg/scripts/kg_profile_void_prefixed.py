#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_profile_void_prefixed.py

Profile an RDF KG and produce:
- CSVs: per-class counts, predicate counts (excluding rdf:type), type-combination counts, co-typing matrix
- Figures: top classes, top predicates, top type combinations, co-typing heatmap
- VoID TTL: dataset description (triples, classes, properties, partitions)
- Optional coverage CSV for selected classes/properties

Key change: **Predicates are rendered with prefixes (CURIEs) where possible**
in CSVs and plots (labels). We also exclude `rdf:type` from predicate stats.

Usage:
python kg_profile_void_prefixed.py \
  --in data/liveaid_instances_master.ttl \
  --out-dir out/profile \
  --dataset-uri http://wembrewind.live/ex/dataset/liveaid \
  --top-classes 15 --top-preds 15 --top-combos 12

Multiple inputs: repeat --in
Coverage table (optional): add one or more --coverage-class and --coverage-prop
"""

import argparse
import csv
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, Set, Tuple, List

from rdflib import Graph, RDF, RDFS, Namespace, URIRef, BNode, Literal, XSD

# Plotting (matplotlib only; no seaborn, no explicit colors)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Namespaces
RDF_NS = Namespace(str(RDF))
RDFS_NS = Namespace(str(RDFS))
SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
VOID = Namespace("http://rdfs.org/ns/void#")
DCT = Namespace("http://purl.org/dc/terms/")
XSD_NS = Namespace(str(XSD))
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
CORE = Namespace("https://w3id.org/polifonia/ontology/core/")
EX = Namespace("http://wembrewind.live/ex#")

def bind_common_prefixes(g: Graph):
    """Bind common prefixes so CURIE rendering works nicely."""
    g.bind("rdf", RDF_NS)
    g.bind("rdfs", RDFS_NS)
    g.bind("xsd", XSD_NS)
    g.bind("schema", SCHEMA)
    g.bind("schema_https", SCHEMA_HTTPS)
    g.bind("void", VOID)
    g.bind("dct", DCT)
    g.bind("mm", MM)
    g.bind("core", CORE)
    g.bind("ex", EX)

def lname(u: URIRef) -> str:
    s = str(u)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def curie(g: Graph, term) -> str:
    """Return a prefixed name if possible, else a clean URI string."""
    try:
        if isinstance(term, (URIRef,)):
            s = g.namespace_manager.normalizeUri(term)
            # normalizeUri returns "<...>" when no prefix bound
            if s.startswith("<") and s.endswith(">"):
                return str(term)
            return s
        return str(term)
    except Exception:
        return str(term)

def get_label(g: Graph, node: URIRef) -> str:
    for p in (SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label):
        val = g.value(node, p)
        if isinstance(val, Literal):
            return str(val)
    # Prefer CURIE over lname for display if available
    try:
        q = curie(g, node)
        if ":" in q:
            return q
    except Exception:
        pass
    return lname(node)

def parse_any(g: Graph, path: str):
    # Try Turtle first, fall back to rdflib auto format
    try:
        g.parse(path, format="turtle")
    except Exception:
        g.parse(path)

def quant_compute(g: Graph):
    triple_count = len(g)

    # Predicate counts excluding rdf:type
    pred_counts = Counter()
    for _, p, _ in g:
        if p == RDF.type:
            continue
        pred_counts[p] += 1

    # Class counts and subject type sets
    class_counts = Counter()
    subject_types: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef):
            subject_types[s].add(o)
            class_counts[o] += 1

    # Type combinations per subject (set of classes)
    combo_counts = Counter()
    for s, types in subject_types.items():
        if types:
            combo_counts[frozenset(types)] += 1

    # Distincts for VoID
    distinct_subjects = set(s for s, _, _ in g)
    distinct_objects = set(o for _, _, o in g)
    distinct_predicates = set(p for _, p, _ in g)

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
        w.writerow(["predicate_uri", "predicate_curie", "predicate_label", "triple_count"])
        for p, cnt in sorted(pred_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            w.writerow([str(p), curie(g, p), get_label(g, p), cnt])
    paths["predicates"] = str(preds_csv)

    # Classes
    classes_csv = out_dir / f"{prefix}_classes.csv"
    with classes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_uri", "class_curie", "class_label", "count"])
        for cls, cnt in sorted(class_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
            w.writerow([str(cls), curie(g, cls), get_label(g, cls), cnt])
    paths["classes"] = str(classes_csv)

    # Type combinations
    combos_csv = out_dir / f"{prefix}_combos.csv"
    with combos_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo_index", "count", "n_types", "types_uris", "types_curies", "types_labels"])
        for i, (combo, cnt) in enumerate(sorted(
            combo_counts.items(),
            key=lambda x: (-x[1], sorted(lname(t) for t in x[0]))
        )):
            types_sorted = sorted(combo, key=lambda t: get_label(g, t).lower())
            types_uris = ";".join(str(t) for t in types_sorted)
            types_curies = ";".join(curie(g, t) for t in types_sorted)
            types_labels = ";".join(get_label(g, t) for t in types_sorted)
            w.writerow([i, cnt, len(combo), types_uris, types_curies, types_labels])
    paths["combos"] = str(combos_csv)

    # Co-typing matrix (pairwise class co-occurrence across subjects)
    all_classes = sorted(class_counts.keys(), key=lambda c: get_label(g, c).lower())
    idx = {c: i for i, c in enumerate(all_classes)}
    size = len(all_classes)
    matrix = [[0]*size for _ in range(size)]
    for s, types in subject_types.items():
        tlist = list(types)
        for a, b in combinations(tlist, 2):
            i, j = idx[a], idx[b]
            matrix[i][j] += 1
            matrix[j][i] += 1
        for t in tlist:
            matrix[idx[t]][idx[t]] += 1

    cot_csv = out_dir / f"{prefix}_co_typing_matrix.csv"
    with cot_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["class_uri", "class_curie"] + [str(c) for c in all_classes]
        w.writerow(header)
        for c in all_classes:
            row = [str(c), curie(g, c)] + matrix[idx[c]]
            w.writerow(row)
    paths["cot_matrix"] = str(cot_csv)

    return paths, all_classes, matrix

def plot_barh(labels, values, title, xlabel, ylabel, png_path: Path, pdf_path: Path):
    plt.figure()
    y = range(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

def plot_figures(out_dir: Path, g: Graph,
                 class_counts: Counter, pred_counts: Counter, combo_counts: Counter,
                 all_classes: List[URIRef], matrix,
                 top_classes=15, top_preds=15, top_combos=12):

    # Top classes
    classes_sorted = sorted(class_counts.items(), key=lambda x: (-x[1], get_label(g, x[0]).lower()))[:top_classes]
    labels = [get_label(g, c) for c, _ in classes_sorted]
    values = [cnt for _, cnt in classes_sorted]
    plot_barh(labels, values, f"Top {top_classes} Classes by Instance Count",
              "Instances", "Class", out_dir / "fig_top_classes.png", out_dir / "fig_top_classes.pdf")

    # Top predicates (labels prefer CURIEs)
    preds_sorted = sorted(pred_counts.items(), key=lambda x: (-x[1], get_label(g, x[0]).lower()))[:top_preds]
    labels = [curie(g, p) for p, _ in preds_sorted]
    values = [cnt for _, cnt in preds_sorted]
    plot_barh(labels, values, f"Top {top_preds} Predicates by Triple Count (excluding rdf:type)",
              "Triples", "Predicate", out_dir / "fig_top_predicates.png", out_dir / "fig_top_predicates.pdf")

    # Top type combinations (labels: semicolon-joined CURIEs)
    combos_sorted = sorted(combo_counts.items(), key=lambda x: (-x[1], len(x[0])))[:top_combos]
    labels_k, vals_k = [], []
    for combo, cnt in combos_sorted:
        curies = sorted(curie(g, t) for t in combo)
        lab = "; ".join(curies)
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
    bind_common_prefixes(vg)

    dataset = URIRef(dataset_uri) if dataset_uri else BNode()
    vg.add((dataset, RDF.type, VOID.Dataset))

    # Core VoID stats
    vg.add((dataset, VOID.triples, Literal(triple_count, datatype=XSD.integer)))
    vg.add((dataset, VOID.distinctSubjects, Literal(len(distinct_subjects), datatype=XSD.integer)))
    vg.add((dataset, VOID.distinctObjects, Literal(len(distinct_objects), datatype=XSD.integer)))
    vg.add((dataset, VOID.properties, Literal(len(distinct_predicates), datatype=XSD.integer)))
    vg.add((dataset, VOID.classes, Literal(len(class_counts), datatype=XSD.integer)))

    # Class partitions
    for cls, cnt in class_counts.items():
        b = BNode()
        vg.add((dataset, VOID.classPartition, b))
        vg.add((b, VOID["class"], cls))
        vg.add((b, VOID.triples, Literal(cnt, datatype=XSD.integer)))

    # Property partitions (excluding rdf:type, since pred_counts already excludes it)
    for p, cnt in pred_counts.items():
        b = BNode()
        vg.add((dataset, VOID.propertyPartition, b))
        vg.add((b, VOID.property, p))
        vg.add((b, VOID.triples, Literal(cnt, datatype=XSD.integer)))

    out_path = out_dir / "void.ttl"
    vg.serialize(destination=str(out_path), format="turtle")
    return str(out_path)

def compute_coverage(g: Graph,
                     classes: Iterable[str],
                     props: Iterable[str],
                     out_csv: Path):
    classes_uris = [URIRef(c) for c in classes]
    props_uris = [URIRef(p) for p in props]

    # Collect instances per class
    instances_by_class: Dict[URIRef, Set[URIRef]] = {c: set(g.subjects(RDF.type, c)) for c in classes_uris}

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["class_uri", "class_curie", "class_label", "instances"] + [str(p) for p in props]
        w.writerow(header)
        for cls in classes_uris:
            insts = instances_by_class[cls]
            row = [str(cls), curie(g, cls), get_label(g, cls), len(insts)]
            for p in props_uris:
                covered = sum(1 for s in insts if any(True for _ in g.triples((s, p, None))))
                pct = 0.0 if len(insts) == 0 else (100.0 * covered / len(insts))
                row.append(f"{pct:.1f}")
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="Profile an RDF KG with CSVs, figures, and a VoID TTL (predicates with prefixes).")
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Input RDF file(s). Repeatable.")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--dataset-uri", default="http://example.org/dataset/unknown", help="URI to use as the VoID dataset IRI.")
    ap.add_argument("--top-classes", type=int, default=15)
    ap.add_argument("--top-preds", type=int, default=15)
    ap.add_argument("--top-combos", type=int, default=12)
    ap.add_argument("--coverage-class", action="append", default=None, help="IRI of a class to include in coverage table. Repeatable.")
    ap.add_argument("--coverage-prop", action="append", default=None, help="IRI of a property to compute coverage for. Repeatable.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Build combined graph
    g = Graph()
    bind_common_prefixes(g)
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
    print(f"Distinct classes: {len(class_counts)}  |  Distinct predicates (no rdf:type): {len(pred_counts)}")
    print(f"Typed subjects: {len(subject_types)}   |  Type combinations: {len(combo_counts)}")
    print(f"Tables: {paths}")
    print(f"VoID TTL: {void_path}")
    print(f"Figures: fig_top_classes.*, fig_top_predicates.*, fig_top_combos.*, fig_cotyping_heatmap.*")

if __name__ == "__main__":
    main()
