
#!/usr/bin/env python3
"""
KG Profiler + VoID generator

Features
- Ingest 1..N RDF files into an rdflib.Graph
- Predicate stats (EXCLUDING rdf:type)
- Class stats (USING rdf:type)
- Class–predicate combos (subject's rdf:type × predicate, excluding rdf:type)
- Co-typing matrix (how often two classes co-occur on the same subject)
- CSV outputs + simple figures (PNG + PDF)
- VoID description (void.ttl)
- CURIE labels for predicates and classes (e.g., schema:name), with schema1→schema normalization

Usage
------
python kg_profile_void.py \
  --in data/liveaid_instances_master.ttl \
  --out-dir out/profile \
  --dataset-uri http://wembrewind.live/ex/dataset/liveaid \
  --top-classes 20 --top-preds 25 --top-combos 20 \
  --coverage-class https://w3id.org/polifonia/ontology/music-meta/LivePerformance \
  --coverage-prop  http://schema.org/name \
  --coverage-prop  http://schema.org/performer
"""

import argparse
import csv
import itertools
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

from rdflib import Graph, RDF, RDFS, Namespace, URIRef, BNode, Literal, XSD
from rdflib.namespace import NamespaceManager, DCTERMS

# ---- Namespaces ----
SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
CORE = Namespace("https://w3id.org/polifonia/ontology/core/")
EX = Namespace("http://wembrewind.live/ex#")
VOID = Namespace("http://rdfs.org/ns/void#")


def bind_common_prefixes(g: Graph) -> None:
    """
    Overwrite the graph's namespace manager so we control displayed CURIEs.
    Ensures schema1 won't appear; it will be normalized to 'schema' where possible.
    """
    nm = NamespaceManager(Graph())
    nm.bind("rdf", RDF)
    nm.bind("rdfs", RDFS)
    nm.bind("xsd", XSD)
    nm.bind("schema", SCHEMA)  # prefer plain 'schema:'
    nm.bind("mm", MM)
    nm.bind("core", CORE)
    nm.bind("ex", EX)
    nm.bind("void", VOID)
    nm.bind("dcterms", DCTERMS)
    g.namespace_manager = nm


def curie(g: Graph, term) -> str:
    """
    Return a CURIE like 'schema:name' for a URIRef when possible; otherwise string(term).
    Normalizes any accidental 'schema1' (or similar) to 'schema' when namespace equals SCHEMA.
    """
    if isinstance(term, URIRef):
        nm = g.namespace_manager
        try:
            prefix, ns, name = nm.compute_qname(term, generate=False)
            # Normalize schema-like aliases to 'schema'
            if str(ns) == str(SCHEMA):
                prefix = "schema"
            return f"{prefix}:{name}"
        except Exception:
            return str(term)
    return str(term)


def get_label(g: Graph, term) -> str:
    """
    Human-ish label for a URI. Prefer rdfs:label, else CURIE, else fragment/localname, else str(term).
    """
    if isinstance(term, (URIRef, BNode)):
        # Try rdfs:label first
        lbl = None
        for _, _, o in g.triples((term, RDFS.label, None)):
            if isinstance(o, Literal):
                lbl = str(o)
                break
        if lbl:
            return lbl
        # Fallback to CURIE
        if isinstance(term, URIRef):
            return curie(g, term)
    return str(term)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile a KG and emit VoID + CSV + figures.")
    p.add_argument("--in", dest="inputs", action="append", required=True,
                   help="Input RDF file(s). Repeat flag for multiple files.")
    p.add_argument("--out-dir", required=True, help="Output directory for CSVs/figures/void.ttl")
    p.add_argument("--dataset-uri", required=True, help="URI for the VoID dataset node")
    p.add_argument("--top-classes", type=int, default=20)
    p.add_argument("--top-preds", type=int, default=25)
    p.add_argument("--top-combos", type=int, default=20)
    p.add_argument("--no-pdf", action="store_true", help="Don't save PDF copies of figures (PNG always saved).")
    p.add_argument("--coverage-class", action="append", default=[],
                   help="IRI of a class whose instances form the rows of a coverage table.")
    p.add_argument("--coverage-prop", action="append", default=[],
                   help="IRI of a property to check presence for in the coverage table (columns). Repeatable.")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_into_graph(files: List[str]) -> Graph:
    g = Graph()
    for f in files:
        # Try Turtle first, then let rdflib sniff
        try:
            g.parse(f, format="turtle")
        except Exception:
            g.parse(f)
    bind_common_prefixes(g)
    return g


def count_predicates(g: Graph) -> Counter:
    """
    Count predicates EXCLUDING rdf:type.
    """
    c = Counter()
    for _, p, _ in g:
        if p == RDF.type:
            continue
        c[p] += 1
    return c


def distinct_predicates(g: Graph) -> set:
    return {p for _, p, _ in g if p != RDF.type}


def count_classes(g: Graph) -> Counter:
    """
    Count objects of rdf:type.
    """
    c = Counter()
    for _, _, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef):
            c[o] += 1
    return c


def subject_classes_map(g: Graph) -> Dict[URIRef, set]:
    sc = defaultdict(set)
    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(s, (URIRef, BNode)) and isinstance(o, URIRef):
            sc[s].add(o)
    return sc


def count_class_predicate_combos(g: Graph, sc: Dict[URIRef, set]) -> Counter:
    """
    Count (class, predicate) where class is an rdf:type of the subject.
    Excludes rdf:type predicate.
    """
    c = Counter()
    for s, p, _ in g:
        if p == RDF.type:
            continue
        classes = sc.get(s, ())
        for cls in classes:
            c[(cls, p)] += 1
    return c


def cotyping_counts(sc: Dict[URIRef, set]) -> Counter:
    """
    Count co-typing pairs across subjects. Includes diagonal (cls, cls).
    """
    c = Counter()
    for classes in sc.values():
        clist = sorted(classes, key=str)
        for a, b in itertools.combinations_with_replacement(clist, 2):
            c[(a, b)] += 1
    return c


def write_csv_predicates(out_dir: str, g: Graph, pred_counts: Counter) -> None:
    path = Path(out_dir) / "kg_full_predicates.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predicate_uri", "predicate_curie", "predicate_label", "triple_count"])
        for p, cnt in sorted(pred_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            w.writerow([str(p), curie(g, p), get_label(g, p), cnt])


def write_csv_classes(out_dir: str, g: Graph, class_counts: Counter) -> None:
    path = Path(out_dir) / "kg_full_classes.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_uri", "class_curie", "class_label", "count"])
        for c, cnt in sorted(class_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            w.writerow([str(c), curie(g, c), get_label(g, c), cnt])


def write_csv_combos(out_dir: str, g: Graph, combos: Counter) -> None:
    path = Path(out_dir) / "kg_full_combos.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_uri", "class_curie", "class_label", "predicate_uri", "predicate_curie", "predicate_label", "count"])
        for (cls, p), cnt in sorted(combos.items(), key=lambda x: (-x[1], str(x[0]))):
            w.writerow([str(cls), curie(g, cls), get_label(g, cls), str(p), curie(g, p), get_label(g, p), cnt])


def write_csv_cotyping(out_dir: str, g: Graph, cot_counts: Counter) -> None:
    path = Path(out_dir) / "kg_full_co_typing_matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_a_uri", "class_a_curie", "class_a_label",
                    "class_b_uri", "class_b_curie", "class_b_label",
                    "count"])
        for (a, b), cnt in sorted(cot_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            w.writerow([str(a), curie(g, a), get_label(g, a),
                        str(b), curie(g, b), get_label(g, b), cnt])


def plot_top_bar(items: List[Tuple[str, int]], title: str, out_path_base: Path, save_pdf: bool) -> None:
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(10, max(3, 0.35 * len(labels))))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_path_base}.png", dpi=150)
    if save_pdf:
        plt.savefig(f"{out_path_base}.pdf")
    plt.close()


def plot_heatmap(matrix, labels: List[str], title: str, out_path_base: Path, save_pdf: bool) -> None:
    import numpy as np
    arr = np.array(matrix, dtype=float)

    plt.figure(figsize=(max(6, 0.5 * len(labels)), max(5, 0.5 * len(labels))))
    plt.imshow(arr, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{out_path_base}.png", dpi=150)
    if save_pdf:
        plt.savefig(f"{out_path_base}.pdf")
    plt.close()


def write_void(out_dir: str, dataset_uri: str, g: Graph,
               class_counts: Counter, distinct_preds: set) -> None:
    vg = Graph()
    bind_common_prefixes(vg)

    ds = URIRef(dataset_uri)
    vg.add((ds, RDF.type, VOID.Dataset))
    vg.add((ds, DCTERMS.title, Literal("KG Profiled Dataset")))
    vg.add((ds, VOID.triples, Literal(len(g))))
    vg.add((ds, VOID.classes, Literal(len(class_counts))))
    vg.add((ds, VOID.properties, Literal(len(distinct_preds))))
    # Optionally: add linksets, partitions etc.

    out_path = Path(out_dir) / "void.ttl"
    vg.serialize(destination=str(out_path), format="turtle")


def coverage_table(out_dir: str, g: Graph, target_class_iris: List[str], prop_iris: List[str]) -> None:
    if not target_class_iris or not prop_iris:
        return

    targets = [URIRef(u) for u in target_class_iris]
    props = [URIRef(u) for u in prop_iris]

    # Instances per class
    class_instances = defaultdict(list)
    for cls in targets:
        for s, _, _ in g.triples((None, RDF.type, cls)):
            class_instances[cls].append(s)

    # For each instance, check presence of each property (at least one triple)
    rows = []
    for cls, instances in class_instances.items():
        for s in instances:
            row = {"class_uri": str(cls), "class_curie": curie(g, cls), "subject": str(s), "subject_curie": curie(g, s) if isinstance(s, URIRef) else str(s)}
            for p in props:
                has = any(True for _ in g.triples((s, p, None)))
                row[curie(g, p)] = int(has)
            rows.append(row)

    # Write CSV
    out_path = Path(out_dir) / "coverage_table.csv"
    # Ensure consistent column order
    prop_cols = [curie(g, p) for p in props]
    fieldnames = ["class_uri", "class_curie", "subject", "subject_curie"] + prop_cols
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    g = parse_into_graph(args.inputs)

    # --- Stats
    pred_counts = count_predicates(g)                 # excludes rdf:type
    classes = count_classes(g)                        # counts rdf:type objects
    scmap = subject_classes_map(g)
    combos = count_class_predicate_combos(g, scmap)   # excludes rdf:type
    cot_counts = cotyping_counts(scmap)
    distinct_preds = distinct_predicates(g)           # excludes rdf:type

    # --- CSVs
    write_csv_predicates(args.out_dir, g, pred_counts)
    write_csv_classes(args.out_dir, g, classes)
    write_csv_combos(args.out_dir, g, combos)
    write_csv_cotyping(args.out_dir, g, cot_counts)

    # --- Figures (top-N)
    top_classes = sorted(classes.items(), key=lambda x: (-x[1], str(x[0])))[: args.top_classes]
    top_classes_plot = [(curie(g, c), cnt) for c, cnt in top_classes]
    plot_top_bar(top_classes_plot, "Top classes (rdf:type)", Path(args.out_dir) / "fig_top_classes", save_pdf=not args.no_pdf)

    top_preds = sorted(pred_counts.items(), key=lambda x: (-x[1], str(x[0])))[: args.top_preds]
    top_preds_plot = [(curie(g, p), cnt) for p, cnt in top_preds]
    plot_top_bar(top_preds_plot, "Top predicates (excluding rdf:type)", Path(args.out_dir) / "fig_top_predicates", save_pdf=not args.no_pdf)

    top_combos = sorted(combos.items(), key=lambda x: (-x[1], (str(x[0][0]), str(x[0][1]))))[: args.top_combos]
    top_combos_plot = [(f"{curie(g, cls)} | {curie(g, p)}", cnt) for (cls, p), cnt in top_combos]
    plot_top_bar(top_combos_plot, "Top class | predicate combos", Path(args.out_dir) / "fig_top_combos", save_pdf=not args.no_pdf)

    # --- Heatmap for co-typing (limit to most frequent classes for readability)
    heat_labels = [curie(g, c) for c, _ in top_classes]
    # Build matrix for these labels only
    label_to_uri = {curie(g, c): c for c, _ in top_classes}
    size = len(heat_labels)
    # Initialize zero matrix
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    # Fill from cot_counts
    # cot_counts keys are URIs; we map them to indices if in top set
    idx = {label_to_uri[uri_label]: i for i, uri_label in enumerate(heat_labels) for uri_label in [heat_labels[i]] if label_to_uri.get(heat_labels[i])}
    # Build reverse map: URIRef -> index, based on top classes
    uri_to_idx = {uri: i for i, (uri, _) in enumerate([(c, cnt) for c, cnt in top_classes])}
    for (a, b), cnt in cot_counts.items():
        ia = uri_to_idx.get(a)
        ib = uri_to_idx.get(b)
        if ia is None or ib is None:
            continue
        matrix[ia][ib] = matrix[ia][ib] + cnt
        if ia != ib:
            matrix[ib][ia] = matrix[ib][ia] + cnt

    if size >= 1:
        plot_heatmap(matrix, heat_labels, "Co-typing matrix (top classes)", Path(args.out_dir) / "fig_cotyping_heatmap", save_pdf=not args.no_pdf)

    # --- Coverage table (optional)
    coverage_table(args.out_dir, g, args.coverage_class, args.coverage_prop)

    # --- VoID
    write_void(args.out_dir, args.dataset_uri, g, classes, distinct_preds)

    print(f"[OK] Wrote CSVs, figures, and VoID to: {args.out_dir}")


if __name__ == "__main__":
    main()
