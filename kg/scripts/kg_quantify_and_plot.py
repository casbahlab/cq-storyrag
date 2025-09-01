
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kg_quantify_and_plot.py

Compute KG quantification tables and generate figures.

Outputs:
- CSV: per-class instance counts
- CSV: predicate triple counts
- CSV: rdf:type combination counts
- PNG/PDF: bar charts for top classes, top predicates, top type combinations
- Optional Markdown summary

Usage example:
python kg_quantify_and_plot.py \\
  --in /path/to/liveaid_instances_master.ttl \\
  --out-dir /path/to/out \\
  --top-classes 15 --top-preds 15 --top-combos 12 \\
  --write-markdown

You may repeat --in to merge multiple RDF files before quantification.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Set, Dict

from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal

# Plotting
import matplotlib.pyplot as plt

SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")

def lname(u: URIRef) -> str:
    s = str(u)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def get_label(g: Graph, node: URIRef) -> str:
    # Prefer schema:name or rdfs:label for readability, else local name
    for p in (SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label):
        val = g.value(node, p)
        if isinstance(val, Literal):
            return str(val)
    return lname(node)

def parse_any(g: Graph, path: str):
    # Try Turtle first, then let rdflib guess
    try:
        g.parse(path, format="turtle")
    except Exception:
        g.parse(path)

def quant_compute(g: Graph):
    # Totals
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

    return triple_count, pred_counts, class_counts, subject_types, combo_counts

def write_tables(out_dir: Path,
                 g: Graph,
                 pred_counts: Counter,
                 class_counts: Counter,
                 combo_counts: Counter,
                 prefix: str = "kg_full"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predicates
    preds_csv = out_dir / f"{prefix}_predicates.csv"
    with preds_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predicate_uri", "predicate_label", "triple_count"])
        for p, cnt in sorted(pred_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
            w.writerow([str(p), lname(p), cnt])

    # Classes
    classes_csv = out_dir / f"{prefix}_classes.csv"
    with classes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_uri", "class_label", "count"])
        for cls, cnt in sorted(class_counts.items(), key=lambda x: (-x[1], lname(x[0]))):
            w.writerow([str(cls), get_label(g, cls), cnt])

    # Type combinations
    combos_csv = out_dir / f"{prefix}_combos.csv"
    with combos_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo_index", "count", "n_types", "types_uris", "types_labels"])
        for i, (combo, cnt) in enumerate(
            sorted(combo_counts.items(),
                   key=lambda x: (-x[1], sorted(lname(t) for t in x[0])))
        ):
            types_uris = ";".join(sorted(str(t) for t in combo))
            types_labels = ";".join(sorted(get_label(g, t) for t in combo))
            w.writerow([i, cnt, len(combo), types_uris, types_labels])

    return str(classes_csv), str(preds_csv), str(combos_csv)

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
                 top_classes: int,
                 top_preds: int,
                 top_combos: int,
                 prefix: str = "kg_full"):
    import pandas as pd

    classes_csv = out_dir / f"{prefix}_classes.csv"
    preds_csv = out_dir / f"{prefix}_predicates.csv"
    combos_csv = out_dir / f"{prefix}_combos.csv"

    df_classes = pd.read_csv(classes_csv)
    df_preds = pd.read_csv(preds_csv)
    df_combos = pd.read_csv(combos_csv)

    # Top classes
    df_c = df_classes.sort_values("count", ascending=False).head(top_classes)
    plot_barh(
        labels=df_c["class_label"].tolist(),
        values=df_c["count"].tolist(),
        title=f"Top {top_classes} Classes by Instance Count",
        xlabel="Instance count",
        ylabel="Class",
        png_path=out_dir / "fig_top_classes.png",
        pdf_path=out_dir / "fig_top_classes.pdf",
    )

    # Top predicates
    df_p = df_preds.sort_values("triple_count", ascending=False).head(top_preds)
    plot_barh(
        labels=df_p["predicate_label"].tolist(),
        values=df_p["triple_count"].tolist(),
        title=f"Top {top_preds} Predicates by Triple Count",
        xlabel="Triple count",
        ylabel="Predicate",
        png_path=out_dir / "fig_top_predicates.png",
        pdf_path=out_dir / "fig_top_predicates.pdf",
    )

    # Top type combinations (labels collapsed)
    def combo_readable(s, max_len=60):
        lab = s.replace(";", " + ")
        return lab if len(lab) <= max_len else (lab[:max_len-3] + "...")

    df_combos["combo_label"] = df_combos["types_labels"].apply(combo_readable)
    df_k = df_combos.sort_values("count", ascending=False).head(top_combos)
    plot_barh(
        labels=df_k["combo_label"].tolist(),
        values=df_k["count"].tolist(),
        title=f"Top {top_combos} Type Combinations by Subject Count",
        xlabel="Subjects",
        ylabel="Type combination",
        png_path=out_dir / "fig_top_combos.png",
        pdf_path=out_dir / "fig_top_combos.pdf",
    )

def write_markdown(out_dir: Path,
                   triple_count: int,
                   subject_types_count: int,
                   class_count: int,
                   combo_count: int,
                   prefix: str = "kg_full"):
    md = out_dir / f"{prefix}_summary.md"
    text = (
        "# Knowledge Graph Quantification — Summary\n\n"
        f"**Total triples:** **{triple_count:,}**\n\n"
        f"**Typed subjects:** **{subject_types_count:,}**\n\n"
        f"**Distinct rdf:type classes:** **{class_count}**\n\n"
        f"**Distinct type combinations:** **{combo_count}**\n\n"
        "Figures: `fig_top_classes.(png|pdf)`, `fig_top_predicates.(png|pdf)`, `fig_top_combos.(png|pdf)`."
    )
    md.write_text(text, encoding="utf-8")
    return str(md)

def main():
    ap = argparse.ArgumentParser(description="Quantify a KG and generate tables + figures.")
    ap.add_argument("--in", dest="inputs", action="append", required=True,
                    help="Input RDF file(s). Repeatable.")
    ap.add_argument("--out-dir", required=True, help="Output directory for CSVs and figures.")
    ap.add_argument("--prefix", default="kg_full", help="Filename prefix for outputs.")
    ap.add_argument("--top-classes", type=int, default=15, help="Top-N classes to plot.")
    ap.add_argument("--top-preds", type=int, default=15, help="Top-N predicates to plot.")
    ap.add_argument("--top-combos", type=int, default=12, help="Top-N type combinations to plot.")
    ap.add_argument("--write-markdown", action="store_true", help="Also write a short Markdown summary.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build combined graph
    g = Graph()
    for f in args.inputs:
        parse_any(g, f)

    # Compute
    triple_count, pred_counts, class_counts, subject_types, combo_counts = quant_compute(g)

    # Tables
    classes_csv, preds_csv, combos_csv = write_tables(out_dir, g, pred_counts, class_counts, combo_counts, prefix=args.prefix)

    # Figures
    plot_figures(out_dir, args.top_classes, args.top_preds, args.top_combos, prefix=args.prefix)

    # Markdown summary (optional)
    if args.write_markdown:
        md = write_markdown(out_dir, triple_count, len(subject_types), len(class_counts), len(combo_counts), prefix=args.prefix)
        print(f"Wrote summary → {md}")

    # Console summary
    print("=== KG Quantification Summary ===")
    print(f"Total triples: {triple_count}")
    print(f"Typed subjects: {len(subject_types)}")
    print(f"Distinct rdf:type classes: {len(class_counts)}")
    print(f"Distinct type combinations: {len(combo_counts)}")
    print(f"Tables → {classes_csv}, {preds_csv}, {combos_csv}")
    print(f"Figures → {out_dir / 'fig_top_classes.png'}, {out_dir / 'fig_top_predicates.png'}, {out_dir / 'fig_top_combos.png'}")

if __name__ == "__main__":
    main()
