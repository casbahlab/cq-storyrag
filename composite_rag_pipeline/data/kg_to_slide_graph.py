#!/usr/bin/env python3
"""
kg_to_slide_graph.py
--------------------
Generate clean, slide-friendly diagrams from a Knowledge Graph.

Features:
- Full KG view or filtered subgraphs
- Schema outline mode (class-to-class edges only)
- Split output by connected components or by predicate
- Group predicates into buckets
- Bundle outputs into a single multi-page PDF
- Deduplicate edges, exclude literals, cap edges per predicate
- Unicode font detection for CJK and symbols
- Auto-import RDF prefixes, and configurable label mode (qname/local)
- Spacious layout (larger canvas and more relaxed spring layout)
"""

import argparse
import collections
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx

try:
    import rdflib
except ImportError:
    rdflib = None


# ----------------- Fonts -----------------
def _pick_unicode_font():
    candidates = [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
        "Arial Unicode MS", "DejaVu Sans"
    ]
    try:
        available = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        available = set()
    for name in candidates:
        if name in available:
            matplotlib.rcParams['font.family'] = name
            return name
    return matplotlib.rcParams.get('font.family', 'default')

chosen_font = _pick_unicode_font()


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, help="Path to RDF file (ttl, jsonld, nt, n3, xml).")
    p.add_argument("--format", type=str, default=None, help="RDF format for rdflib.")
    p.add_argument("--csv", type=str, help="CSV with subject,predicate,object columns")
    p.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    p.add_argument("--include-preds", type=str, default="", help="Predicates to include (comma-separated)")
    p.add_argument("--exclude-preds", type=str, default="", help="Predicates to exclude (comma-separated)")
    p.add_argument("--seeds", type=str, default="", help="Seed nodes to keep (comma-separated)")
    p.add_argument("--radius", type=int, default=1, help="Radius around seeds (0=only seeds)")
    p.add_argument("--prefix", action="append", default=[], help="Prefix mapping prefix=IRI (optional, auto-imports from RDF too)")
    p.add_argument("--output", type=str, required=True, help="Output path prefix (no extension)")
    p.add_argument("--min_degree", type=int, default=0, help="Drop nodes with degree < N")
    p.add_argument("--max_nodes", type=int, default=0, help="Hard cap on number of nodes (0 = no cap)")

    # Outline
    p.add_argument("--schema-outline", dest="schema_outline", action="store_true",
                   help="Collapse to class-to-class outline using rdf:type assertions.")

    # Edge controls
    p.add_argument("--dedupe-parallel", action="store_true", help="Remove duplicate edges with same (s,p,o)")
    p.add_argument("--unique-predicates", action="store_true", help="Keep only one example per predicate (after grouping)")
    p.add_argument("--max-per-predicate", type=int, default=0, help="Cap edges per predicate")
    p.add_argument("--exclude-literals", action="store_true", help="Drop edges with literal objects")

    # Split & grouping
    p.add_argument("--no_dot", action="store_true", help="Do not emit DOT file")
    p.add_argument("--split", type=str, choices=["components", "predicate"], default=None,
                   help="Split output: one file per connected component or per predicate")
    p.add_argument("--topk-components", type=int, default=0, help="When splitting by components, keep top-K largest")
    p.add_argument("--min-component-nodes", type=int, default=0, help="Drop component if nodes < N")
    p.add_argument("--min-component-edges", type=int, default=0, help="Drop component if edges < N")
    p.add_argument("--topk-predicates", type=int, default=0, help="When splitting by predicate, keep top-K")
    p.add_argument("--min-predicate-edges", type=int, default=0, help="Drop predicate slice if edges < N")
    p.add_argument("--bundle-pdf", type=str, default="", help="Path to a multi-page PDF bundling all figures")
    p.add_argument("--group-pred", type=str, default="",
                   help="Grouping: GroupA=pred1,pred2;GroupB=pred3 (after labelization)")
    p.add_argument("--label-mode", type=str, choices=["qname", "local"], default="qname",
                   help="How to label IRIs: qname (prefix:Local) or local. Default: qname.")
    return p.parse_args()


# ----------------- Prefixes & labeling -----------------
def parse_prefixes(pairs: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in pairs:
        if "=" in item:
            k, v = item.split("=", 1)
            mapping[k.strip()] = v.strip()
    return mapping


def prefixes_from_graph(g) -> Dict[str,str]:
    mapping = {}
    for pref, ns in g.namespaces():
        if str(ns).startswith("http://schema.org"):
            mapping["schema"] = str(ns)   # force schema
        else:
            mapping[str(pref)] = str(ns)
    return mapping



def qname(s: str, prefixes: Dict[str, str], label_mode: str = "qname") -> str:
    if label_mode == "qname":
        for pref, base in prefixes.items():
            if s.startswith(base):
                return f"{pref}:{s[len(base):]}"
    # fallback: local name
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rstrip("/").rsplit("/", 1)[-1]
    return s


# ----------------- IO -----------------
def read_rdf_edges(path: Path, fmt: Optional[str], prefixes: Dict[str, str]):
    if rdflib is None:
        raise RuntimeError("rdflib not installed. pip install rdflib")

    g = rdflib.Graph()
    g.parse(path.as_posix(), format=fmt or None)

    # Merge namespaces from RDF file into provided prefixes (do not overwrite CLI-provided)
    graph_prefixes = prefixes_from_graph(g)
    for k, v in graph_prefixes.items():
        prefixes.setdefault(k, v)

    triples = []
    for s, p, o in g:
        s_str, p_str = str(s), str(p)
        if isinstance(o, rdflib.term.Literal):
            o_str, is_lit = str(o), True
        else:
            o_str, is_lit = str(o), False
        kind = "rdf_type" if p_str.endswith("#type") or p_str.endswith("/type") else "edge"
        triples.append((s_str, p_str, o_str, is_lit, kind))
    return triples


def read_csv_edges(path: Path, delimiter: str):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=delimiter)
        for row in r:
            s = (row.get("subject") or "").strip()
            p = (row.get("predicate") or "").strip()
            o = (row.get("object") or "").strip()
            if s and p and o:
                is_lit = not (o.startswith("http://") or o.startswith("https://"))
                rows.append((s, p, o, is_lit, "edge"))
    return rows


# ----------------- Filters & transforms -----------------
def parse_group_spec(spec: str) -> dict:
    groups = {}
    if not spec:
        return groups
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, preds = chunk.split("=", 1)
        for p in [x.strip() for x in preds.split(",") if x.strip()]:
            groups[p] = name.strip()
    return groups


def relabel_predicates_by_group(edges, groups: dict):
    if not groups:
        return edges
    out = []
    for s, p, o, is_lit, kind in edges:
        out.append((s, groups.get(p, p), o, is_lit, kind))
    return out


def filter_edges(edges, include_preds: Set[str], exclude_preds: Set[str], prefixes: Dict[str, str], label_mode: str):
    # include/exclude after labelization; sets should include either qnames or local names
    include_resolved = set(include_preds)
    exclude_resolved = set(exclude_preds)
    if not include_resolved and not exclude_resolved:
        return edges
    keep = []
    for s, p, o, is_lit, kind in edges:
        p_norm = p  # already labelized
        if include_resolved and p_norm not in include_resolved:
            continue
        if exclude_resolved and p_norm in exclude_resolved:
            continue
        keep.append((s, p, o, is_lit, kind))
    return keep


def dedupe_parallel_edges(edges):
    seen = set()
    out = []
    for e in edges:
        key = (e[0], e[1], e[2])  # (s,p,o)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def cap_edges_per_predicate(edges, max_per_pred: int, unique: bool):
    if max_per_pred <= 0 and not unique:
        return edges
    cap = 1 if unique else max_per_pred
    grouped = collections.defaultdict(list)
    for e in edges:
        grouped[e[1]].append(e)  # by predicate label
    out = []
    for pred, es in grouped.items():
        # Prefer edges connecting high-degree nodes (heuristic)
        deg = collections.Counter()
        for s, p, o, is_lit, kind in es:
            deg[s] += 1; deg[o] += 1
        es_sorted = sorted(es, key=lambda e: deg[e[0]] + deg[e[2]], reverse=True)
        out.extend(es_sorted[:cap])
    return out


def build_schema_outline(edges, prefixes, label_mode="qname"):
    """
    Collapse instance graph into class-to-class outline using rdf:type assertions.
    Returns edges as tuples on class labels with is_lit=False and kind='edge'.
    """
    type_map = {}
    for s, p, o, is_lit, kind in edges:
        if kind == "rdf_type" and not is_lit:
            type_map.setdefault(s, o)  # first wins

    agg = set()
    out = []
    for s, p, o, is_lit, kind in edges:
        if kind != "edge" or is_lit:
            continue
        s_cls = qname(type_map.get(s, "Thing"), prefixes, label_mode)
        o_cls = qname(type_map.get(o, "Thing"), prefixes, label_mode)
        key = (s_cls, p, o_cls)
        if key in agg:
            continue
        agg.add(key)
        out.append((s_cls, p, o_cls, False, "edge"))
    return out


# ----------------- Graph build & draw -----------------
def build_graph(edges, drop_literals=False):
    G = nx.DiGraph()
    for s, p, o, is_lit, kind in edges:
        if drop_literals and is_lit:
            continue
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, label=p)
    return G


def draw_graph(G, out_prefix):
    Path(os.path.dirname(out_prefix) or ".").mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=200)
    plt.figure(figsize=(28, 28), dpi=100)
    nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    plt.axis("off")
    plt.savefig(out_prefix + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(out_prefix + ".svg", bbox_inches="tight")
    plt.close()


def export_dot(G, out_prefix):
    import html
    lines = ["digraph KG {"]
    for u, v, data in G.edges(data=True):
        lab = data.get("label", "")
        lines.append(f'  "{html.escape(u)}" -> "{html.escape(v)}" [label="{html.escape(lab)}"];')
    lines.append("}")
    Path(out_prefix + ".dot").write_text("\n".join(lines), encoding="utf-8")


def save_graph(G, out_prefix, no_dot, pdf: Optional[PdfPages] = None):
    if not no_dot:
        export_dot(G, out_prefix)
    draw_graph(G, out_prefix)
    if pdf is not None:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=200)
        fig = plt.figure(figsize=(28, 28))
        nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ----------------- Main -----------------
def main():
    args = parse_args()
    prefixes = parse_prefixes(args.prefix)

    # Load edges (raw)
    if args.csv:
        raw_edges = read_csv_edges(Path(args.csv), args.delimiter)
    else:
        raw_edges = read_rdf_edges(Path(args.input), args.format, prefixes)

    # Labelize (apply qname/local)
    edges = []
    for s, p, o, is_lit, kind in raw_edges:
        s_lab = qname(s, prefixes, args.label_mode)
        p_lab = qname(p, prefixes, args.label_mode)
        if not is_lit and (o.startswith("http://") or o.startswith("https://")):
            o_lab = qname(o, prefixes, args.label_mode)
        else:
            o_lab = o
        edges.append((s_lab, p_lab, o_lab, is_lit, kind))

    # Group predicates (optional)
    groups = parse_group_spec(args.group_pred)
    edges = relabel_predicates_by_group(edges, groups)

    # Outline collapse (optional)
    if args.schema_outline:
        edges = build_schema_outline(edges, prefixes, args.label_mode)

    # Includes/Excludes (after labelization)
    inc = {s.strip() for s in args.include_preds.split(",") if s.strip()}
    exc = {s.strip() for s in args.exclude_preds.split(",") if s.strip()}
    if inc or exc:
        edges = filter_edges(edges, inc, exc, prefixes, args.label_mode)

    # Seed-based subgraph (optional)
    seeds = {s.strip() for s in args.seeds.split(",") if s.strip()}
    if seeds:
        G_seed = nx.DiGraph()
        for s, p, o, is_lit, kind in edges:
            G_seed.add_edge(s, o, label=p)
        keep_nodes = set()
        UG = G_seed.to_undirected()
        for seed in seeds:
            if seed in UG:
                keep_nodes.add(seed)
                lengths = nx.single_source_shortest_path_length(UG, seed, cutoff=args.radius)
                keep_nodes |= set(lengths.keys())
        edges = [(s, p, o, is_lit, kind) for s, p, o, is_lit, kind in edges if s in keep_nodes and o in keep_nodes]

    # Dedupe parallel edges
    if args.dedupe_parallel:
        edges = dedupe_parallel_edges(edges)

    # Cap edges per predicate / unique per predicate
    edges = cap_edges_per_predicate(edges, args.max_per_predicate, args.unique_predicates)

    # Build graph
    G = build_graph(edges, drop_literals=args.exclude_literals)

    # Post-filter by degree and node cap
    if args.min_degree > 0:
        to_drop = [n for n in list(G.nodes) if G.degree(n) < args.min_degree]
        G.remove_nodes_from(to_drop)
    if args.max_nodes and G.number_of_nodes() > args.max_nodes:
        deg_sorted = sorted(G.degree, key=lambda kv: kv[1], reverse=True)
        keep = set([n for n, _d in deg_sorted[:args.max_nodes]])
        drop = [n for n in G.nodes if n not in keep]
        G.remove_nodes_from(drop)

    # Output
    pdf = PdfPages(args.bundle_pdf) if args.bundle_pdf else None

    if args.split == "predicate":
        grouped = collections.defaultdict(list)
        for s, p, o, is_lit, kind in edges:
            grouped[p].append((s, p, o, is_lit, kind))
        pred_list = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
        if args.topk_predicates:
            pred_list = pred_list[:args.topk_predicates]
        for pred, es in pred_list:
            if args.min_predicate_edges and len(es) < args.min_predicate_edges:
                continue
            sub = build_graph(es, args.exclude_literals)
            safe_pred = "".join(c if c.isalnum() or c in "-_." else "_" for c in pred)[:40] or "unlabeled"
            save_graph(sub, f"{args.output}__pred_{safe_pred}", args.no_dot, pdf)

    elif args.split == "components":
        comps = sorted(nx.connected_components(G.to_undirected()), key=len, reverse=True)
        if args.topk_components:
            comps = comps[:args.topk_components]
        for i, nodes in enumerate(comps, 1):
            sub = G.subgraph(nodes).copy()
            if args.min_component_nodes and sub.number_of_nodes() < args.min_component_nodes:
                continue
            if args.min_component_edges and sub.number_of_edges() < args.min_component_edges:
                continue
            save_graph(sub, f"{args.output}__comp{i:02d}", args.no_dot, pdf)

    else:
        save_graph(G, args.output, args.no_dot, pdf)

    if pdf:
        pdf.close()

    print(f"[font] {chosen_font}  [nodes]={G.number_of_nodes()} [edges]={G.number_of_edges()}")


if __name__ == "__main__":
    main()
