#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

SCHEMA_IRI = "http://schema.org/"
MM_IRI = "https://w3id.org/polifonia/ontology/music-meta/"

EX = Namespace("http://wembrewind.live/ex#")
SCHEMA = Namespace(SCHEMA_IRI)
SCHEMA1 = Namespace(SCHEMA_IRI)
MM = Namespace(MM_IRI)

ITEMLIST_ORDER_ASC = URIRef(SCHEMA_IRI + "ItemListOrderAscending")

def local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.rsplit("/", 1)[-1]

def count_itemlist_elements(g: Graph, set_uri: URIRef) -> int:
    # Accept schema:itemListElement under either schema or schema1
    elems = set()
    for _, _, li in g.triples((set_uri, SCHEMA.itemListElement, None)):
        elems.add(li)
    for _, _, li in g.triples((set_uri, SCHEMA1.itemListElement, None)):
        elems.add(li)
    return len(elems)

def detect_setlists(g: Graph) -> set:
    sets = set()
    # Typed as ItemList (schema or schema1)
    for s in g.subjects(RDF.type, SCHEMA.ItemList):
        sets.add(s)
    for s in g.subjects(RDF.type, SCHEMA1.ItemList):
        sets.add(s)
    # Or anything that has itemListElement
    for s, _, _ in g.triples((None, SCHEMA.itemListElement, None)):
        sets.add(s)
    for s, _, _ in g.triples((None, SCHEMA1.itemListElement, None)):
        sets.add(s)
    return sets

def detect_performances(g: Graph, perf_suffix: str) -> set:
    perfs = set()
    # Typed LivePerformance
    for s in g.subjects(RDF.type, MM.LivePerformance):
        perfs.add(s)
    # Name pattern *_<event>_Performance
    for s in g.all_nodes():
        if isinstance(s, URIRef) and local_name(s).endswith(perf_suffix):
            perfs.add(s)
    return perfs

def try_link_pair(g: Graph, perf_uri: URIRef, set_uri: URIRef, add_counts=True):
    # ex:setList perf -> set
    g.add((perf_uri, EX.setList, set_uri))
    # schema:about set -> perf
    g.add((set_uri, SCHEMA.about, perf_uri))
    # itemListOrder ascending
    g.set((set_uri, SCHEMA.itemListOrder, ITEMLIST_ORDER_ASC))
    # numberOfItems (if discoverable)
    if add_counts:
        n = count_itemlist_elements(g, set_uri)
        if n > 0:
            g.set((set_uri, SCHEMA.numberOfItems, Literal(n, datatype=XSD.integer)))

def run(in_path: str, out_path: str, event_suffix: str, perf_suffix: str, set_suffix: str, dry_run=False, verbose=False):
    g = Graph()
    g.parse(in_path, format="turtle")

    # Bind common prefixes (harmless if already present)
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("schema1", SCHEMA1)
    g.bind("mm", MM)

    sets = detect_setlists(g)
    perfs = detect_performances(g, perf_suffix=perf_suffix)

    links = 0
    sets_updated = 0
    missing_perf = []
    missing_set = []

    # First pass: for each set, infer its performance by swapping suffix
    for set_uri in sets:
        ln = local_name(set_uri)
        if ln.endswith(set_suffix):
            perf_ln = ln[: -len(set_suffix)] + perf_suffix
            perf_uri = URIRef(str(set_uri).replace(ln, perf_ln))
            if perf_uri in perfs:
                try_link_pair(g, perf_uri, set_uri)
                links += 1
                sets_updated += 1
            else:
                missing_perf.append((set_uri, perf_uri))

    # Second pass: for each performance, infer its set by swapping suffix
    for perf_uri in perfs:
        ln = local_name(perf_uri)
        if ln.endswith(perf_suffix):
            set_ln = ln[: -len(perf_suffix)] + set_suffix
            set_uri = URIRef(str(perf_uri).replace(ln, set_ln))
            if set_uri in sets:
                try_link_pair(g, perf_uri, set_uri)
                links += 1
                sets_updated += 1
            else:
                missing_set.append((perf_uri, set_uri))

    summary = {
        "input_ttl": in_path,
        "output_ttl": out_path,
        "performances_detected": len(perfs),
        "setlists_detected": len(sets),
        "links_created_or_confirmed": links,
        "setlists_updated_with_counts": sets_updated,
        "missing_performance_for_setlists": len(missing_perf),
        "missing_setlist_for_performances": len(missing_set),
    }

    if verbose:
        if missing_perf:
            print("\n[WARN] Setlists with no matching performance (by name):")
            for s, p in missing_perf:
                print("  set:", s, "=> expected perf:", p)
        if missing_set:
            print("\n[WARN] Performances with no matching setlist (by name):")
            for p, s in missing_set:
                print("  perf:", p, "=> expected set:", s)

    if dry_run:
        print("\n[DRY-RUN] Summary:", summary)
        return

    g.serialize(out_path, format="turtle")
    print("\n[OK] Wrote:", out_path)
    print("Summary:", summary)

def main():
    parser = argparse.ArgumentParser(
        description="Link *_<Event>_Performance to *_<Event>_Set in a TTL by naming convention; add counts/order/about."
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input TTL path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output TTL path")
    parser.add_argument("--event-suffix", default="LiveAid1985", help="Event suffix used in names (default: LiveAid1985)")
    parser.add_argument("--perf-suffix", default="_LiveAid1985_Performance", help="Performance suffix token")
    parser.add_argument("--set-suffix", default="_LiveAid1985_Set", help="Set list suffix token")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only; no write")
    parser.add_argument("--verbose", action="store_true", help="Print missing pairs")
    args = parser.parse_args()

    run(
        in_path=args.in_path,
        out_path=args.out_path,
        event_suffix=args.event_suffix,
        perf_suffix=args.perf_suffix,
        set_suffix=args.set_suffix,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
