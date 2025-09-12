#!/usr/bin/env python3
"""
Deduplicate mm:LivePerformance nodes across:
  - 40_setlists_songs.ttl
  - 33_recordings_works.ttl

Default key: (schema:isPartOf EVENT, schema:performer ARTIST, schema:location LOCATION)

Usage:
  python dedupe_performances.py \
    --setlists-in kg/40_setlists_songs.ttl \
    --recworks-in kg/33_recordings_works.ttl

Options:
  --key event-artist-location   (default)
  --key event-artist
  --inplace                     (overwrite inputs, writing .bak first)
  --dry-run                     (report only)
"""

from rdflib import Graph, Namespace, URIRef, RDF
from rdflib.namespace import RDFS, OWL, XSD
from pathlib import Path
import argparse, json

MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

SCHEMA_HTTP  = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"

IS_PART_OF = [URIRef(SCHEMA_HTTP+"isPartOf"), URIRef(SCHEMA_HTTPS+"isPartOf")]
PERFORMER  = [URIRef(SCHEMA_HTTP+"performer"), URIRef(SCHEMA_HTTPS+"performer")]
LOCATION   = [URIRef(SCHEMA_HTTP+"location"),  URIRef(SCHEMA_HTTPS+"location")]

def bind_common(g: Graph):
    g.bind("schema", Namespace(SCHEMA_HTTP))
    g.bind("mm", MM); g.bind("rdfs", RDFS); g.bind("owl", OWL); g.bind("xsd", XSD)

def load_graph(p: Path) -> Graph:
    g = Graph(); g.parse(p, format="turtle"); bind_common(g); return g

def first_obj_any(g: Graph, s: URIRef, props: list[URIRef]):
    for p in props:
        for o in g.objects(s, p):
            if isinstance(o, URIRef):
                return o
    return None

def perf_candidates(g: Graph):
    # nodes typed as LivePerformance OR having event+artist props (any schema http/https)
    cand = set(g.subjects(RDF.type, MM.LivePerformance))
    for s in g.subjects(None, None):
        if not isinstance(s, URIRef): continue
        e = first_obj_any(g, s, IS_PART_OF)
        a = first_obj_any(g, s, PERFORMER)
        if e and a:
            cand.add(s)
    return cand

def key_for(g: Graph, s: URIRef, mode: str):
    event  = first_obj_any(g, s, IS_PART_OF)
    artist = first_obj_any(g, s, PERFORMER)
    if mode == "event-artist-location":
        loc = first_obj_any(g, s, LOCATION)
        return (event, artist, loc)
    return (event, artist)

def triple_count_across(gs, node: URIRef) -> int:
    c = 0
    for g in gs:
        c += sum(1 for _ in g.triples((node, None, None)))
        c += sum(1 for _ in g.triples((None, None, node)))
    return c

def choose_canonical(gs, nodes):
    best = None; bestn = -1
    for n in nodes:
        ntrip = triple_count_across(gs, n)
        if ntrip > bestn or (ntrip == bestn and (best is None or str(n) < str(best))):
            best, bestn = n, ntrip
    return best

def rewrite_graph(g: Graph, mapping: dict[URIRef, URIRef]) -> Graph:
    out = Graph(); bind_common(out)
    for s,p,o in g:
        s2 = mapping.get(s, s)
        o2 = mapping.get(o, o) if isinstance(o, URIRef) else o
        out.add((s2,p,o2))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setlists-in",  default="kg/40_setlists_songs.ttl")
    ap.add_argument("--recworks-in",  default="kg/33_recordings_works.ttl")
    ap.add_argument("--setlists-out", default=None)
    ap.add_argument("--recworks-out", default=None)
    ap.add_argument("--key", choices=["event-artist-location","event-artist"], default="event-artist-location")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    set_in = Path(args.setlists_in); rw_in = Path(args.recworks_in)
    g_set = load_graph(set_in); g_rw = load_graph(rw_in)

    perfs = perf_candidates(g_set) | perf_candidates(g_rw)

    # cluster by key (prefer keys from a graph that actually has the props)
    clusters = {}
    for p in perfs:
        k = key_for(g_set, p, args.key)
        if all(x is None for x in k):
            k = key_for(g_rw, p, args.key)
        clusters.setdefault(k, set()).add(p)

    mapping = {}
    groups = []
    for k, nodes in clusters.items():
        nodes = {n for n in nodes if isinstance(n, URIRef)}
        if len(nodes) <= 1: continue
        canon = choose_canonical([g_set, g_rw], nodes)
        dups  = [n for n in nodes if n != canon]
        for d in dups: mapping[d] = canon
        groups.append({
            "key": tuple(str(x) if isinstance(x, URIRef) else None for x in k),
            "canonical": str(canon),
            "dups": [str(d) for d in dups]
        })

    if not mapping:
        print("[dedupe] No duplicate performance groups found.")
        return

    print(f"[dedupe] groups: {len(groups)} | nodes to merge: {len(mapping)}")

    if args.dry_run:
        print(json.dumps({"groups_preview": groups[:5], "total_groups": len(groups)}, indent=2))
        return

    g_set2 = rewrite_graph(g_set, mapping)
    g_rw2  = rewrite_graph(g_rw, mapping)

    # outputs
    if args.inplace:
        set_bak = set_in.with_suffix(set_in.suffix + ".bak")
        rw_bak  = rw_in.with_suffix(rw_in.suffix + ".bak")
        set_bak.write_text(g_set.serialize(format="turtle"), encoding="utf-8")
        rw_bak.write_text(g_rw.serialize(format="turtle"), encoding="utf-8")
        set_out, rw_out = set_in, rw_in
    else:
        set_out = Path(args.setlists_out) if args.setlists_out else set_in.with_suffix(".deduped.ttl")
        rw_out  = Path(args.recworks_out) if args.recworks_out else rw_in.with_suffix(".deduped.ttl")

    g_set2.serialize(str(set_out), format="turtle")
    g_rw2.serialize(str(rw_out),   format="turtle")

    report = {
        "key": args.key,
        "totals": {"groups": len(groups), "dups": len(mapping), "perfs_seen": len(perfs)},
        "groups": groups,
        "outputs": {"setlists": str(set_out), "recworks": str(rw_out)}
    }
    rep_path = Path(args.report) if args.report else set_out.with_suffix(".dedupe.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[write] {set_out}")
    print(f"[write] {rw_out}")
    print(f"[report] {rep_path}")

if __name__ == "__main__":
    main()
