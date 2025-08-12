#!/usr/bin/env python3
"""
Deduplicate Works (songs) across 33_recordings_works.ttl and 40_setlists_songs.ttl,
merging nodes that share the same MusicBrainz Work MBID. Rewrites all references to
the canonical Work IRI, normalizes schema.org http/https, and (optionally) drops
schema:superEvent from performances.

Usage:
  python dedupe_works_by_mbid.py \
    --recworks-in kg/33_recordings_works.ttl \
    --setlists-in kg/40_setlists_songs.ttl \
    --drop-superevent
"""
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL
from pathlib import Path
from typing import Optional, List, Set, Dict
import argparse, re, json

SCHEMA_HTTP  = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
SCHEMA = Namespace(SCHEMA_HTTP)
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

IS_PART_OF = [URIRef(SCHEMA_HTTP+"isPartOf"), URIRef(SCHEMA_HTTPS+"isPartOf")]
SUPEREVENT = [URIRef(SCHEMA_HTTP+"superEvent"), URIRef(SCHEMA_HTTPS+"superEvent")]

def load_graph(p: Path) -> Graph:
    g = Graph(); g.parse(p, format="turtle")
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("rdfs", RDFS); g.bind("owl", OWL)
    return g

def mb_work_id_from_sameas(g: Graph, s: URIRef) -> Optional[str]:
    for o in g.objects(s, OWL.sameAs):
        u = str(o)
        m = re.search(r"/work/([0-9a-f-]{36})", u, re.I)
        if m:
            return m.group(1).lower()
    return None

def normalize_schema_http(g: Graph) -> Graph:
    out = Graph()
    for pfx, ns in g.namespaces(): out.bind(pfx, ns)
    for s, p, o in g:
        p2, o2 = p, o
        if isinstance(p, URIRef) and str(p).startswith(SCHEMA_HTTPS):
            p2 = URIRef(str(p).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if isinstance(o, URIRef) and str(o).startswith(SCHEMA_HTTPS):
            o2 = URIRef(str(o).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        out.add((s, p2, o2))
    return out

def choose_canonical(graphs: List[Graph], nodes: Set[URIRef]) -> URIRef:
    def tcount(n: URIRef) -> int:
        total = 0
        for g in graphs:
            total += sum(1 for _ in g.triples((n, None, None)))
            total += sum(1 for _ in g.triples((None, None, n)))
        return total
    best, bestn = None, -1
    for n in nodes:
        c = tcount(n)
        if c > bestn or (c == bestn and (best is None or str(n) < str(best))):
            best, bestn = n, c
    return best  # type: ignore

def rewrite_graph(g: Graph, mapping: Dict[URIRef, URIRef], drop_superevent: bool) -> Graph:
    out = Graph()
    for pfx, ns in g.namespaces(): out.bind(pfx, ns)
    for s, p, o in g:
        if drop_superevent and (p in SUPEREVENT):
            continue
        s2 = mapping.get(s, s)
        o2 = mapping.get(o, o) if isinstance(o, URIRef) else o

        # normalize schema https->http on predicate/object
        p2 = p
        if isinstance(p, URIRef) and str(p).startswith(SCHEMA_HTTPS):
            p2 = URIRef(str(p).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if isinstance(o2, URIRef) and str(o2).startswith(SCHEMA_HTTPS):
            o2 = URIRef(str(o2).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))

        out.add((s2, p2, o2))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recworks-in",  default="kg/33_recordings_works.ttl")
    ap.add_argument("--setlists-in",  default="kg/40_setlists_songs.ttl")
    ap.add_argument("--recworks-out", default=None)
    ap.add_argument("--setlists-out", default=None)
    ap.add_argument("--drop-superevent", action="store_true",
                    help="Remove schema:superEvent triples on performances.")
    args = ap.parse_args()

    rw_in  = Path(args.recworks_in)
    st_in  = Path(args.setlists_in)
    rw_out = Path(args.recworks_out) if args.recworks_out else rw_in.with_suffix(".worksdedup.ttl")
    st_out = Path(args.setlists_out) if args.setlists_out else st_in.with_suffix(".worksdedup.ttl")

    g_rw = normalize_schema_http(load_graph(rw_in))
    g_st = normalize_schema_http(load_graph(st_in))

    # All Works
    works = set()
    works |= {w for w in g_rw.subjects(RDF.type, SCHEMA.MusicComposition) if isinstance(w, URIRef)}
    works |= {w for w in g_st.subjects(RDF.type, SCHEMA.MusicComposition) if isinstance(w, URIRef)}

    # Cluster by MBID
    by_mbid: Dict[str, Set[URIRef]] = {}
    no_mbid: Set[URIRef] = set()

    for w in works:
        mbid = mb_work_id_from_sameas(g_rw, w) or mb_work_id_from_sameas(g_st, w)
        if mbid:
            by_mbid.setdefault(mbid, set()).add(w)
        else:
            no_mbid.add(w)

    # Build mapping (only MBIDs with >1 nodes)
    mapping: Dict[URIRef, URIRef] = {}
    groups = []
    for mbid, nodes in by_mbid.items():
        if len(nodes) <= 1:
            continue
        canon = choose_canonical([g_rw, g_st], nodes)
        dups = [n for n in nodes if n != canon]
        for d in dups:
            mapping[d] = canon
        groups.append({"mbid": mbid, "canonical": str(canon), "dups": [str(d) for d in dups]})

    if mapping:
        print(f"[dedupe] merging {sum(len(g['dups']) for g in groups)} Work nodes across {len(groups)} MBIDs")
    else:
        print("[dedupe] No duplicate Works detected by MBID.")

    # Rewrite both graphs
    g_rw2 = rewrite_graph(g_rw, mapping, args.drop_superevent)
    g_st2 = rewrite_graph(g_st, mapping, args.drop_superevent)

    # Save
    rw_out.parent.mkdir(parents=True, exist_ok=True)
    g_rw2.serialize(str(rw_out), format="turtle")
    g_st2.serialize(str(st_out), format="turtle")

    report = {
        "works_total": len(works),
        "works_with_mbid": sum(len(v) for v in by_mbid.values()),
        "mbid_clusters": sum(1 for v in by_mbid.values() if len(v) > 1),
        "merged_nodes": sum(len(g["dups"]) for g in groups),
        "no_mbid_works": len(no_mbid),
        "outputs": {"recworks": str(rw_out), "setlists": str(st_out)},
        "groups_preview": groups[:10]
    }
    rep = rw_out.with_suffix(".worksdedup.json")
    Path(rep).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[write] {rw_out}")
    print(f"[write] {st_out}")
    print(f"[report] {rep}")

if __name__ == "__main__":
    main()
