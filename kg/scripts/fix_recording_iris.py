#!/usr/bin/env python3
from rdflib import Graph, Namespace, URIRef, RDF, RDFS
from rdflib.namespace import OWL
import re, sys, pathlib

SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX_BASE = "http://wembrewind.live/ex#"

def clean_title(title: str) -> str:
    if not title: return ""
    t = title.strip()
    t = re.sub(r"^\[(.+?)\]$", r"\1", t).strip()
    t = re.sub(r"\s*\((?:live|live aid|wembley|philadelphia|1985|remaster|mix|version|edit|stereo|mono)[^)]*\)\s*$","",t,flags=re.I).strip()
    return re.sub(r"\s+"," ",t)

def pascal_slug(text: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+",(text or "").strip())
    slug = "".join(p.capitalize() for p in parts if p) or "Recording"
    return ("A"+slug) if slug[0].isdigit() else slug

def shortid(s: str) -> str:
    return re.sub(r"[^0-9a-fA-F]","", s or "")[:8]

def parse_mbid_from_sameas(g, subj):
    for o in g.objects(subj, OWL.sameAs):
        m = re.search(r"/recording/([0-9a-f-]{36})", str(o), flags=re.I)
        if m: return m.group(1)
    return None

def get_title(g, s):
    for o in g.objects(s, SCHEMA.name): return str(o)
    for o in g.objects(s, RDFS.label):  return str(o)
    return str(s).split("#")[-1]

def rec_new_iri(g, s):
    title = clean_title(get_title(g, s))
    base = pascal_slug(title)
    mbid = parse_mbid_from_sameas(g, s) or str(s)
    sid  = shortid(mbid)
    return URIRef(f"{EX_BASE}Recording_{base}_{sid}" if sid else f"{EX_BASE}Recording_{base}")

def rewrite_graph_iris(g, m):
    out = Graph()
    for pfx, ns in g.namespaces(): out.bind(pfx, ns)
    for s,p,o in g:
        s2 = m.get(s, s)
        o2 = m.get(o, o) if isinstance(o, URIRef) else o
        out.add((s2,p,o2))
    return out

def main(rec_in, perf_in, rec_out, perf_out):
    g_rec = Graph(); g_rec.parse(rec_in, format="turtle")
    g_perf = Graph(); g_perf.parse(perf_in, format="turtle")

    mapping = {}
    seen = set()
    for rec in g_rec.subjects(RDF.type, MM.Recording):
        new = rec_new_iri(g_rec, rec)
        if new != rec:
            u = new; k=2
            while u in seen and u != rec:
                u = URIRef(str(new)+f"_{k}"); k+=1
            mapping[rec] = u; seen.add(u)

    g_rec2  = rewrite_graph_iris(g_rec, mapping)
    g_perf2 = rewrite_graph_iris(g_perf, mapping)

    g_rec2.serialize(rec_out, format="turtle")
    g_perf2.serialize(perf_out, format="turtle")
    print(f"Renamed {len(mapping)} recordings.\n-> {rec_out}\n-> {perf_out}")

if __name__ == "__main__":
    rec_in  = sys.argv[1] if len(sys.argv)>1 else "kg/24_recordings_works.ttl"
    perf_in = sys.argv[2] if len(sys.argv)>2 else "kg/23_liveaid_setlists.ttl"
    rec_out = sys.argv[3] if len(sys.argv)>3 else "kg/24_recordings_works.fixed.ttl"
    perf_out= sys.argv[4] if len(sys.argv)>4 else "kg/23_liveaid_setlists.fixed.ttl"
    main(rec_in, perf_in, rec_out, perf_out)
