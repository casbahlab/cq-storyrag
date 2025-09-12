#!/usr/bin/env python3
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL
from pathlib import Path
import argparse, re, hashlib

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EXBASE = "http://wembrewind.live/ex#"

DROP_PAREN_RE = re.compile(r"\s*\((?:live|live aid|wembley|philadelphia|1985|remaster|mix|version|edit|stereo|mono)[^)]*\)\s*$", re.I)

def pascal_slug(text):
    parts = re.split(r"[^A-Za-z0-9]+", (text or "").strip())
    slug = "".join(p.capitalize() for p in parts if p)
    if not slug: slug = "Work"
    if slug[0].isdigit(): slug = "A" + slug
    return slug

def clean_title(title):
    if not title: return ""
    t = title.strip()
    t = re.sub(r"^\[(.+?)\]$", r"\1", t).strip()
    t = DROP_PAREN_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def short_mbid(s):
    m = re.search(r"[0-9a-f]{8}", re.sub(r"[^0-9a-f]", "", (s or "").lower()))
    return m.group(0) if m else ""

def parse_work_mbid(g, s):
    for o in g.objects(s, OWL.sameAs):
        u = str(o)
        m = re.search(r"/work/([0-9a-f-]{36})", u, re.I)
        if m: return m.group(1)
    return None

def label_of(g, s):
    for o in g.objects(s, SCHEMA.name): return str(o)
    for o in g.objects(s, RDFS.label):  return str(o)
    return None

def new_work_iri(title, mbid, fallback_seed):
    base = pascal_slug(clean_title(title))
    sid = short_mbid(mbid or "")
    if not sid:
        sid = hashlib.md5((fallback_seed or "").encode("utf-8")).hexdigest()[:8]
    return URIRef(f"{EXBASE}Work_{base}_{sid}")

def rewrite_graph(g, mapping):
    out = Graph()
    for pfx, ns in g.namespaces(): out.bind(pfx, ns)
    for s,p,o in g:
        s2 = mapping.get(s, s)
        o2 = mapping.get(o, o) if isinstance(o, URIRef) else o
        out.add((s2,p,o2))
    return out

def run(recworks_in, setlists_in, recworks_out, setlists_out):
    g_rw = Graph(); g_rw.parse(recworks_in, format="turtle")
    g_st = Graph(); g_st.parse(setlists_in, format="turtle")

    works = (set(g_rw.subjects(RDF.type, SCHEMA.MusicComposition)) |
             set(g_st.subjects(RDF.type, SCHEMA.MusicComposition)))
    works = {w for w in works if isinstance(w, URIRef)}

    mapping, seen_new = {}, set()
    for w in works:
        title = label_of(g_rw, w) or label_of(g_st, w) or str(w).split("#")[-1]
        mbid  = parse_work_mbid(g_rw, w) or parse_work_mbid(g_st, w)
        new   = new_work_iri(title, mbid, str(w))
        cand  = new; k = 2
        while cand in seen_new and cand != w:
            cand = URIRef(str(new) + f"_{k}"); k += 1
        mapping[w] = cand; seen_new.add(cand)

    g_rw2 = rewrite_graph(g_rw, mapping)
    g_st2 = rewrite_graph(g_st, mapping)
    Path(recworks_out).parent.mkdir(parents=True, exist_ok=True)
    g_rw2.serialize(recworks_out, format="turtle")
    g_st2.serialize(setlists_out, format="turtle")
    return len(works), len([1 for k,v in mapping.items() if k!=v])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rename Work IRIs to Work_<Title>_<mbid8> and update references.")
    ap.add_argument("--recworks-in",  default="../33_recordings_works.ttl")
    ap.add_argument("--setlists-in",  default="../40_setlists_songs.ttl")
    ap.add_argument("--recworks-out", default="../33_recordings_works.ttl")
    ap.add_argument("--setlists-out", default="../40_setlists_songs.ttl")
    args = ap.parse_args()
    works, renamed = run(args.recworks_in, args.setlists_in, args.recworks_out, args.setlists_out)
    print(f"[done] works seen={works}, renamed={renamed}")
