#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read existing TTLs, find Works (schema:MusicComposition) and Recordings (mm:Recording),
enrich from MusicBrainz: credits (composer/lyricist), ISWC, ISRCs, durations, and release links.
Writes kg/24_mb_enrich.ttl (separate file).

Py3.9-friendly. Requires: rdflib, musicbrainzngs, requests.
"""

import argparse, json, time, random, re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, Literal
from rdflib.namespace import XSD

# Namespaces
SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

SCHEMA_HTTPS = "https://schema.org/"
SCHEMA_HTTP  = "http://schema.org/"

# Caching
CACHE_DIR = Path("kg/enrichment/cache/musicbrainz")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def normalize_schema(g: Graph) -> Graph:
    out = Graph()
    for pfx, ns in g.namespaces(): out.bind(pfx, ns)
    for s,p,o in g:
        p2, o2 = p, o
        if isinstance(p, URIRef) and str(p).startswith(SCHEMA_HTTPS):
            p2 = URIRef(str(p).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if isinstance(o, URIRef) and str(o).startswith(SCHEMA_HTTPS):
            o2 = URIRef(str(o).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        out.add((s,p2,o2))
    return out

def load_graphs(paths: List[str]) -> Graph:
    g = Graph()
    for p in paths:
        gp = Graph()
        gp.parse(p, format="turtle")
        g += normalize_schema(gp)
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("ex", EX); g.bind("owl", OWL); g.bind("xsd", XSD)
    return g

def mbid_from_sameas(urls: List[str], kind: str) -> Optional[str]:
    # kind in {"work","recording"}
    for u in urls:
        if "musicbrainz.org/%s/" % kind in u:
            return u.rstrip("/").rsplit("/", 1)[-1]
    return None

def collect_entities(g: Graph) -> Tuple[Dict[URIRef, str], Dict[URIRef, str], Dict[URIRef, List[str]]]:
    works, recs, sameas = {}, {}, {}
    # gather owl:sameAs URLs for each subject (helps extract mbids)
    for s, _, o in g.triples((None, OWL.sameAs, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            sameas.setdefault(s, []).append(str(o))

    # Works
    for w in g.subjects(RDF.type, SCHEMA.MusicComposition):
        if not isinstance(w, URIRef): continue
        urls = sameas.get(w, [])
        mbid = mbid_from_sameas(urls, "work")
        if mbid: works[w] = mbid

    # Recordings
    for r in g.subjects(RDF.type, MM.Recording):
        if not isinstance(r, URIRef): continue
        urls = sameas.get(r, [])
        mbid = mbid_from_sameas(urls, "recording")
        if mbid: recs[r] = mbid

    return works, recs, sameas

# ---------- MusicBrainz client ----------
def mb_client(app="WembleyRewind", ver="0.1.0", contact="mailto:you@example.com"):
    import musicbrainzngs as mb
    mb.set_useragent(app, ver, contact)
    mb.set_hostname("musicbrainz.org")
    mb.set_rate_limit(True)
    return mb

def cached_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text("utf-8")) if path.exists() else None

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")

def backoff(attempt: int):
    time.sleep(min(30, 1.5 ** attempt + random.random()))

def fetch_work(mb, mbid: str) -> Optional[dict]:
    key = CACHE_DIR / f"work_{mbid}.json"
    data = cached_json(key)
    if data: return data
    import musicbrainzngs as mbx
    for a in range(6):
        try:
            data = mb.get_work_by_id(mbid, includes=["artist-rels", "isw-codes"])
            save_json(key, data); return data
        except mbx.NetworkError:
            backoff(a)
        except Exception:
            break
    return None

def fetch_recording(mb, mbid: str) -> Optional[dict]:
    key = CACHE_DIR / f"rec_{mbid}.json"
    data = cached_json(key)
    if data: return data
    import musicbrainzngs as mbx
    for a in range(6):
        try:
            data = mb.get_recording_by_id(mbid, includes=["isrcs", "releases", "artist-credits"])
            save_json(key, data); return data
        except mbx.NetworkError:
            backoff(a)
        except Exception:
            break
    return None

def iso8601_duration(ms: Optional[int]) -> Optional[str]:
    if not ms or ms <= 0: return None
    sec = ms // 1000
    return "PT%uS" % sec

def add_mb_work_triples(out: Graph, node: URIRef, work: dict):
    w = work.get("work") or {}
    # ISWC codes
    for it in (w.get("isw-code-list") or []):
        code = it.get("isw-code") or it.get("iswc")
        if code:
            out.add((node, SCHEMA.identifier, Literal(code)))
    # composer/lyricist from artist-rels
    for rel in (w.get("artist-relation-list") or []):
        rtype = rel.get("type") or ""
        a = (rel.get("artist") or {}).get("name")
        # we only emit literal names if we don't have nodes; you can post-link later
        if not a: continue
        if rtype in ("composer", "composer/lyricist", "composer & lyricist"):
            out.add((node, SCHEMA.composer, Literal(a)))
        if rtype in ("lyricist", "composer/lyricist", "composer & lyricist"):
            out.add((node, SCHEMA.lyricist, Literal(a)))
    # add owl:sameAs MB URL if missing (safety)
    mbid = w.get("id")
    if mbid:
        out.add((node, OWL.sameAs, URIRef(f"https://musicbrainz.org/work/{mbid}")))

def add_mb_recording_triples(out: Graph, node: URIRef, rec: dict):
    r = rec.get("recording") or {}
    # duration
    dur = r.get("length")  # milliseconds
    iso = iso8601_duration(int(dur)) if dur else None
    if iso:
        out.add((node, SCHEMA.duration, Literal(iso, datatype=XSD.duration)))
    # ISRCs
    for code in (r.get("isrc-list") or []):
        out.add((node, URIRef(str(MM) + "hasISRC"), Literal(code)))
    # Release (minimal)
    for rel in (r.get("release-list") or [])[:3]:  # keep light
        title = rel.get("title")
        rid   = rel.get("id")
        if not rid or not title: continue
        rel_node = URIRef(str(node) + "_on_" + rid[:8])
        out.add((rel_node, RDF.type, URIRef(str(MM) + "Release")))
        out.add((rel_node, SCHEMA.name, Literal(title)))
        out.add((rel_node, OWL.sameAs, URIRef(f"https://musicbrainz.org/release/{rid}")))
        out.add((node, URIRef(str(MM) + "onRelease"), rel_node))
    # External URLs from URL rels (if present via includes)
    for urlrel in (r.get("url-relation-list") or []):
        tgt = urlrel.get("target")
        if tgt:
            out.add((node, SCHEMA.sameAs, URIRef(tgt)))
    # add owl:sameAs MB URL if missing
    mbid = r.get("id")
    if mbid:
        out.add((node, OWL.sameAs, URIRef(f"https://musicbrainz.org/recording/{mbid}")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input TTL files (existing KG pieces)")
    ap.add_argument("--out", default="kg/24_mb_enrich.ttl")
    ap.add_argument("--app", default="WembleyRewind")
    ap.add_argument("--ver", default="0.1.0")
    ap.add_argument("--contact", default="mailto:you@example.com", help="Your email or project URL for MB user-agent")
    args = ap.parse_args()

    base = load_graphs(args.inputs)
    works, recs, _ = collect_entities(base)
    print(f"[scan] works={len(works)} recordings={len(recs)}")

    mb = mb_client(args.app, args.ver, args.contact)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("xsd", XSD)

    # Enrich works
    for w_node, w_mbid in works.items():
        data = fetch_work(mb, w_mbid)
        if data: add_mb_work_triples(out, w_node, data)
        time.sleep(1.0)  # polite

    # Enrich recordings
    for r_node, r_mbid in recs.items():
        data = fetch_recording(mb, r_mbid)
        if data: add_mb_recording_triples(out, r_node, data)
        time.sleep(1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[write] {args.out} (triples: {len(out)})")

if __name__ == "__main__":
    main()
