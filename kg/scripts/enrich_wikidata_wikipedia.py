#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Wikidata/Wikipedia enrichments for entities in current TTLs.
- Resolve QIDs via MB URL relations or WDQS (by MBID).
- Add schema:image (P18), schema:video (P1651), schema:geo (P625),
  schema:url (P856), schema:description (en), schema:sameAs (QID/Wikipedia).

Writes kg/25_wikidata_enrich.ttl
Requires: rdflib, requests
"""

import argparse, json, re, time, random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, Literal
from rdflib.namespace import XSD

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

SCHEMA_HTTP = "http://schema.org/"
SCHEMA_HTTPS= "https://schema.org/"

WD_ENTITY   = "https://www.wikidata.org/entity/"
WD_ITEM     = "https://www.wikidata.org/wiki/"
WDQS        = "https://query.wikidata.org/sparql"
WPREST_SUM  = "https://en.wikipedia.org/api/rest_v1/page/summary/"

CACHE_DIR = Path("kg/enrichment/cache/wikidata"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Known MBID properties on WD (expandable)
MB_PROPS = [
    "P434",   # MusicBrainz artist ID
    "P435",   # MusicBrainz work ID
    "P436",   # MusicBrainz release group ID
    "P4404",  # MusicBrainz recording ID (newer property)
    "P5813",  # MusicBrainz area ID (venues sometimes via place/area)
]

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
        gp = Graph(); gp.parse(p, format="turtle"); g += normalize_schema(gp)
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("ex", EX); g.bind("owl", OWL); g.bind("xsd", XSD)
    return g

def collect_targets(g: Graph) -> Dict[URIRef, Dict[str,str]]:
    """Return {node: {'type': 'artist|work|recording|place|event', 'mbid': '…', 'wd_qid': 'Q…' (optional), 'wp': 'enwiki title' (optional)}}"""
    nodes: Dict[URIRef, Dict[str,str]] = {}

    def add_node(s: URIRef, t: str):
        nodes.setdefault(s, {}).setdefault('type', t)

    # Types (broad)
    for s in g.subjects(RDF.type, SCHEMA.Person): add_node(s, "artist")
    for s in g.subjects(RDF.type, SCHEMA.MusicGroup): add_node(s, "artist")
    for s in g.subjects(RDF.type, SCHEMA.MusicComposition): add_node(s, "work")
    for s in g.subjects(RDF.type, MM.Recording): add_node(s, "recording")
    for s in g.subjects(RDF.type, SCHEMA.Place): add_node(s, "place")
    for s in g.subjects(RDF.type, MM.LivePerformance): add_node(s, "event")
    for s in g.subjects(RDF.type, SCHEMA.Event): add_node(s, "event")

    # MBIDs and existing sameAs
    for s, _, url in g.triples((None, OWL.sameAs, None)):
        if not isinstance(s, URIRef) or not isinstance(url, URIRef): continue
        u = str(url)
        if "musicbrainz.org/" in u:
            mbid = u.rstrip("/").rsplit("/", 1)[-1]
            nodes.setdefault(s, {})['mbid'] = mbid
        if u.startswith(WD_ITEM+"Q"):
            nodes.setdefault(s, {})['wd_qid'] = u.split("/")[-1]
        if "wikipedia.org/wiki/" in u:
            nodes.setdefault(s, {})['wp'] = u.rsplit("/",1)[-1]

    # Also check schema:sameAs
    for s, _, url in g.triples((None, SCHEMA.sameAs, None)):
        if not isinstance(s, URIRef) or not isinstance(url, URIRef): continue
        u = str(url)
        if u.startswith(WD_ITEM+"Q"):
            nodes.setdefault(s, {})['wd_qid'] = u.split("/")[-1]
        if "wikipedia.org/wiki/" in u:
            nodes.setdefault(s, {})['wp'] = u.rsplit("/",1)[-1]
    return nodes

def wdqs_qid_from_mbid(mbid: str) -> Optional[str]:
    key = CACHE_DIR / f"qid_from_mbid_{mbid}.json"
    if key.exists():
        return json.loads(key.read_text("utf-8")).get("qid")
    query = """
    SELECT ?item WHERE {
      VALUES ?prop { %s }
      ?item ?prop "%s".
    } LIMIT 1
    """ % (" ".join("wdt:%s" % p for p in MB_PROPS), mbid)
    for attempt in range(6):
        try:
            r = requests.get(WDQS, params={"query": query, "format": "json"}, headers={"User-Agent":"WembleyRewind/0.1"})
            r.raise_for_status()
            data = r.json()
            qid = None
            bs = data.get("results", {}).get("bindings", [])
            if bs:
                qid = bs[0]["item"]["value"].split("/")[-1]
            save = {"qid": qid}
            key.write_text(json.dumps(save), encoding="utf-8")
            return qid
        except Exception:
            time.sleep(min(30, 1.5 ** attempt + random.random()))
    return None

def wd_entity(qid: str) -> Optional[dict]:
    key = CACHE_DIR / f"entity_{qid}.json"
    if key.exists():
        return json.loads(key.read_text("utf-8"))
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    for a in range(6):
        try:
            r = requests.get(url, headers={"User-Agent":"WembleyRewind/0.1"})
            r.raise_for_status()
            data = r.json()
            key.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception:
            time.sleep(min(30, 1.5 ** a + random.random()))
    return None

def extract_claim(entity: dict, prop: str) -> List[str]:
    res = []
    claims = entity.get("claims", {})
    for c in claims.get(prop, []):
        m = c.get("mainsnak", {})
        dat = m.get("datavalue", {})
        val = dat.get("value")
        if isinstance(val, dict) and "id" in val:
            res.append(val["id"])
        elif isinstance(val, str):
            res.append(val)
    return res

def extract_commons(entity: dict) -> Optional[str]:
    # P18 image (Commons file name) → turn into full URL
    imgs = extract_claim(entity, "P18")
    if not imgs: return None
    # Simple commons path (no hashing); good enough for slides/demos
    title = imgs[0].replace(" ", "_")
    return f"https://upload.wikimedia.org/wikipedia/commons/{title}"

def extract_coords(entity: dict) -> Optional[Tuple[float,float]]:
    # P625 globe-coordinate
    claims = entity.get("claims", {})
    for c in claims.get("P625", []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        lat = v.get("latitude"); lon = v.get("longitude")
        if isinstance(lat, (int,float)) and isinstance(lon, (int,float)):
            return float(lat), float(lon)
    return None

def extract_official_site(entity: dict) -> Optional[str]:
    urls = extract_claim(entity, "P856")
    return urls[0] if urls else None

def extract_youtube(entity: dict) -> Optional[str]:
    vids = extract_claim(entity, "P1651")  # YouTube video ID
    if vids:
        return "https://www.youtube.com/watch?v=" + vids[0]
    return None

def wd_label_desc(entity: dict, lang="en") -> Tuple[Optional[str], Optional[str]]:
    ents = entity.get("entities", {})
    qid  = next(iter(ents.keys()), None)
    if not qid: return None, None
    obj  = ents[qid]
    lab  = obj.get("labels", {}).get(lang, {}).get("value")
    desc = obj.get("descriptions", {}).get(lang, {}).get("value")
    return lab, desc

def enrich_node(out: Graph, node: URIRef, qid: str, entity: dict):
    # sameAs to QID
    out.add((node, SCHEMA.sameAs, URIRef(WD_ITEM + qid)))
    # description
    _, desc = wd_label_desc(entity, "en")
    if desc:
        out.add((node, SCHEMA.description, Literal(desc, lang="en")))
    # image
    img = extract_commons(entity)
    if img:
        out.add((node, SCHEMA.image, URIRef(img)))
    # coords
    cd = extract_coords(entity)
    if cd:
        lat, lon = cd
        geo = URIRef(str(node) + "_Geo")
        out.add((geo, RDF.type, SCHEMA.GeoCoordinates))
        out.add((geo, SCHEMA.latitude, Literal(str(lat), datatype=XSD.float)))
        out.add((geo, SCHEMA.longitude, Literal(str(lon), datatype=XSD.float)))
        out.add((node, SCHEMA.geo, geo))
    # official site
    site = extract_official_site(entity)
    if site:
        out.add((node, SCHEMA.url, URIRef(site)))
    # youtube
    yt = extract_youtube(entity)
    if yt:
        vid = URIRef(str(node) + "_Video")
        out.add((vid, RDF.type, SCHEMA.VideoObject))
        out.add((vid, SCHEMA.embedUrl, URIRef(yt)))
        out.add((node, SCHEMA.video, vid))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input TTL files (existing KG pieces)")
    ap.add_argument("--out", default="kg/25_wikidata_enrich.ttl")
    args = ap.parse_args()

    base = load_graphs(args.inputs)
    targets = collect_targets(base)
    print(f"[scan] entities to try: {len(targets)}")

    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("xsd", XSD)

    for node, info in targets.items():
        qid = info.get("wd_qid")
        if not qid:
            mbid = info.get("mbid")
            if mbid:
                qid = wdqs_qid_from_mbid(mbid)
        if not qid:
            continue
        ent = wd_entity(qid)
        if not ent:
            continue
        enrich_node(out, node, qid, ent)
        time.sleep(0.2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[write] {args.out} (triples: {len(out)})")

if __name__ == "__main__":
    main()
