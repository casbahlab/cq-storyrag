#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link Songfacts for works (P5241) and artists (P5287) using Wikidata.
Now with --resolve-by-mbid: if a node lacks a QID, resolve it from its MusicBrainz ID via WDQS.

Writes: kg/26_songfacts_links.ttl
Requires: rdflib, requests (Py3.9-friendly)
"""

import argparse, json, time, random, re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, Literal
from rdflib.namespace import XSD

SCHEMA_HTTP  = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
SCHEMA = Namespace(SCHEMA_HTTP)
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

WD_ITEM  = "https://www.wikidata.org/wiki/"
WD_ENT   = "https://www.wikidata.org/entity/"
WD_JSON  = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WDQS     = "https://query.wikidata.org/sparql"

CACHE_ENTITY = Path("kg/enrichment/cache/wikidata_songfacts"); CACHE_ENTITY.mkdir(parents=True, exist_ok=True)
CACHE_QID    = Path("kg/enrichment/cache/wdqs_qid"); CACHE_QID.mkdir(parents=True, exist_ok=True)

# Songfacts properties
P_SONGFACTS_SONG   = "P5241"  # Songfacts song ID
P_SONGFACTS_ARTIST = "P5287"  # Songfacts artist ID

# MBID properties on Wikidata (used to resolve QIDs)
# artist → P434, work → P435, recording → P4404; we can union these for fuzzier cases
MB_PROPS = {
    "artist":   "P434",
    "work":     "P435",
    "recording":"P4404",
}

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
    out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)
    return out

def load_graphs(paths: List[str]) -> Graph:
    g = Graph()
    for p in paths:
        gp = Graph(); gp.parse(p, format="turtle"); g += normalize_schema(gp)
    return g

def label_of(g: Graph, u: URIRef) -> str:
    for o in g.objects(u, SCHEMA.name): return str(o)
    for o in g.objects(u, RDFS.label):  return str(o)
    su = str(u); return su.split("#")[-1] if "#" in su else su.rsplit("/",1)[-1]

def extract_qid_from_sameas(url: URIRef) -> Optional[str]:
    u = str(url)
    if u.startswith(WD_ITEM + "Q"):  return u.split("/")[-1]
    if u.startswith(WD_ENT  + "Q"):  return u.split("/")[-1]
    return None

def collect_targets(g: Graph) -> Dict[URIRef, Dict[str,str]]:
    """
    Return { node: { 'kind': 'artist|work|recording|unknown', 'qid': 'Q..' (optional), 'mbid': '...' (optional), 'mbkind': 'artist|work|recording' (optional)} }
    """
    targets: Dict[URIRef, Dict[str,str]] = {}

    # Heuristic kind by RDF.type
    def set_kind(s: URIRef, k: str):
        targets.setdefault(s, {})['kind'] = k

    for s in g.subjects(RDF.type, SCHEMA.MusicComposition):
        if isinstance(s, URIRef): set_kind(s, "work")
    for s in g.subjects(RDF.type, SCHEMA.Person):
        if isinstance(s, URIRef): set_kind(s, "artist")
    for s in g.subjects(RDF.type, SCHEMA.MusicGroup):
        if isinstance(s, URIRef): set_kind(s, "artist")
    for s in g.subjects(RDF.type, MM.Recording):
        if isinstance(s, URIRef): set_kind(s, "recording")

    # Discover QIDs and MBIDs from sameAs/ owl:sameAs
    for s,p,url in g.triples((None, None, None)):
        if not isinstance(s, URIRef) or not isinstance(url, URIRef): continue
        u = str(url)

        # QID
        if p in (SCHEMA.sameAs, OWL.sameAs):
            qid = extract_qid_from_sameas(url)
            if qid:
                targets.setdefault(s, {})['qid'] = qid

        # MBID via MusicBrainz URLs
        if p in (SCHEMA.sameAs, OWL.sameAs) and "musicbrainz.org/" in u:
            # Try to infer type from the path
            m = re.search(r"musicbrainz\.org/(artist|work|recording)/([0-9a-f-]{36})", u, re.I)
            if m:
                mbkind = m.group(1).lower()
                mbid   = m.group(2).lower()
                t = targets.setdefault(s, {})
                t['mbid']   = mbid
                t['mbkind'] = mbkind

    # Fill unknown kinds
    for s, info in targets.items():
        if 'kind' not in info:
            info['kind'] = info.get('mbkind', 'unknown')
    return targets

def fetch_entity(qid: str) -> Optional[dict]:
    cache = CACHE_ENTITY / f"{qid}.json"
    if cache.exists():
        try: return json.loads(cache.read_text("utf-8"))
        except Exception: pass
    url = WD_JSON.format(qid=qid)
    for a in range(6):
        try:
            r = requests.get(url, headers={"User-Agent":"WembleyRewind/0.1"})
            r.raise_for_status()
            data = r.json()
            cache.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception:
            time.sleep(min(30, 1.5 ** a + random.random()))
    return None

def extract_claim_values(ent: dict, prop: str) -> List[str]:
    entities = ent.get("entities", {})
    if not entities: return []
    qid = next(iter(entities.keys()))
    claims = entities.get(qid, {}).get("claims", {})
    vals = []
    for snak in claims.get(prop, []):
        dv = snak.get("mainsnak", {}).get("datavalue", {})
        v = dv.get("value")
        if isinstance(v, str): vals.append(v)
        elif isinstance(v, dict) and "id" in v: vals.append(v["id"])
    return vals

def wdqs_qid_from_mbid(mbid: str, kind_hint: Optional[str]) -> Optional[str]:
    """
    Resolve QID by MBID using WDQS. Try specific property first by kind,
    then fall back to union of known MB properties.
    """
    # Cache key
    ck = CACHE_QID / f"{kind_hint or 'any'}_{mbid}.json"
    if ck.exists():
        try: return json.loads(ck.read_text("utf-8")).get("qid")
        except Exception: pass

    def run_query(props: List[str]) -> Optional[str]:
        prop_str = " ".join(f"wdt:{p}" for p in props)
        q = f"""
        SELECT ?item WHERE {{
          VALUES ?prop {{ {prop_str} }}
          ?item ?prop "{mbid}" .
        }}
        LIMIT 1
        """
        for a in range(5):
            try:
                r = requests.get(WDQS, params={"query": q, "format": "json"}, headers={"User-Agent":"WembleyRewind/0.1"})
                r.raise_for_status()
                data = r.json()
                bs = data.get("results", {}).get("bindings", [])
                if bs:
                    return bs[0]["item"]["value"].rsplit("/",1)[-1]
            except Exception:
                time.sleep(min(20, 1.5 ** a + random.random()))
        return None

    # Try by hint
    hint_prop = MB_PROPS.get(kind_hint or "", None)
    qid = None
    if hint_prop:
        qid = run_query([hint_prop])
    # Fallback: union of all
    if not qid:
        qid = run_query(list(MB_PROPS.values()))

    ck.write_text(json.dumps({"qid": qid}), encoding="utf-8")
    return qid

def add_property_value(g: Graph, s: URIRef, propid: str, val: str):
    pv = URIRef(str(s) + f"_prop_{propid}_{abs(hash(val)) % 10**8}")
    g.add((pv, RDF.type, SCHEMA.PropertyValue))
    g.add((pv, SCHEMA.propertyID, Literal(propid)))
    g.add((pv, SCHEMA.value, Literal(val)))
    g.add((s,  SCHEMA.identifier, pv))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input TTL files")
    ap.add_argument("--out", default="kg/26_songfacts_links.ttl")
    ap.add_argument("--resolve-by-mbid", action="store_true", help="If a node has no QID, try resolving QID from its MusicBrainz ID via WDQS")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_graphs(args.inputs)
    targets = collect_targets(base)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)

    added_sf_song = added_sf_artist = 0
    used_qids = 0
    resolved_qids = 0
    misses = 0

    for node, info in targets.items():
        qid = info.get("qid")
        if not qid and args.resolve_by_mbid:
            mbid = info.get("mbid")
            if mbid:
                qid = wdqs_qid_from_mbid(mbid, info.get("mbkind") or info.get("kind"))
                if qid:
                    resolved_qids += 1
        if not qid:
            continue

        used_qids += 1
        ent = fetch_entity(qid)
        if not ent:
            misses += 1
            continue

        # P5241 → song/work page
        for sid in extract_claim_values(ent, P_SONGFACTS_SONG):
            if sid.startswith("http"):
                song_url = sid
                sid_val  = sid.rsplit("/", 1)[-1]
            else:
                song_url = f"https://www.songfacts.com/facts/{sid}"
                sid_val  = sid
            add_property_value(out, node, P_SONGFACTS_SONG, sid_val)
            out.add((node, SCHEMA.sameAs, URIRef(song_url)))

            added_sf_song += 1

        # P5287 → artist page
        for aid in extract_claim_values(ent, P_SONGFACTS_ARTIST):
            if aid.startswith("http"):
                art_url = aid
                aid_val = aid.rsplit("/", 1)[-1]
            else:
                art_url = f"https://www.songfacts.com/songs/{aid}"
                aid_val = aid
            add_property_value(out, node, P_SONGFACTS_ARTIST, aid_val)
            out.add((node, SCHEMA.sameAs, URIRef(art_url)))

            added_sf_artist += 1

        # Re-assert sameAs to QID (helps downstream linking)
        out.add((node, SCHEMA.sameAs, URIRef(WD_ITEM + qid)))

        # polite pause for WD servers
        time.sleep(0.15)

    if args.dry_run:
        print(f"[dry-run] nodes: {len(targets)} | with_qid: {used_qids} | qid_resolved_by_mbid: {resolved_qids} | sf_song: {added_sf_song} | sf_artist: {added_sf_artist} | misses: {misses}")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[write] {args.out} (triples={len(out)})")
    print(f"[stats] nodes: {len(targets)} | used_qids: {used_qids} | qid_resolved_by_mbid: {resolved_qids} | Songfacts song IDs: {added_sf_song} | Songfacts artist IDs: {added_sf_artist} | QID fetch misses: {misses}")

if __name__ == "__main__":
    main()
