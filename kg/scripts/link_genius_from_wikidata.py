#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Link Genius (artists & works) using Wikidata properties.

Adds, additively:
  • Works (schema:MusicComposition):
      - P6218 (Genius ID) as schema:identifier (schema:PropertyValue)
      - schema:sameAs https://genius.com/$1
      - (optional) P6361 numeric ID -> schema:sameAs https://genius.com/songs/$1
  • Artists (schema:Person / schema:MusicGroup):
      - P2373 (Genius artist ID) as schema:identifier
      - schema:sameAs https://genius.com/artists/$1
      - (optional) P6351 numeric ID -> schema:sameAs https://genius.com/artists/$1

If a node has no QID but has a MusicBrainz link, --resolve-by-mbid will fetch the
QID from WDQS (via MB properties) and proceed.

Requires: rdflib, requests (Py3.9-friendly)
"""

import argparse, json, time, random, re
from pathlib import Path
from typing import Optional, Dict, List

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

# Caches
CACHE_ENTITY = Path("kg/enrichment/cache/wikidata_genius"); CACHE_ENTITY.mkdir(parents=True, exist_ok=True)
CACHE_QID    = Path("kg/enrichment/cache/wdqs_qid");        CACHE_QID.mkdir(parents=True, exist_ok=True)

# Wikidata properties we read
P_GENIUS_ID_WORK     = "P6218"  # Genius ID (slug for songs/works)
P_GENIUS_ID_ARTIST   = "P2373"  # Genius artist ID (slug)
P_GENIUS_NUM_SONG    = "P6361"  # Genius song numeric ID
P_GENIUS_NUM_ARTIST  = "P6351"  # Genius artist numeric ID

# MBID props used to resolve QIDs if needed
MB_PROPS = {"artist": "P434", "work": "P435", "recording": "P4404"}

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

def extract_qid_from_sameas(url: URIRef) -> Optional[str]:
    u = str(url)
    if u.startswith(WD_ITEM + "Q"):  return u.split("/")[-1]
    if u.startswith(WD_ENT  + "Q"):  return u.split("/")[-1]
    return None

def collect_targets(g: Graph) -> Dict[URIRef, Dict[str,str]]:
    """
    Return { node: { 'kind': 'artist|work|recording|unknown', 'qid'?, 'mbid'?, 'mbkind'? } }
    """
    targets: Dict[URIRef, Dict[str,str]] = {}

    # Types → kind
    for s in g.subjects(RDF.type, SCHEMA.MusicComposition):
        if isinstance(s, URIRef): targets.setdefault(s, {})['kind'] = "work"
    for s in g.subjects(RDF.type, SCHEMA.Person):
        if isinstance(s, URIRef): targets.setdefault(s, {})['kind'] = "artist"
    for s in g.subjects(RDF.type, SCHEMA.MusicGroup):
        if isinstance(s, URIRef): targets.setdefault(s, {})['kind'] = "artist"
    for s in g.subjects(RDF.type, MM.Recording):
        if isinstance(s, URIRef): targets.setdefault(s, {})['kind'] = "recording"

    # Discover QIDs + MBIDs
    for s,p,url in g.triples((None, None, None)):
        if not isinstance(s, URIRef) or not isinstance(url, URIRef): continue
        if p in (SCHEMA.sameAs, OWL.sameAs):
            qid = extract_qid_from_sameas(url)
            if qid:
                targets.setdefault(s, {})['qid'] = qid
            u = str(url)
            m = re.search(r"musicbrainz\.org/(artist|work|recording)/([0-9a-f-]{36})", u, re.I)
            if m:
                targets.setdefault(s, {})['mbkind'] = m.group(1).lower()
                targets[s]['mbid'] = m.group(2).lower()

    # Fill kind from mbkind if missing
    for s, info in targets.items():
        info.setdefault('kind', info.get('mbkind', 'unknown'))
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

def extract_claim_strings(ent: dict, prop: str) -> List[str]:
    entities = ent.get("entities", {})
    if not entities: return []
    qid = next(iter(entities.keys()))
    claims = entities.get(qid, {}).get("claims", {})
    vals = []
    for snak in claims.get(prop, []):
        dv = snak.get("mainsnak", {}).get("datavalue", {})
        v = dv.get("value")
        if isinstance(v, str):
            vals.append(v)
        elif isinstance(v, dict) and "id" in v:
            vals.append(v["id"])
    return vals

def wdqs_qid_from_mbid(mbid: str, kind_hint: Optional[str]) -> Optional[str]:
    ck = CACHE_QID / f"{kind_hint or 'any'}_{mbid}.json"
    if ck.exists():
        try: return json.loads(ck.read_text("utf-8")).get("qid")
        except Exception: pass

    def run_query(props: List[str]) -> Optional[str]:
        prop_str = " ".join(f"wdt:{p}" for p in props)
        q = f"""SELECT ?item WHERE {{
          VALUES ?prop {{ {prop_str} }}
          ?item ?prop "{mbid}" .
        }} LIMIT 1"""
        for a in range(5):
            try:
                r = requests.get(WDQS, params={"query": q, "format": "json"},
                                 headers={"User-Agent":"WembleyRewind/0.1"})
                r.raise_for_status()
                b = r.json().get("results", {}).get("bindings", [])
                if b:
                    return b[0]["item"]["value"].rsplit("/",1)[-1]
            except Exception:
                time.sleep(min(20, 1.5 ** a + random.random()))
        return None

    hint_prop = MB_PROPS.get(kind_hint or "", None)
    qid = run_query([hint_prop]) if hint_prop else None
    if not qid: qid = run_query(list(MB_PROPS.values()))

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
    ap.add_argument("inputs", nargs="+", help="Input TTLs")
    ap.add_argument("--out", default="kg/26_genius_links.ttl")
    ap.add_argument("--resolve-by-mbid", action="store_true",
                    help="Resolve QIDs from MusicBrainz IDs via WDQS if missing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_graphs(args.inputs)
    targets = collect_targets(base)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)

    used_qids = resolved_qids = 0
    added_work_slug = added_artist_slug = 0
    added_work_num  = added_artist_num  = 0
    misses = 0

    for node, info in targets.items():
        qid = info.get("qid")
        if not qid and args.resolve_by_mbid and info.get("mbid"):
            qid = wdqs_qid_from_mbid(info["mbid"], info.get("mbkind") or info.get("kind"))
            if qid: resolved_qids += 1
        if not qid: continue

        used_qids += 1
        ent = fetch_entity(qid)
        if not ent:
            misses += 1
            continue

        # Works/songs: P6218 (slug) + P6361 (numeric)
        for slug in extract_claim_strings(ent, P_GENIUS_ID_WORK):
            add_property_value(out, node, P_GENIUS_ID_WORK, slug)
            out.add((node, SCHEMA.sameAs, URIRef(f"https://genius.com/{slug}")))
            added_work_slug += 1
        for num in extract_claim_strings(ent, P_GENIUS_NUM_SONG):
            add_property_value(out, node, P_GENIUS_NUM_SONG, num)
            out.add((node, SCHEMA.sameAs, URIRef(f"https://genius.com/songs/{num}")))
            added_work_num += 1

        # Artists: P2373 (slug) + P6351 (numeric)
        for slug in extract_claim_strings(ent, P_GENIUS_ID_ARTIST):
            add_property_value(out, node, P_GENIUS_ID_ARTIST, slug)
            out.add((node, SCHEMA.sameAs, URIRef(f"https://genius.com/artists/{slug}")))
            added_artist_slug += 1
        for num in extract_claim_strings(ent, P_GENIUS_NUM_ARTIST):
            add_property_value(out, node, P_GENIUS_NUM_ARTIST, num)
            out.add((node, SCHEMA.sameAs, URIRef(f"https://genius.com/artists/{num}")))
            added_artist_num += 1

        # Re-assert sameAs to QID (handy downstream)
        out.add((node, SCHEMA.sameAs, URIRef(WD_ITEM + qid)))
        time.sleep(0.15)

    if args.dry_run:
        print(f"[dry-run] nodes: {len(targets)} | used_qids: {used_qids} | qid_resolved_by_mbid: {resolved_qids} | "
              f"work_slug: {added_work_slug} | work_num: {added_work_num} | artist_slug: {added_artist_slug} | artist_num: {added_artist_num} | misses: {misses}")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[write] {args.out} (triples={len(out)})")
    print(f"[stats] nodes: {len(targets)} | used_qids: {used_qids} | qid_resolved_by_mbid: {resolved_qids} | "
          f"Genius work slugs: {added_work_slug} | Genius work numeric: {added_work_num} | "
          f"Genius artist slugs: {added_artist_slug} | Genius artist numeric: {added_artist_num} | misses: {misses}")

if __name__ == "__main__":
    main()
