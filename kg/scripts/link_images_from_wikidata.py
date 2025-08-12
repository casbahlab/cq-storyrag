#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add images (and optional logos) from Wikidata to existing nodes.

- Looks for schema:sameAs / owl:sameAs pointing to Wikidata QIDs.
- (Optional) If a node has only a MusicBrainz link, --resolve-by-mbid will
  query WDQS to find the QID and proceed.
- Pulls P18 (image) -> schema:image using Commons Special:FilePath URL.
- (Optional) Pulls P154 (logo image) -> schema:logo using Special:FilePath.
- Writes an additive TTL (default: kg/26_wikidata_images.ttl).

Py3.9-friendly. Requires: rdflib, requests.
"""

import argparse, json, time, random, re, urllib.parse
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
CACHE_ENTITY = Path("kg/enrichment/cache/wikidata_images"); CACHE_ENTITY.mkdir(parents=True, exist_ok=True)
CACHE_QID    = Path("kg/enrichment/cache/wdqs_qid");        CACHE_QID.mkdir(parents=True, exist_ok=True)

# Wikidata properties
P_IMAGE = "P18"   # image
P_LOGO  = "P154"  # logo image

# For MBID→QID resolution
MB_PROPS = {"artist": "P434", "work": "P435", "recording": "P4404", "place": "P1004"}

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
    Return { node: { 'kind': 'artist|work|recording|event|place|unknown', 'qid'?, 'mbid'?, 'mbkind'? } }
    """
    targets: Dict[URIRef, Dict[str,str]] = {}

    # Types → kind (broad)
    def set_kind(s, k): targets.setdefault(s, {})['kind'] = k
    for s in g.subjects(RDF.type, SCHEMA.Person):        set_kind(s, "artist")
    for s in g.subjects(RDF.type, SCHEMA.MusicGroup):    set_kind(s, "artist")
    for s in g.subjects(RDF.type, SCHEMA.MusicComposition): set_kind(s, "work")
    for s in g.subjects(RDF.type, MM.Recording):         set_kind(s, "recording")
    for s in g.subjects(RDF.type, SCHEMA.Event):         set_kind(s, "event")
    for s in g.subjects(RDF.type, MM.LivePerformance):   set_kind(s, "event")
    for s in g.subjects(RDF.type, SCHEMA.Place):         set_kind(s, "place")

    # Discover QIDs & MBIDs
    for s,p,url in g.triples((None, None, None)):
        if not isinstance(s, URIRef) or not isinstance(url, URIRef): continue
        if p in (SCHEMA.sameAs, OWL.sameAs):
            qid = extract_qid_from_sameas(url)
            if qid:
                targets.setdefault(s, {})['qid'] = qid
            u = str(url)
            m = re.search(r"musicbrainz\.org/(artist|work|recording|area|place)/([0-9a-f-]{36}|[0-9a-f-]{8}-[0-9a-f-]{4}-[0-9a-f-]{4}-[0-9a-f-]{4}-[0-9a-f-]{12})", u, re.I)
            if m:
                targets.setdefault(s, {})['mbkind'] = m.group(1).lower().replace("area", "place")
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

def commons_file_path_url(filename: str) -> str:
    # Use Special:FilePath to avoid hashing; it will redirect to the actual CDN URL.
    safe = urllib.parse.quote(filename.replace(" ", "_"))
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{safe}"

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input TTL files")
    ap.add_argument("--out", default="kg/26_wikidata_images.ttl")
    ap.add_argument("--resolve-by-mbid", action="store_true", help="Resolve QIDs from MusicBrainz IDs if missing")
    ap.add_argument("--include-logos", action="store_true", help="Also add schema:logo from P154 where present")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_graphs(args.inputs)
    targets = collect_targets(base)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)

    used_qids = resolved_qids = 0
    images_added = logos_added = 0

    for node, info in targets.items():
        qid = info.get("qid")
        if not qid and args.resolve_by_mbid and info.get("mbid"):
            qid = wdqs_qid_from_mbid(info["mbid"], info.get("mbkind") or info.get("kind"))
            if qid: resolved_qids += 1
        if not qid:
            continue

        used_qids += 1
        ent = fetch_entity(qid)
        if not ent:
            continue

        # P18 image(s) → schema:image
        imgs = extract_claim_strings(ent, P_IMAGE)
        for fname in imgs[:2]:  # at most two to keep it light
            url = commons_file_path_url(fname)
            out.add((node, SCHEMA.image, URIRef(url)))
            images_added += 1

        # Optional P154 logo → schema:logo
        if args.include_logos:
            logos = extract_claim_strings(ent, P_LOGO)
            for fname in logos[:2]:
                url = commons_file_path_url(fname)
                out.add((node, SCHEMA.logo, URIRef(url)))
                logos_added += 1

        # Re-assert QID for convenience
        out.add((node, SCHEMA.sameAs, URIRef(WD_ITEM + qid)))

        time.sleep(0.15)

    if args.dry_run:
        print(f"[dry-run] nodes={len(targets)} | used_qids={used_qids} | qid_resolved_by_mbid={resolved_qids} | images_added={images_added} | logos_added={logos_added}")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[write] {args.out} (triples={len(out)})")
    print(f"[stats] used_qids={used_qids} | qid_resolved_by_mbid={resolved_qids} | images_added={images_added} | logos_added={logos_added}")

if __name__ == "__main__":
    main()
