#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add YouTube links to Works (songs) — and optionally Artists — using Wikidata.

- For Works (schema:MusicComposition): use P1651 (YouTube video ID)
  → emit schema:video [ a schema:VideoObject ;
                        schema:embedUrl <https://www.youtube.com/watch?v=ID> ;
                        schema:url      <https://www.youtube.com/watch?v=ID> ;
                        schema:provider <https://www.youtube.com/> ] .

- For Artists (schema:Person/MusicGroup) with --include-artists:
  use P2397 (YouTube channel ID)
  → emit schema:sameAs <https://www.youtube.com/channel/ID> .

Also supports:
  --resolve-by-mbid   : resolve missing QIDs from MusicBrainz IDs via WDQS.
  --manual-csv        : CSV with columns [iri,youtube_id] to force-add videos.

Writes additive TTL (default: kg/26_youtube_links.ttl).
Requires: rdflib, requests
"""

import argparse, csv, json, re, time, random
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

# Wikidata properties
P_YT_VIDEO   = "P1651"  # YouTube video ID
P_YT_CHANNEL = "P2397"  # YouTube channel ID

# MusicBrainz → Wikidata MBID properties (for resolution)
MB_PROPS = {"artist": "P434", "work": "P435", "recording": "P4404"}

# Caches
CACHE_ENTITY = Path("kg/enrichment/cache/wikidata_youtube"); CACHE_ENTITY.mkdir(parents=True, exist_ok=True)
CACHE_QID    = Path("kg/enrichment/cache/wdqs_qid");          CACHE_QID.mkdir(parents=True, exist_ok=True)

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
    Return { node: { 'kind': 'work|artist|recording|unknown', 'qid'?, 'mbid'?, 'mbkind'? } }
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

    # Discover QIDs and MBIDs
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
        if isinstance(v, str): vals.append(v)
        elif isinstance(v, dict) and "id" in v: vals.append(v["id"])
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

    from_props = MB_PROPS.get(kind_hint or "", None)
    qid = run_query([from_props]) if from_props else None
    if not qid: qid = run_query(list(MB_PROPS.values()))
    ck.write_text(json.dumps({"qid": qid}), encoding="utf-8")
    return qid

def add_video(out: Graph, subj: URIRef, yt_id: str, name: Optional[str] = None):
    vid = URIRef(str(subj) + "_YT_" + re.sub(r"[^A-Za-z0-9_-]", "", yt_id)[:32])
    out.add((vid, RDF.type, SCHEMA.VideoObject))
    out.add((vid, SCHEMA.embedUrl, URIRef(f"https://www.youtube.com/watch?v={yt_id}")))
    out.add((vid, SCHEMA.url,      URIRef(f"https://www.youtube.com/watch?v={yt_id}")))
    out.add((vid, SCHEMA.provider, URIRef("https://www.youtube.com/")))
    if name:
        out.add((vid, SCHEMA.name, Literal(name)))
    out.add((subj, SCHEMA.video, vid))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input TTL files")
    ap.add_argument("--out", default="kg/26_youtube_links.ttl")
    ap.add_argument("--resolve-by-mbid", action="store_true", help="Resolve QIDs from MusicBrainz IDs if missing")
    ap.add_argument("--include-artists", action="store_true", help="Also add YouTube channel links for artists (P2397)")
    ap.add_argument("--manual-csv", help="CSV with columns: iri,youtube_id[,name]")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = load_graphs(args.inputs)
    targets = collect_targets(base)

    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("ex", EX); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)

    used_qids = resolved_qids = 0
    songs_added = channels_added = 0

    # Manual CSV first (force-add without Wikidata)
    if args.manual_csv and Path(args.manual_csv).exists():
        with open(args.manual_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iri = URIRef(row["iri"])
                yt  = row["youtube_id"].strip()
                nm  = row.get("name") or None
                if yt:
                    add_video(out, iri, yt, nm)
                    songs_added += 1

    # WD-based enrichment
    for node, info in targets.items():
        kind = info.get("kind")
        qid  = info.get("qid")
        if not qid and args.resolve_by_mbid and info.get("mbid"):
            qid = wdqs_qid_from_mbid(info["mbid"], info.get("mbkind") or kind)
            if qid: resolved_qids += 1
        if not qid:
            continue

        used_qids += 1
        ent = fetch_entity(qid)
        if not ent:
            continue

        # Works → P1651
        if kind == "work":
            for vid in extract_claim_strings(ent, P_YT_VIDEO):
                vid_id = vid.rsplit("v=",1)[-1] if "http" in vid else vid
                add_video(out, node, vid_id)
                songs_added += 1

        # Artists → P2397 (optional)
        if args.include_artists and kind == "artist":
            for ch in extract_claim_strings(ent, P_YT_CHANNEL):
                ch_id = ch.rsplit("/",1)[-1] if "http" in ch else ch
                out.add((node, SCHEMA.sameAs, URIRef(f"https://www.youtube.com/channel/{ch_id}")))
                channels_added += 1

        # Re-assert sameAs to QID (handy downstream)
        out.add((node, SCHEMA.sameAs, URIRef(WD_ITEM + qid)))
        time.sleep(0.15)

    if args.dry_run:
        print(f"[dry-run] nodes={len(targets)} | used_qids={used_qids} | qid_resolved_by_mbid={resolved_qids} | videos_added={songs_added} | channels_added={channels_added}")
        return

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[write] {args.out} (triples={len(out)})")
    print(f"[stats] used_qids={used_qids} | qid_resolved_by_mbid={resolved_qids} | videos_added={songs_added} | channels_added={channels_added}")

if __name__ == "__main__":
    main()
