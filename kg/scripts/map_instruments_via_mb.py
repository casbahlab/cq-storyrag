#!/usr/bin/env python3
"""
Map instruments in a TTL to MusicBrainz and Wikidata by **exact** instrument name,
and also record the **Wikidata QID** when present.

Workflow
1) Read subjects of a given rdf:type from your TTL (e.g., ex:MusicInstrument).
2) For each entity, read its label (rdfs:label or schema:name).
3) Query MusicBrainz for an **exact** instrument name match.
4) If found, fetch the instrument with `inc=url-rels` and grab the Wikidata URL relation.
5) Extract the **QID** from the Wikidata URL.
6) Write schema:sameAs triples for both MB and WD into a new TTL, and output a CSV report.

Install
    pip install rdflib requests

Usage
    python map_instruments_via_mb.py \
      --in-ttl 50_instruments.ttl \
      --out-ttl 50_instruments_with_links.ttl \
      --out-csv instrument_link_map.csv \
      --type-uri http://wembrewind.live/ex#MusicInstrument \
      --label-order rdfs,schema \
      --sameas-prop schema \
      --mb-useragent "wembrewind-linker/0.1 (you@example.com)" \
      --sleep 1.2 \
      [--overrides overrides.csv]

Override CSV
  label,musicbrainz_id
  Bassguitar,17f9f065-2312-4a24-8309-6f6dd63e2e33
"""

import argparse
import csv
import re
import time
from typing import Dict, Optional, Tuple

import requests
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL, SKOS

SCHEMA = Namespace("http://schema.org/")

MB_SEARCH = "https://musicbrainz.org/ws/2/instrument"
MB_LOOKUP = "https://musicbrainz.org/ws/2/instrument/{mbid}"

def load_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "musicbrainz_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("overrides CSV must have columns: label,musicbrainz_id")
        for row in reader:
            label = (row.get("label") or "").strip()
            mbid = (row.get("musicbrainz_id") or "").strip()
            if label and mbid:
                out[label.lower()] = mbid
    return out

def get_label(g: Graph, s: URIRef, order=("rdfs","schema")) -> str:
    if "schema" in order:
        for o in g.objects(s, SCHEMA.name):
            return str(o)
    if "rdfs" in order:
        for o in g.objects(s, RDFS.label):
            return str(o)
    # Fallback to fragment
    uri = str(s)
    if "#" in uri:
        return uri.split("#", 1)[1]
    return uri.rsplit("/", 1)[-1]

def mb_search_exact(name: str, session: requests.Session) -> Optional[Tuple[str, int]]:
    """Return (mbid, score) for an exact-name instrument if found, else None.
    Uses case-insensitive equality on `name`. Picks highest score among exact matches.
    """
    params = {"query": f'instrument:"{name}"', "fmt": "json"}
    resp = session.get(MB_SEARCH, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    best = None
    for inst in data.get("instruments", []):
        inst_name = inst.get("name") or ""
        if inst_name.lower() == name.lower():
            score = int(inst.get("score", 0))
            mbid = inst.get("id")
            if mbid:
                if not best or score > best[1]:
                    best = (mbid, score)
    return best

def mb_fetch_wikidata_url(mbid: str, session: requests.Session) -> Optional[str]:
    """Fetch Wikidata URL from MB instrument relations if present."""
    params = {"inc": "url-rels", "fmt": "json"}
    resp = session.get(MB_LOOKUP.format(mbid=mbid), params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    for rel in data.get("relations", []):
        # Relation type is usually 'wikidata'; guard by URL domain as well
        if rel.get("type", "").lower() == "wikidata":
            url = rel.get("url", {}).get("resource")
            print(f"url : {url}")
            if url and "wikidata.org/" in url:
                return url
        # Fallback: any URL with wikidata domain
        url = rel.get("url", {}).get("resource")
        if url and "wikidata.org/entity/" in url:
            return url
    return None

_QID_RE = re.compile(r"/entity/(Q[0-9]+)$", re.IGNORECASE)

def extract_qid(wd_url: Optional[str]) -> str:
    """Extract QID (e.g., Q31561) from wikidata URL; return '' if not parseable."""
    if not wd_url:
        return ""
    m = _QID_RE.search(wd_url.strip())
    return m.group(1).upper() if m else ""

def add_if_absent(g: Graph, s: URIRef, p: URIRef, o: URIRef) -> bool:
    if (s, p, o) in g:
        return False
    g.add((s, p, o))
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ttl", required=True)
    ap.add_argument("--out-ttl", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--type-uri", required=True, help="rdf:type URI for instrument entities")
    ap.add_argument("--label-order", default="rdfs,schema", help="comma list of label order preference")
    ap.add_argument("--sameas-prop", choices=["schema", "owl"], default="schema")
    ap.add_argument("--mb-useragent", required=True, help='MusicBrainz User-Agent, e.g. "app/1.0 (email@example.com)"')
    ap.add_argument("--sleep", type=float, default=1.2, help="Seconds to sleep between MB requests")
    ap.add_argument("--overrides", help="CSV with columns: label,musicbrainz_id (forces MBID for label)")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_ttl, format="turtle")

    sameas_pred = SCHEMA.sameAs if args.sameas_prop == "schema" else OWL.sameAs
    label_order = tuple(part.strip() for part in args.label_order.split(","))

    overrides = load_overrides(args.overrides)

    session = requests.Session()
    session.headers.update({"User-Agent": args.mb_useragent})

    type_uri = URIRef(args.type_uri)

    rows = []
    processed = 0
    added = 0

    for s in g.subjects(RDF.type, type_uri):
        processed += 1
        label = get_label(g, s, order=label_order)
        label_key = label.lower()

        mbid = None
        mb_score = None

        if label_key in overrides:
            mbid = overrides[label_key]
        else:
            # Exact search
            try:
                res = mb_search_exact(label, session)
            except requests.HTTPError as e:
                print(f"[WARN] MB search failed for {label}: {e}")
                res = None
            if res:
                mbid, mb_score = res
            time.sleep(args.sleep)  # be nice to MB

        mb_url = f"https://musicbrainz.org/instrument/{mbid}" if mbid else ""

        # Fetch Wikidata from MB relations
        wd_url = ""
        wd_qid = ""
        if mbid:
            try:
                wdu = mb_fetch_wikidata_url(mbid, session) or ""
                wd_url = wdu
                wd_qid = extract_qid(wd_url)
            except requests.HTTPError as e:
                print(f"[WARN] MB lookup failed for {label} ({mbid}): {e}")
            time.sleep(args.sleep)

        # Add triples
        added_mb = False
        added_wd = False
        if mb_url:
            added_mb = add_if_absent(g, s, sameas_pred, URIRef(mb_url))
            if added_mb:
                added += 1
        if wd_url:
            added_wd = add_if_absent(g, s, sameas_pred, URIRef(wd_url))
            if added_wd:
                added += 1

        rows.append({
            "entity_uri": str(s),
            "label": label,
            "musicbrainz_id": mbid or "",
            "musicbrainz_url": mb_url,
            "mb_score": mb_score if mb_score is not None else "",
            "wikidata_qid": wd_qid,
            "wikidata_url": wd_url,
            "added_mb": "yes" if added_mb else "no",
            "added_wd": "yes" if added_wd else "no",
        })

    g.serialize(destination=args.out_ttl, format="turtle")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entity_uri","label",
            "musicbrainz_id","musicbrainz_url","mb_score",
            "wikidata_qid","wikidata_url",
            "added_mb","added_wd"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Processed {processed} entities. Added {added} triples.")
    print(f"Wrote TTL: {args.out_ttl}")
    print(f"Wrote CSV: {args.out_csv}")

if __name__ == "__main__":
    main()