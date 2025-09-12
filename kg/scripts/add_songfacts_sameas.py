#!/usr/bin/env python3
"""
Add Songfacts artist pages to schema:sameAs.

- Input:  20_artists.ttl (your existing KG slice)
- Output: 20_artists_songfacts.ttl (new triples only)
         songfacts_artists_log.csv (summary log)

Usage:
  python add_songfacts_sameas.py --in 20_artists.ttl --out-ttl 20_artists_songfacts.ttl --log songfacts_artists_log.csv
"""

import argparse
import csv
import re
import time
import unicodedata
from pathlib import Path

import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

# ---------- Config ----------
USER_AGENT = "Wembrewind-KG/1.0 (+contact: you@example.com)"
TIMEOUT = 12
DELAY_SEC = 0.6   # be polite
VERIFY_FACTS = True  # make a GET and check for “facts” heuristic
# ----------------------------

SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX = Namespace("http://wembrewind.live/ex#")

SONGFACTS_BASE = "https://www.songfacts.com/facts/"

def slugify_artist(name: str) -> str:
    """Lowercase, ASCII-fold, replace non-alnum with hyphen, collapse/trim."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    n = n.lower()
    n = re.sub(r"[^a-z0-9]+", "-", n)
    n = re.sub(r"-{2,}", "-", n).strip("-")
    return n

def build_songfacts_url(name: str) -> str:
    return SONGFACTS_BASE + slugify_artist(name)

def _req(method, url):
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    return requests.request(method, url, headers=headers, allow_redirects=True, timeout=TIMEOUT)

def url_exists(url: str) -> bool:
    try:
        r = _req("HEAD", url)
        if r.status_code in (405, 403):  # some sites block HEAD
            r = _req("GET", url)
        return 200 <= r.status_code < 400
    except requests.RequestException:
        return False

def looks_like_facts_page(url: str) -> bool:
    """Light heuristic: title includes 'Songfacts', or body has 'facts' blocks."""
    try:
        r = _req("GET", url)
        if not (200 <= r.status_code < 300):
            return False
        html = r.text.lower()
        # heuristics: page title mentions songfacts; or body has 'songfacts' and 'facts' markers
        if "<title>" in html and "songfacts" in html:
            return True
        # common patterns
        patterns = [
            r"class=\".*?facts.*?\"",
            r"id=\"facts\"",
            r">facts<",
            r"songfacts\u00ae",  # Songfacts®
            r"songfacts®",
        ]
        return any(re.search(p, html) for p in patterns)
    except requests.RequestException:
        return False

def get_artist_names(g: Graph, artist_uri: URIRef):
    """Yield candidate names: schema:name first, then rdfs:label, then any alternates."""
    names = []
    for _, _, n in g.triples((artist_uri, SCHEMA.name, None)):
        if isinstance(n, Literal):
            names.append(str(n))
    for _, _, n in g.triples((artist_uri, RDFS.label, None)):
        if isinstance(n, Literal):
            names.append(str(n))
    for _, _, n in g.triples((artist_uri, SCHEMA.alternateName, None)):
        if isinstance(n, Literal):
            names.append(str(n))
    # de-dup preserving order
    seen = set()
    out = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

def already_has_sameas(g: Graph, artist_uri: URIRef, url: str) -> bool:
    return (artist_uri, SCHEMA.sameAs, URIRef(url)) in g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_ttl", required=True, help="Input TTL (20_artists.ttl)")
    ap.add_argument("--out-ttl", dest="out_ttl", default="20_artists_songfacts.ttl", help="Output TTL with new sameAs triples")
    ap.add_argument("--log", dest="log_csv", default="songfacts_artists_log.csv", help="CSV log output")
    ap.add_argument("--no-verify-facts", action="store_true", help="Skip GET-based content check")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of artists processed (0 = all)")
    args = ap.parse_args()

    verify = not args.no_verify_facts

    src = Graph()
    src.parse(args.in_ttl, format="turtle")

    outg = Graph()
    outg.bind("schema", SCHEMA)
    outg.bind("mm", MM)
    outg.bind("ex", EX)

    targets = set()
    # mm:MusicArtist
    for s in src.subjects(RDF.type, MM.MusicArtist):
        targets.add(s)
    # mm:Musician (capitalization varies across graphs sometimes; use URI)
    for s in src.subjects(RDF.type, MM.Musician):
        targets.add(s)

    rows = []
    processed = 0
    added = 0

    for artist in sorted(targets, key=lambda x: str(x)):
        if args.limit and processed >= args.limit:
            break
        processed += 1

        names = get_artist_names(src, artist)
        status = "skipped_no_name"
        chosen_name = ""
        url = ""
        exists = False
        facts_ok = False
        note = ""

        if not names:
            rows.append([str(artist), "", "", "", False, False, "no name candidates"])
            continue

        # try each name until one works
        for nm in names:
            candidate = build_songfacts_url(nm)
            if url_exists(candidate):
                url = candidate
                exists = True
                chosen_name = nm
                if verify:
                    facts_ok = looks_like_facts_page(candidate)
                else:
                    facts_ok = True
                break
            time.sleep(DELAY_SEC)

        if exists and facts_ok:
            if not already_has_sameas(src, artist, url):
                outg.add((artist, SCHEMA.sameAs, URIRef(url)))
                added += 1
                status = "added"
            else:
                status = "already_present"
        else:
            if exists and not facts_ok:
                status = "failed_content_check"
            else:
                status = "no_url_found"

        rows.append([str(artist), chosen_name, url, status, exists, facts_ok, note])
        time.sleep(DELAY_SEC)

    # write outputs
    out_path = Path(args.out_ttl)
    outg.serialize(destination=str(out_path), format="turtle")

    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["artist_uri", "chosen_name", "songfacts_url", "status", "url_exists", "facts_ok", "note"])
        w.writerows(rows)

    print(f"Artists processed: {processed}")
    print(f"New sameAs triples added: {added}")
    print(f"Wrote TTL: {out_path}")
    print(f"Wrote log: {args.log_csv}")

if __name__ == "__main__":
    main()
