
#!/usr/bin/env python3
"""
enrich_20_artists_literals.py

Append-only enrichment for 20_artists.ttl:
- MusicGroup  -> schema:foundingDate (literal), schema:foundingLocation (literal)
- Person/Musician -> schema:birthDate (literal), schema:birthPlace (literal), optional schema:gender (literal)

Notes:
- Detects artist classes regardless of whether the file uses "schema:" or "schema1:".
- Reads MBID from schema:sameAs link to musicbrainz.org.
- Stores places as plain text literals (no new nodes).
"""

import argparse
import re
import sys
import time
from urllib.parse import urlparse

from rdflib import Graph, Namespace, URIRef, Literal, OWL
from rdflib.namespace import RDF, XSD

import musicbrainzngs as mb

SCHEMA = Namespace("http://schema.org/")
SCHEMA1 = Namespace("http://schema.org/")  # alias; user uses schema1
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

MB_URL_RE = re.compile(r"https?://(www\.)?musicbrainz\.org/artist/([0-9a-f-]{36})/?$")

def setup_mb():
    mb.set_useragent("wembrewind-kg", "1.0", "https://github.com/casbahlab/comp70225-wembrewind")

from typing import Optional

def extract_mbid(g: Graph, artist: URIRef) -> Optional[str]:
    for _, _, same in g.triples((artist, OWL.sameAs, None)):
        m = MB_URL_RE.match(str(same))
        if m:
            return m.group(2)
    # also check "schema1:sameAs" if present in input
    for _, _, same in g.triples((artist, SCHEMA1.sameAs, None)):
        m = MB_URL_RE.match(str(same))
        if m:
            return m.group(2)
    return None

def coerce_date_literal(val: str) -> Literal:
    if not val:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
        return Literal(val, datatype=XSD.date)
    # allow YYYY or YYYY-MM
    return Literal(val)

def compose_place_literal(begin_area: dict) -> Optional[str]:
    """
    MusicBrainz "begin-area" often only has "name" and maybe "disambiguation".
    We'll compose a readable literal using what's available.
    """
    if not begin_area:
        return None
    name = begin_area.get("name")
    disamb = begin_area.get("disambiguation")
    if name and disamb:
        return f"{name} ({disamb})"
    return name or None

def fetch_artist(mbid: str) -> Optional[dict]:
    try:
        # include areas so we get begin-area content
        return mb.get_artist_by_id(mbid).get("artist")
    except Exception as e:
        print(f"[warn] MB fetch failed for {mbid}: {e}", file=sys.stderr)
        return None

def is_group(g: Graph, s: URIRef) -> bool:
    return ((s, RDF.type, SCHEMA.MusicGroup) in g) or ((s, RDF.type, SCHEMA1.MusicGroup) in g)

def is_person(g: Graph, s: URIRef) -> bool:
    # Consider schema:MusicArtist, schema:Person, mm:Musician (as per user's data)
    return ((s, RDF.type, SCHEMA.MusicArtist) in g) or \
           ((s, RDF.type, SCHEMA1.MusicArtist) in g) or \
           ((s, RDF.type, SCHEMA.Person) in g) or \
           ((s, RDF.type, SCHEMA1.Person) in g) or \
           ((s, RDF.type, MM.Musician) in g)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl-in", required=True, help="Input 20_artists.ttl")
    ap.add_argument("--ttl-out", required=True, help="Output enriched TTL")
    ap.add_argument("--sleep", type=float, default=0.8, help="Delay between MB requests")
    ap.add_argument("--dry-run", action="store_true", help="Print planned changes without writing TTL")
    args = ap.parse_args()

    setup_mb()

    g = Graph()
    # Bind both schema and schema1, but we will serialize using schema1 prefix
    g.bind("schema", SCHEMA)
    g.bind("schema1", SCHEMA1)
    g.bind("mm", MM)
    g.parse(args.ttl_in, format="turtle")

    # We'll append triples to this same graph and write it out to a new file
    updates = 0

    # Iterate over all artist-like nodes in the file
    subjects = set(s for s, _, _ in g.triples((None, RDF.type, None)))
    for s in subjects:
        is_g = is_group(g, s)
        is_p = is_person(g, s)
        if not (is_g or is_p):
            continue

        mbid = extract_mbid(g, s)
        if not mbid:
            print(f"[skip] No MusicBrainz link for {s}", file=sys.stderr)
            continue

        data = fetch_artist(mbid)
        if not data:
            continue

        ls = data.get("life-span") or {}
        begin = ls.get("begin")

        begin_area = data.get("begin-area") or {}
        place_literal = compose_place_literal(begin_area)

        if is_g:
            # foundingDate
            lit = coerce_date_literal(begin)
            if lit is not None:
                g.add((s, SCHEMA.foundingDate, lit))
                g.add((s, SCHEMA1.foundingDate, lit))  # ensure presence under schema1 as well
                updates += 1
            # foundingLocation as plain literal
            if place_literal:
                g.add((s, SCHEMA.foundingLocation, Literal(place_literal)))
                g.add((s, SCHEMA1.foundingLocation, Literal(place_literal)))
                updates += 1

        if is_p:
            # birthDate
            lit = coerce_date_literal(begin)
            if lit is not None:
                g.add((s, SCHEMA.birthDate, lit))
                g.add((s, SCHEMA1.birthDate, lit))
                updates += 1
            # birthPlace as plain literal
            if place_literal:
                g.add((s, SCHEMA.birthPlace, Literal(place_literal)))
                g.add((s, SCHEMA1.birthPlace, Literal(place_literal)))
                updates += 1
            # gender
            gender = data.get("gender")
            if gender:
                g.add((s, SCHEMA.gender, Literal(gender.title())))
                g.add((s, SCHEMA1.gender, Literal(gender.title())))
                updates += 1

        time.sleep(args.sleep)

    if args.dry_run:
        print(f"[dry-run] Would add ~{updates} triples")
        return

    # Serialize using schema1 as the visible prefix for http://schema.org/
    # rdflib will keep both bindings; that's fine. The user uses schema1 in their files.
    g.serialize(destination=args.ttl_out, format="turtle")
    print(f"[ok] Wrote enriched TTL: {args.ttl_out}")
    print(f"[ok] Added ~{updates} triples")

if __name__ == "__main__":
    main()
