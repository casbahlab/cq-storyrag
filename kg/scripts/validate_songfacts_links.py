#!/usr/bin/env python3
import re
import unicodedata
from pathlib import Path
import requests
from rdflib import Graph, Namespace, URIRef

SCHEMA = Namespace("http://schema.org/")
USER_AGENT = "Wembrewind-KG/1.0"
TIMEOUT = 10

def url_valid(url: str) -> bool:
    """Check if Songfacts URL works and looks like a facts page."""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, allow_redirects=True)
        if not (200 <= r.status_code < 400):
            return False
        # Ensure it’s still under /facts/
        if not r.url.startswith("https://www.songfacts.com/facts/"):
            return False
        html = r.text.lower()
        # Check for title containing 'songfacts'
        if "<title>" in html and "songfacts" in html:
            return True
        # Or common patterns in body
        patterns = [
            r"class=\".*?facts.*?\"",
            r"id=\"facts\"",
            r">facts<",
            r"songfacts®",
            r"songfacts\u00ae",
        ]
        return any(re.search(p, html) for p in patterns)
    except requests.RequestException:
        return False

def main():
    in_ttl = Path("../20_artists_songfacts.ttl")
    out_ttl = Path("../20_artists_songfacts.ttl")

    g = Graph()
    g.parse(in_ttl, format="turtle")

    outg = Graph()
    outg.bind("schema", SCHEMA)

    kept = 0
    removed = 0

    for s, p, o in g.triples((None, SCHEMA.sameAs, None)):
        if isinstance(o, URIRef) and "songfacts.com/facts/" in str(o):
            if url_valid(str(o)):
                outg.add((s, p, o))
                kept += 1
            else:
                removed += 1
        else:
            # Keep any sameAs triple that is not Songfacts
            outg.add((s, p, o))
            kept += 1

    outg.serialize(destination=out_ttl, format="turtle")
    print(f"Kept: {kept} triples, Removed: {removed} Songfacts links")
    print(f"Validated TTL written to {out_ttl}")

if __name__ == "__main__":
    main()
