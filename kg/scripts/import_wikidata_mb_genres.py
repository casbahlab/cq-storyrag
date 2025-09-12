#!/usr/bin/env python3
import argparse, requests, re
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")
EX_BASE = "http://wembrewind.live/ex#"

WD_ENTITY = "http://www.wikidata.org/entity/"
MB_GENRE  = "https://musicbrainz.org/genre/"

SPARQL = """
SELECT DISTINCT ?genre ?mbid (SAMPLE(?label) AS ?label) (SAMPLE(?anyLabel) AS ?anyLabel) (SAMPLE(?enwikiTitle) AS ?enwikiTitle)
WHERE {
  ?genre wdt:P8052 ?mbid .
  OPTIONAL { ?genre rdfs:label ?label FILTER (lang(?label) = "en") }
  OPTIONAL { ?genre rdfs:label ?anyLabel FILTER (lang(?anyLabel) != "") }
  OPTIONAL {
    ?enwiki_article schema:about ?genre ;
                    schema:isPartOf <https://en.wikipedia.org/> ;
                    schema:name ?enwikiTitle .
  }
}
GROUP BY ?genre ?mbid
"""
def best_label(row):
    # Order of preference: English label, any label, enwiki title, MBID
    lab = row.get("label", {}).get("value")
    if lab: return lab
    anylab = row.get("anyLabel", {}).get("value")
    if anylab: return anylab
    enwiki = row.get("enwikiTitle", {}).get("value")
    if enwiki: return enwiki
    return row["mbid"]["value"]  # last resort

def pascal_slug(text: str) -> str:
    import re, unicodedata
    # fold diacritics; keep letters/digits; PascalCase
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    parts = re.split(r"[^A-Za-z0-9]+", text.strip())
    slug = "".join(p.capitalize() for p in parts if p)
    if not slug: slug = "Genre"
    if slug[0].isdigit(): slug = "G" + slug
    return slug

def fetch_wikidata_genres(user_agent: str):
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": user_agent, "Accept": "application/sparql-results+json"}
    r = requests.get(url, params={"query": SPARQL, "format": "json"}, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()["results"]["bindings"]
    out = []
    for b in data:
        qid  = b["genre"]["value"]                        # e.g., https://www.wikidata.org/entity/Q187760
        mbid = b["mbid"]["value"]                         # e.g., 56407f9d-3398-4bf3-bbbd-ea372fa5adeb
        lab  = b["label"]["value"]
        out.append((qid, mbid, lab))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kg/20_genres.ttl")
    ap.add_argument("--ua", default="WembrewindKG/1.0 (you@example.com)")
    args = ap.parse_args()

    rows = fetch_wikidata_genres(args.ua)
    print(f"[wd] genres with MB IDs: {len(rows)}")

    g = Graph()
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("rdfs", RDFS); g.bind("owl", OWL)

    seen_mbids = set()
    for row in rows:
        print(f"row : {row}")
        qid_url = row[0]
        mb_uuid = row[1]
        label = row[2]
        if mb_uuid in seen_mbids:
            continue  # skip duplicate
        seen_mbids.add(mb_uuid)
        # local node (readable IRI from label; safe fallback to QID)
        local = URIRef(EX_BASE + pascal_slug(label) if label else EX_BASE + qid_url.rsplit("/",1)[-1])

        g.add((local, RDF.type, MM.MusicGenre))
        g.set((local, RDFS.label, Literal(label)))
        # cross-links
        g.add((local, OWL.sameAs, URIRef(qid_url)))
        g.add((local, OWL.sameAs, URIRef(MB_GENRE + mb_uuid)))
        # optional literal id
        g.set((local, URIRef(EX_BASE + "mbGenreId"), Literal(mb_uuid)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(args.out, format="turtle")
    print(f"[write] {args.out} (triples: {len(g)})")

if __name__ == "__main__":
    main()
