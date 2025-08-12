#!/usr/bin/env python3
"""
Pull performing artists for a MusicBrainz Event and emit artist + LivePerformance triples.

- Uses musicbrainzngs (pip install musicbrainzngs)
- Idempotent: safe to re-run; it won't duplicate existing triples.
- Writes to:
    kg/20_artists.ttl
    kg/30_performances.ttl
- Keeps clean prefixes (ex, schema, mm); normalizes schema.org to http://

Usage example:
  python kg/scripts/enrich_from_mb_event.py \
    --event-mbid 08657c50-71c6-4a3d-b7c4-3db6efbf07fd \
    --parent-event ex:LiveAid1985_Wembley \
    --venue ex:WembleyStadium \
    --out kg \
    --contact you@example.com \
    --app WembrewindKG/1.0

You can do a dry-run first:
  ... --dry-run
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# ---- Namespaces
SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")
OWL    = Namespace("http://www.w3.org/2002/07/owl#")


import re
from rdflib import URIRef

EX_BASE = "http://wembrewind.live/ex#"

def pascal_slug(text: str) -> str:
    # Keep letters/digits, split on anything else, PascalCase join
    parts = re.split(r"[^A-Za-z0-9]+", text.strip())
    slug = "".join(p.capitalize() for p in parts if p)
    if not slug:
        slug = "Entity"
    if slug[0].isdigit():
        slug = "A" + slug  # IRIs can’t start with a digit (safe guard)
    return slug

def local_name_from_ex(iri: URIRef) -> str:
    s = str(iri)
    return s.split("#", 1)[1] if "#" in s else s.rsplit("/", 1)[-1]

def ensure_unique_subject(g, base_local: str) -> URIRef:
    """
    Ensure ex:{base_local} is unique in graph; if taken, append _2, _3...
    Deterministic and stable across re-runs.
    """
    candidate = URIRef(f"{EX_BASE}{base_local}")
    if (candidate, None, None) not in g and (None, None, candidate) not in g:
        return candidate
    i = 2
    while True:
        cand = URIRef(f"{EX_BASE}{base_local}_{i}")
        if (cand, None, None) not in g and (None, None, cand) not in g:
            return cand
        i += 1

# ---- Helpers for RDF graph hygiene
def bind_prefixes(g: Graph):
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("mm", MM)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)

def normalize_schema_http(g: Graph):
    """Unify any https://schema.org/ IRIs to http:// to avoid schema1: surprises."""
    to_add, to_del = [], []
    for s, p, o in g:
        p2, o2, s2 = p, o, s
        if isinstance(p, URIRef) and str(p).startswith("https://schema.org/"):
            p2 = URIRef(str(p).replace("https://", "http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o2 = URIRef(str(o).replace("https://", "http://"))
        if isinstance(s, URIRef) and str(s).startswith("https://schema.org/"):
            s2 = URIRef(str(s).replace("https://", "http://"))
        if (s2, p2, o2) != (s, p, o):
            to_del.append((s, p, o))
            to_add.append((s2, p2, o2))
    for t in to_del: g.remove(t)
    for t in to_add: g.add(t)

def load_or_new(path: Path) -> Graph:
    g = Graph()
    if path.exists():
        g.parse(path, format="turtle")
    bind_prefixes(g)
    normalize_schema_http(g)
    return g

def save_graph(g: Graph, path: Path):
    bind_prefixes(g)
    normalize_schema_http(g)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(path), format="turtle")
    print(f"[write] {path}  (triples: {len(g)})")

# ---- IRI minting
def ex_artist_uri(mbid: str) -> URIRef:
    # stable, collision-free; human-readable labels are in schema:name
    return URIRef(f"{EX}artist/mbid/{mbid}")

def ex_performance_uri(event_mbid: str, artist_mbid: str) -> URIRef:
    return URIRef(f"{EX}performance/event/{event_mbid}/artist/{artist_mbid}")

def ex_event_uri_from_prefix(prefixed: str) -> URIRef:
    # expects something like "ex:LiveAid1985_Wembley"
    if prefixed.startswith("ex:"):
        return URIRef(f"{EX}{prefixed[3:]}")
    return URIRef(prefixed)

# ---- MusicBrainz
def mb_client(user_app: str, contact: str):
    import musicbrainzngs as mb
    ua = user_app or "WembrewindKG/0.1"
    mb.set_useragent(ua, version="0.1", contact=contact)
    mb.set_rate_limit(True)
    return mb

def fetch_event_relations(mb, event_mbid: str) -> Dict:
    ev = mb.get_event_by_id(event_mbid, includes=["artist-rels", "place-rels"])["event"]
    return ev

def fetch_artist(mb, artist_mbid: str) -> Dict:
    return mb.get_artist_by_id(artist_mbid)["artist"]

# ---- RDF emitters
def ensure_artist_triples(artists_g, sameas_g, artist_json):
    mbid = artist_json.get("id")
    name = artist_json.get("name") or mbid
    a_type = artist_json.get("type")

    # NEW: human-friendly subject IRI
    base_local = pascal_slug(name)              # e.g., "FreddieMercury"
    s = ensure_unique_subject(artists_g, base_local)

    created = (s, RDF.type, None) not in artists_g

    # Types
    if a_type == "Group":
        artists_g.add((s, RDF.type, SCHEMA.MusicGroup))
        artists_g.add((s, RDF.type, MM.MusicArtist))
    else:
        artists_g.add((s, RDF.type, SCHEMA.Person))
        artists_g.add((s, RDF.type, MM.Musician))

    # Labels & IDs
    artists_g.set((s, SCHEMA.name, Literal(name)))

    # Keep MBID as identifier (and dereferenceable MB URL via owl:sameAs)
    if mbid:
        artists_g.set((s, OWL.sameAs, URIRef(f"https://musicbrainz.org/artist/{mbid}")))

    # ... keep the rest (sort-name, disambiguation, gender, areas, isni/ipi) unchanged
    return s, created

def ex_performance_uri_from_name(perfs_g, parent_event_iri: URIRef, artist_uri: URIRef) -> URIRef:
    artist_local = local_name_from_ex(artist_uri)         # e.g., "FreddieMercury"
    event_tag   = local_name_from_ex(parent_event_iri)    # e.g., "LiveAid1985"
    base_local  = f"{artist_local}{event_tag}Performance" # "FreddieMercuryLiveAid1985Performance"
    return ensure_unique_subject(perfs_g, base_local)


def ensure_performance_triples(perf_g, event_mbid, parent_event_iri, venue_iri,
                               artist_uri, artist_name, event_name, event_date_iso):
    p = ex_performance_uri_from_name(perf_g, parent_event_iri, artist_uri)
    perf_g.add((p, RDF.type, MM.LivePerformance))
    perf_g.set((p, MM.isCreatedBy, artist_uri))
    perf_g.set((p, SCHEMA.isPartOf, parent_event_iri))
    if venue_iri:
        perf_g.set((p, SCHEMA.location, venue_iri))
    # label
    if artist_name or event_name:
        label = " — ".join([x for x in [artist_name, event_name] if x])
        perf_g.set((p, SCHEMA.name, Literal(label)))
    if event_date_iso:
        perf_g.set((p, SCHEMA.startDate, Literal(event_date_iso, datatype=XSD.date)))
    # Keep sameAs to the MB event URI if you like
    if event_mbid:
        perf_g.add((p, OWL.sameAs, URIRef(f"https://musicbrainz.org/event/{event_mbid}")))
    return p

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, XSD

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")
OWL    = Namespace("http://www.w3.org/2002/07/owl#")

def normalize_schema_http_term(t):
    if isinstance(t, URIRef) and str(t).startswith("https://schema.org/"):
        return URIRef(str(t).replace("https://schema.org/", "http://schema.org/"))
    return t

def fresh_graph_with_prefixes():
    g = Graph()
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("mm", MM)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    return g

def save_graph_clean(g_in: Graph,  path: Path):
    # 1) write into a fresh graph (so no inherited weird bindings)
    g = fresh_graph_with_prefixes()

    for s,p,o in g_in:
        s2 = normalize_schema_http_term(s)
        p2 = normalize_schema_http_term(p)
        o2 = normalize_schema_http_term(o)
        g.add((s2,p2,o2))
    # 2) ensure no lingering bindings
    nm = g.namespace_manager
    nm.bind("schema1", None, replace=True)
    nm.bind("schema",  None, replace=True)
    g.bind("schema", SCHEMA)
    # 3) serialize
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(str(path), format="turtle")
    print(f"[write] {path} (triples: {len(g)})")


# ---- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-mbid", required=True, help="MusicBrainz Event MBID (e.g., 08657c50-...)")
    ap.add_argument("--parent-event", required=True, help="Your event IRI or ex:CURIE (e.g., ex:LiveAid1985_Wembley)")
    ap.add_argument("--venue", help="Your venue IRI or ex:CURIE (e.g., ex:WembleyStadium)")
    ap.add_argument("--out", default="kg", help="Root KG folder (default: kg)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--app", default="WembrewindKG/1.0", help="User-Agent app name for MB")
    ap.add_argument("--contact", required=True, help="Contact email/URL for MusicBrainz UA (required)")
    args = ap.parse_args()

    out_root = Path(args.out)
    artists_ttl = out_root / "20_artists.ttl"
    perfs_ttl   = out_root / "30_performances.ttl"
    sameas_ttl  = out_root / "81_links_sameAs.ttl"

    # Load graphs (create if missing)
    artists_g = load_or_new(artists_ttl)
    perfs_g   = load_or_new(perfs_ttl)
    sameas_g  = load_or_new(sameas_ttl)

    parent_event_iri = ex_event_uri_from_prefix(args.parent_event)
    venue_iri = ex_event_uri_from_prefix(args.venue) if args.venue else None

    # MusicBrainz client
    mb = mb_client(args.app, args.contact)
    ev = fetch_event_relations(mb, args.event_mbid)

    #print(f"[fetch] MB Event: {args.event_mbid} — {ev.get('name', 'unknown')}")

    event_name = ev.get("name")
    # Try to pick YYYY-MM-DD from event's time fields if present
    event_date_iso = None
    if "time" in ev:
        # MB "time" is HH:MM; "begin" might be date; prefer begin
        pass
    if ev.get("begin"):
        event_date_iso = ev["begin"][:10]  # '1985-07-13'

    # Extract performer relations
    rels = ev.get("artist-relation-list", [])
    PERF_TYPES = {"main performer"}
    performers = [r for r in rels if r.get("type") in PERF_TYPES and "artist" in r]
    created_artists = 0
    created_perfs = 0

    for r in performers:
        a = r["artist"]
        artist_mbid = a.get("id")
        if not artist_mbid:
            continue

        # Fetch full artist (to get reliable 'type' and canonical name)
        art = fetch_artist(mb, artist_mbid)
        #print(f"[fetch] MB Artist: {artist_mbid} — {art}")
        artist_uri, created = ensure_artist_triples(artists_g, sameas_g, art)
        if created:
            created_artists += 1

        # credited-as → alternateName (optional)
        credited_as = r.get("attribute-credits") or r.get("credited-as")
        if credited_as:
            artists_g.add((artist_uri, SCHEMA.alternateName, Literal(credited_as)))

        # Make performance node
        perf_uri = ensure_performance_triples(perfs_g,
                                              event_mbid=args.event_mbid,
                                              parent_event_iri=parent_event_iri,
                                              venue_iri=venue_iri,
                                              artist_uri=artist_uri,
                                              artist_name=art.get("name"),
                                              event_name=event_name,
                                              event_date_iso=event_date_iso)
        # Link performance back to MB event (optional)
        perfs_g.add((perf_uri, OWL.sameAs, URIRef(f"https://musicbrainz.org/event/{args.event_mbid}")))
        created_perfs += 1  # we don't check existence deeply; simplest counter

    # Summaries
    print(f"[event] {event_name or args.event_mbid} — performers: {len(performers)}")
    print(f"[artists] created/updated: {created_artists}")
    print(f"[performances] created: {created_perfs}")

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    save_graph_clean(artists_g, artists_ttl)
    save_graph_clean(perfs_g, perfs_ttl)
    save_graph_clean(sameas_g, sameas_ttl)

if __name__ == "__main__":
    main()
