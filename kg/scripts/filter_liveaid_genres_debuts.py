#!/usr/bin/env python3
import argparse, re
from pathlib import Path
from typing import Set, Dict
from rdflib import Graph, Namespace, URIRef, RDF
from rdflib.namespace import OWL, XSD, RDFS

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

def load_graph(p: Path) -> Graph:
    g = Graph()
    if p.exists():
        g.parse(p, format="turtle")
    # normalize schema.org https->http just in case
    fixes = []
    for s,p,o in g:
        s2,p2,o2 = s,p,o
        if isinstance(p, URIRef) and str(p).startswith("https://schema.org/"):
            p2 = URIRef(str(p).replace("https://","http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o2 = URIRef(str(o).replace("https://","http://"))
        if (s2,p2,o2)!=(s,p,o): fixes.append(((s,p,o),(s2,p2,o2)))
    for (s,p,o),(s2,p2,o2) in fixes:
        g.remove((s,p,o)); g.add((s2,p2,o2))
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("owl", OWL); g.bind("xsd", XSD)
    return g

def save_graph(g: Graph, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(p), format="turtle")
    print(f"[write] {p} (triples: {len(g)})")

def get_kept_artists(artists_main_g: Graph) -> Set[URIRef]:
    return set(artists_main_g.subjects(RDF.type, SCHEMA.Person)) | set(artists_main_g.subjects(RDF.type, SCHEMA.MusicGroup))

# ---------- GENRES (unchanged from before) ----------
def filter_genres(artists_main_g: Graph, genres_g: Graph) -> Graph:
    kept_artists = get_kept_artists(artists_main_g)
    preds = set()
    if (None, MM.hasGenre, None) in artists_main_g:
        preds.add(MM.hasGenre)
    else:
        # Any predicate from kept artist → node existing in genres graph
        genre_nodes = set(s for s,_,_ in genres_g)
        for a in kept_artists:
            for p,o in artists_main_g.predicate_objects(a):
                if isinstance(o, URIRef) and (o, None, None) in genres_g:
                    preds.add(p)
    kept_genres: Set[URIRef] = set()
    for a in kept_artists:
        for p in preds:
            for g in artists_main_g.objects(a, p):
                if isinstance(g, URIRef) and (g, None, None) in genres_g:
                    kept_genres.add(g)

    out = Graph(); [out.bind(pfx, ns) for pfx,ns in genres_g.namespaces()]
    for s,p,o in genres_g:
        if s in kept_genres:
            out.add((s,p,o))
    print(f"[genres] artists={len(kept_artists)} preds={len(preds)} kept_genres={len(kept_genres)}")
    return out

# ---------- DEBUT ALBUMS (fixed) ----------
ALBUM_TYPE_CANDIDATES = {
    SCHEMA.MusicAlbum,
    URIRef(str(MM) + "Album"),
    URIRef(str(MM) + "Release"),
    URIRef(str(MM) + "ReleaseGroup"),
}

def is_album_type(t: URIRef) -> bool:
    if t in ALBUM_TYPE_CANDIDATES:
        return True
    return isinstance(t, URIRef) and re.search(r"(album|releasegroup|release)$", str(t), re.I) is not None

def discover_album_nodes(debut_g: Graph) -> Set[URIRef]:
    albums: Set[URIRef] = set()
    # Subjects with hasPart (album → parts)
    for s in debut_g.subjects(SCHEMA.hasPart, None):
        if isinstance(s, URIRef):
            albums.add(s)
    # Objects of inAlbum (recording → album)
    for o in debut_g.objects(None, SCHEMA.inAlbum):
        if isinstance(o, URIRef):
            albums.add(o)
    # Typed like albums
    for s,t in debut_g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and is_album_type(t):
            albums.add(s)
    return albums

# --- replace your filter_debuts() with this version ---

ALBUM_TYPE_CANDIDATES = {
    SCHEMA.MusicAlbum,
    URIRef(str(MM) + "Album"),
    URIRef(str(MM) + "Release"),
    URIRef(str(MM) + "ReleaseGroup"),
}

def is_album_type(t: URIRef) -> bool:
    if t in ALBUM_TYPE_CANDIDATES: return True
    return isinstance(t, URIRef) and re.search(r"(album|releasegroup|release)$", str(t), re.I)

def discover_album_nodes(debut_g: Graph) -> Set[URIRef]:
    albums: Set[URIRef] = set()
    for s in debut_g.subjects(SCHEMA.hasPart, None):
        if isinstance(s, URIRef): albums.add(s)
    for o in debut_g.objects(None, SCHEMA.inAlbum):
        if isinstance(o, URIRef): albums.add(o)
    for s,t in debut_g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and is_album_type(t): albums.add(s)
    return albums

def filter_debuts(artists_main_g: Graph, debut_g: Graph) -> Graph:
    kept_artists = set(artists_main_g.subjects(RDF.type, SCHEMA.Person)) | \
                   set(artists_main_g.subjects(RDF.type, SCHEMA.MusicGroup))
    album_nodes = discover_album_nodes(debut_g)

    # album → byArtist → artist (this is how your debut file is modeled)
    BY_ARTIST_PROPS = [SCHEMA.byArtist, URIRef(str(MM) + "byArtist")]  # include mm:byArtist just in case

    kept_albums: Set[URIRef] = set()
    kept_recs: Set[URIRef] = set()

    for alb in album_nodes:
        # keep album if any byArtist is a kept artist
        keep = False
        for prop in BY_ARTIST_PROPS:
            for a in debut_g.objects(alb, prop):
                if isinstance(a, URIRef) and a in kept_artists:
                    keep = True
                    break
            if keep: break
        if not keep:
            continue

        kept_albums.add(alb)
        # pull along its parts (recordings/tracks)
        for rec in debut_g.objects(alb, SCHEMA.hasPart):
            if isinstance(rec, URIRef):
                kept_recs.add(rec)

    out = Graph(); [out.bind(pfx, ns) for pfx, ns in debut_g.namespaces()]
    for s,p,o in debut_g:
        if s in kept_albums or s in kept_recs:
            out.add((s,p,o))

    print(f"[debuts] kept albums: {len(kept_albums)}, kept tracks: {len(kept_recs)}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Filter genres and debut albums to only those referenced by main Live Aid artists.")
    ap.add_argument("--artists-main", default="kg/20_artists.mainonly.ttl")
    ap.add_argument("--genres-in",    default="kg/20_genres.ttl")
    ap.add_argument("--debut-in",     default="kg/30_debut_albums.ttl")
    ap.add_argument("--genres-out",   default="kg/20_genres.mainonly.ttl")
    ap.add_argument("--debut-out",    default="kg/30_debut_albums.mainonly.ttl")
    args = ap.parse_args()

    artists_main_g = load_graph(Path(args.artists_main))
    genres_g       = load_graph(Path(args.genres_in))
    debut_g        = load_graph(Path(args.debut_in))

    # Genres
    genres_out = filter_genres(artists_main_g, genres_g)
    save_graph(genres_out, Path(args.genres_out))

    # Debut albums
    debuts_out = filter_debuts(artists_main_g, debut_g)
    save_graph(debuts_out, Path(args.debut_out))

if __name__ == "__main__":
    main()
