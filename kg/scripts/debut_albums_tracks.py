#!/usr/bin/env python3
"""
Read artists from 20_artists.ttl, fetch their debut album + tracks from MusicBrainz,
and save as RDF/Turtle (album, recordings, songs).

Usage:
  python debut_albums_tracks.py --in 20_artists.ttl
  python debut_albums_tracks.py --in 20_artists.ttl --out my_output.ttl
"""

import argparse
import re
import time
from typing import Dict, List, Optional, Tuple

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD
import musicbrainzngs

# ---------------- Config ----------------
APP_NAME = "WembrewindDebutAlbums"
APP_VERSION = "1.0"
APP_CONTACT = "you@example.com"  # update to your email/site
REQUEST_DELAY_SEC = 1.0
MAX_RETRIES = 5
BACKOFF_BASE_SEC = 1.6

SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
MB_BASE = "https://musicbrainz.org/"

EXCLUDED_SECONDARY_TYPES = {
    "Compilation", "Live", "Remix", "Soundtrack", "Interview",
    "Audiobook", "Spokenword", "DJ-mix", "Demo", "Mixtape/Street"
}

# --------------- Utils ------------------

def set_user_agent():
    musicbrainzngs.set_useragent(APP_NAME, APP_VERSION, APP_CONTACT)

def sleep_backoff(attempt: int):
    delay = min(BACKOFF_BASE_SEC ** attempt, 20.0)
    time.sleep(delay)

def with_retries(fn, *args, **kwargs):
    attempt = 0
    while True:
        try:
            res = fn(*args, **kwargs)
            time.sleep(REQUEST_DELAY_SEC)
            return res
        except musicbrainzngs.NetworkError:
            if attempt >= MAX_RETRIES:
                raise
            sleep_backoff(attempt)
            attempt += 1
        except musicbrainzngs.ResponseError as e:
            code = getattr(getattr(e, "cause", None), "code", None)
            if code == 503 and attempt < MAX_RETRIES:
                sleep_backoff(attempt)
                attempt += 1
                continue
            raise

def looks_like_mbid(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F-]{36}", text.strip()))

def mb_artist_url_to_mbid(url: str) -> Optional[str]:
    m = re.search(r"/artist/([0-9a-fA-F-]{36})", url)
    return m.group(1) if m else None

def sanitize_slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return re.sub(r"_+", "_", s).strip("_") or "unnamed"

# --------------- MusicBrainz fetchers ---------------

def search_artist_mbid_by_name(name: str) -> Optional[str]:
    res = with_retries(musicbrainzngs.search_artists, artist=name, limit=5)
    candidates = res.get("artist-list", [])
    if not candidates:
        return None
    exact = [a for a in candidates if a.get("name", "").lower() == name.lower()]
    pool = exact if exact else candidates
    pool.sort(key=lambda a: (int(a.get("ext:score", "0")), a.get("life-span", {}).get("ended") == "true"), reverse=True)
    return pool[0]["id"]

# --- replace fetch_debut_release_group() with ---
def fetch_debut_release_group(artist_mbid: str) -> Optional[Dict]:
    # Do NOT include 'releases' here; browse doesn't support it.
    res = with_retries(
        musicbrainzngs.browse_release_groups,
        artist=artist_mbid,
        release_type=["album"],
        limit=100
    )
    rgs = res.get("release-group-list", []) or []

    def is_clean_album(rg: Dict) -> bool:
        if rg.get("primary-type") != "Album":
            return False
        for sec in rg.get("secondary-type-list", []):
            if sec in EXCLUDED_SECONDARY_TYPES:
                return False
        return True

    clean = [rg for rg in rgs if is_clean_album(rg)]
    if not clean:
        return None

    def first_date(rg: Dict) -> str:
        if rg.get("first-release-date"):
            return rg["first-release-date"]
        # no release dates available at this stage; leave a late sentinel
        return "9999-99-99"

    clean.sort(key=first_date)
    return clean[0]

import re
from rdflib import Literal, URIRef
from rdflib.namespace import XSD

XSD_GYEAR = getattr(XSD, "gYear", URIRef("http://www.w3.org/2001/XMLSchema#gYear"))
XSD_GYEARMONTH = getattr(XSD, "gYearMonth", URIRef("http://www.w3.org/2001/XMLSchema#gYearMonth"))

def make_date_literal(s: str):
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return Literal(s, datatype=XSD.date)
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return Literal(s, datatype=XSD_GYEARMONTH)
    if re.fullmatch(r"\d{4}", s):
        return Literal(s, datatype=XSD_GYEAR)
    # unknown format: store as plain string to avoid crashes
    return Literal(s)


# --- replace choose_release_for_tracks() with ---
def choose_release_for_tracks(rg: Dict) -> Optional[Dict]:
    # Always fetch releases here (this endpoint supports inc="releases").
    rg_full = with_retries(
        musicbrainzngs.get_release_group_by_id,
        rg["id"],
        includes=["releases"]
    )
    releases = rg_full.get("release-group", {}).get("release-list", []) or []

    def key_rel(rel: Dict):
        status = rel.get("status") or ""
        date = rel.get("date") or "9999-99-99"
        # Prefer Official, then earliest date
        return ("" if status == "Official" else "z", date)

    releases.sort(key=key_rel)
    return releases[0] if releases else None


def fetch_release_with_recordings(release_id: str) -> Dict:
    return with_retries(
        musicbrainzngs.get_release_by_id,
        release_id,
        includes=["recordings", "labels"]
    )

# --------------- RDF helpers ----------------

def add_album(graph: Graph, EX: Namespace, artist_uri: URIRef, artist_name: str, rg: Dict, release: Dict) -> URIRef:
    album_title = rg.get("title")
    album_slug = sanitize_slug(f"{artist_name}_{album_title}_Album")
    album_uri = EX[album_slug]

    graph.add((album_uri, RDF.type, SCHEMA.MusicAlbum))
    if album_title:
        graph.add((album_uri, SCHEMA.name, Literal(album_title)))

    # Mark this album as the debut
    graph.add((album_uri, EX.isDebutAlbum, Literal(True)))

    date = rg.get("first-release-date") or release.get("date")
    dt_lit = make_date_literal(date)  # assuming you’re using the helper to handle YYYY/MM/DD cases
    if dt_lit is not None:
        graph.add((album_uri, SCHEMA.datePublished, dt_lit))

    graph.add((album_uri, SCHEMA.sameAs, URIRef(f"{MB_BASE}release-group/{rg['id']}")))
    if release.get("id"):
        graph.add((album_uri, SCHEMA.sameAs, URIRef(f"{MB_BASE}release/{release['id']}")))
    graph.add((album_uri, SCHEMA.byArtist, artist_uri))
    return album_uri



def add_recording_and_song(graph: Graph, EX: Namespace, album_uri: URIRef, artist_uri: URIRef, artist_name: str, track: Dict):
    rec_title = track.get("recording", {}).get("title") or track.get("title")
    rec_id = (track.get("recording") or {}).get("id")

    rec_slug = sanitize_slug(f"{artist_name}_{rec_title}_Rec")
    rec_uri = EX[rec_slug]

    graph.add((rec_uri, RDF.type, MM.Recording))
    if rec_title:
        graph.add((rec_uri, SCHEMA.name, Literal(rec_title)))
    graph.add((rec_uri, SCHEMA.inAlbum, album_uri))
    graph.add((album_uri, SCHEMA.hasPart, rec_uri))
    if rec_id:
        graph.add((rec_uri, SCHEMA.sameAs, URIRef(f"{MB_BASE}recording/{rec_id}")))

    song_slug = sanitize_slug(f"{artist_name}_{rec_title}_Song")
    song_uri = EX[song_slug]
    graph.add((song_uri, RDF.type, SCHEMA.MusicComposition))
    if rec_title:
        graph.add((song_uri, SCHEMA.name, Literal(rec_title)))
    graph.add((rec_uri, SCHEMA.recordingOf, song_uri))
    graph.add((song_uri, SCHEMA.byArtist, artist_uri))

def resolve_artist_nodes(g_in: Graph) -> List[Tuple[URIRef, str, Optional[str]]]:
    """
    Return (artist_uri, name, mbid_or_none) for all mm:Musician in the input TTL.
    """
    artists: List[Tuple[URIRef, str, Optional[str]]] = []

    q = """
    SELECT ?a ?name
    WHERE {
      ?a a mm:Musician .
      OPTIONAL { ?a schema:name ?name . }
    }
    """
    for a_uri, name in g_in.query(q, initNs={"schema": SCHEMA, "mm": MM}):
        name_str = str(name) if name else (str(a_uri).split("#")[-1])
        mbid: Optional[str] = None
        # Check for MusicBrainz MBID in schema:sameAs links
        for same in g_in.objects(a_uri, SCHEMA.sameAs):
            if isinstance(same, URIRef):
                mbid = mb_artist_url_to_mbid(str(same))
                if mbid:
                    break
        artists.append((a_uri, name_str, mbid))
    return artists


def ensure_artist_mbid(artist_name: str, mbid: Optional[str]) -> Optional[str]:
    if mbid and looks_like_mbid(mbid):
        return mbid
    return search_artist_mbid_by_name(artist_name)

# --------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_file", required=True, help="Input TTL file with artists")
    parser.add_argument("--out", dest="out_file", help="Output TTL file (default: 30_debut_albums.ttl)")
    parser.add_argument("--ex-base", dest="ex_base", default="http://wembrewind.live/ex#", help="Base EX namespace")
    args = parser.parse_args()

    out_file = args.out_file or "30_debut_albums.ttl"

    set_user_agent()

    g_in = Graph()
    g_in.parse(args.in_file, format="turtle")

    EX = Namespace(args.ex_base)
    g_out = Graph()
    g_out.bind("schema", SCHEMA)
    g_out.bind("mm", MM)
    g_out.bind("ex", EX)

    artists = resolve_artist_nodes(g_in)
    if not artists:
        print("[WARN] No artists found (expected schema:MusicArtist or schema:MusicGroup).")
        g_out.serialize(destination=out_file, format="turtle")
        print(f"Saved: {out_file}")
        return

    for artist_uri, artist_name, mbid in artists:
        try:
            artist_mbid = ensure_artist_mbid(artist_name, mbid)
            if not artist_mbid:
                print(f"[WARN] No MBID for {artist_name}")
                continue

            rg = fetch_debut_release_group(artist_mbid)
            if not rg:
                print(f"[WARN] No debut album for {artist_name}")
                continue

            release = choose_release_for_tracks(rg)
            if not release:
                print(f"[WARN] No release for debut album of {artist_name}")
                continue

            release_full = fetch_release_with_recordings(release["id"])
            mediums = release_full.get("release", {}).get("medium-list", []) or []

            album_uri = add_album(g_out, EX, artist_uri, artist_name, rg, release)

            for medium in mediums:
                for tr in medium.get("track-list", []) or []:
                    add_recording_and_song(g_out, EX, album_uri, artist_uri, artist_name, tr)

            print(f"[OK] {artist_name}: {rg.get('title')}")

        except Exception as e:
            print(f"[ERROR] {artist_name}: {e}")

    g_out.serialize(destination=out_file, format="turtle")
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
