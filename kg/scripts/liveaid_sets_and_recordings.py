#!/usr/bin/env python3
import time, random, re
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

# ================= Config / Files =================
SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX_BASE = "http://wembrewind.live/ex#"
EX = Namespace(EX_BASE)

KG_DIR = Path("kg")
ARTISTS_TTL = KG_DIR / "20_artists.ttl"
PERF_TTL    = KG_DIR / "23_liveaid_setlists.ttl"
RECWORK_TTL = KG_DIR / "24_recordings_works.ttl"

# assumes this event already exists in your KG
LIVE_AID = EX.LiveAid1985

# Live Aid: The Global Jukebox (1985-07-13) release
RELEASE_MBID = "8aef762f-6a9d-4596-90af-436a515628ff"

APP_NAME = "WembleyRewind"
APP_VER  = "0.1.0"
CONTACT  = "mailto:you@example.com"   # <-- set your contact email

# ================= Helpers =================
NON_SONG_HINTS = {
    "intermission", "intro", "outro", "presentation", "speech", "talk",
    "crowd", "interview", "tuning", "advert", "commercial", "news", "commentary"
}

def looks_like_non_song(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    if t.startswith("[") and t.endswith("]"):
        return True
    return any(h in t for h in NON_SONG_HINTS)

def clean_title(title: str) -> str:
    if not title:
        return ""
    t = title.strip()
    # unwrap a single surrounding [ ... ]
    t = re.sub(r"^\[(.+?)\]$", r"\1", t).strip()
    # strip common trailing qualifiers in parentheses
    t = re.sub(
        r"\s*\((?:live|live aid|wembley|philadelphia|1985|remaster|mix|version|edit|stereo|mono)[^)]*\)\s*$",
        "", t, flags=re.I
    ).strip()
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def medley_split(title: str) -> List[str]:
    if not title:
        return []
    # split on common separators (protect inner parentheses implicitly)
    parts = re.split(r"\s*(?:/|–|-)\s*", title)
    return [p.strip() for p in parts if p.strip()]

def pascal_slug(text: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", (text or "").strip())
    slug = "".join(p.capitalize() for p in parts if p)
    if not slug: slug = "Entity"
    if slug[0].isdigit(): slug = "A" + slug
    return slug

def load_graph(p: Path) -> Graph:
    g = Graph()
    if p.exists():
        g.parse(p, format="turtle")
    # normalize https->http schema.org just in case
    to_del, to_add = [], []
    for s, p_, o in g:
        s2, p2, o2 = s, p_, o
        if isinstance(p_, URIRef) and str(p_).startswith("https://schema.org/"):
            p2 = URIRef(str(p_).replace("https://", "http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o2 = URIRef(str(o).replace("https://", "http://"))
        if (s2, p2, o2) != (s, p_, o):
            to_del.append((s, p_, o)); to_add.append((s2, p2, o2))
    for t in to_del: g.remove(t)
    for t in to_add: g.add(t)
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("owl", OWL); g.bind("xsd", XSD)
    return g

def save_graph(g: Graph, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(p), format="turtle")
    print(f"[write] {p} (triples: {len(g)})")

def ensure_artist_node(artists_g: Graph, name: str, mbid_or_key: str, is_person: bool=True) -> URIRef:
    """Reuse artist if present; otherwise mint in your style and annotate."""
    ex_mbid = URIRef(EX_BASE + "mbid")

    # Try by MBID literal
    for s, _, _ in artists_g.triples((None, ex_mbid, Literal(mbid_or_key))):
        return s

    # Try by owl:sameAs MB URL when mbid-like
    if re.fullmatch(r"[0-9a-f-]{36}", mbid_or_key or ""):
        mb_url = URIRef(f"https://musicbrainz.org/artist/{mbid_or_key}")
        for s, _, _ in artists_g.triples((None, OWL.sameAs, mb_url)):
            return s

    # Mint like your style (ex:Sting)
    base = pascal_slug(name or "Artist")
    node = URIRef(EX_BASE + base)
    i = 2
    while any(True for _ in artists_g.triples((node, None, None))):
        node = URIRef(EX_BASE + f"{base}{i}"); i += 1

    if is_person:
        for t in [SCHEMA.Person, SCHEMA.PropertyValue, MM.Musician]:
            artists_g.add((node, RDF.type, t))
    else:
        for t in [SCHEMA.MusicGroup, SCHEMA.PropertyValue, MM.MusicArtist]:
            artists_g.add((node, RDF.type, t))

    artists_g.add((node, RDFS.label, Literal(name)))
    if re.fullmatch(r"[0-9a-f-]{36}", mbid_or_key or ""):
        artists_g.add((node, ex_mbid, Literal(mbid_or_key)))
        artists_g.add((node, OWL.sameAs, URIRef(f"https://musicbrainz.org/artist/{mbid_or_key}")))
    return node

def _shortid(s: Optional[str]) -> str:
    if not s: return ""
    return re.sub(r"[^0-9a-fA-F]", "", s)[:8]

def rec_iri(rec_mbid: str, title: str) -> URIRef:
    base = pascal_slug(clean_title(title)) or "Recording"
    sid  = _shortid(rec_mbid)
    return URIRef(EX_BASE + (f"Recording_{base}_{sid}" if sid else f"Recording_{base}"))

def work_iri(work_title: str, work_mbid: Optional[str]) -> URIRef:
    base = pascal_slug(clean_title(work_title)) or "Work"
    sid  = _shortid(work_mbid) if work_mbid and not work_mbid.startswith("WORK-") else ""
    # keep it readable; add suffix only when we have a real MBID
    return URIRef(EX_BASE + (f"{base}_{sid}" if sid else base))

def perf_iri(artist_tail: str) -> URIRef:
    return URIRef(EX_BASE + f"{artist_tail}_LiveAid1985_Performance")


def setlist_iri(artist_tail: str) -> URIRef:
    return URIRef(EX_BASE + f"{artist_tail}_LiveAid1985_Set")

def map_place_name_to_ex(name: str) -> Optional[URIRef]:
    n = (name or "").lower()
    if "wembley" in n:
        return EX.WembleyStadium
    if ("kennedy" in n and "stadium" in n) or "jfk stadium" in n:
        return EX.JohnFKennedyStadium
    return None

# ================= MusicBrainz client =================
def mb_client():
    import musicbrainzngs as mb
    mb.set_useragent(APP_NAME, APP_VER, CONTACT)
    mb.set_hostname("musicbrainz.org")
    mb.set_rate_limit(True)
    return mb

def mb_call(fn, *args, **kwargs):
    import musicbrainzngs as mbx
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except mbx.NetworkError:
            time.sleep(min(30, 2 ** attempt + random.random()))
    raise RuntimeError("MusicBrainz kept failing")

# ---------- work resolution ----------
def resolve_recording_works(mb, rec_id: str, rec_title: str, artist_name: str) -> List[Tuple[str, str]]:
    """Return list of (work_mbid, work_title). Tries relations, then medley split + work search."""
    # 1) direct relations
    r = mb_call(mb.get_recording_by_id, rec_id, includes=["work-rels", "artist-credits"])
    rec = r.get("recording", {}) or {}
    wrels = rec.get("relation-list", []) or rec.get("work-relation-list", [])
    works = []
    for reln in wrels:
        typ = (reln.get("type") or "").lower()
        if typ in {"performance", "recording of"}:
            w = reln.get("work") or {}
            wid = w.get("id"); wtitle = w.get("title") or ""
            if wid:
                works.append((wid, wtitle))
    if works:
        return works

    # 2) medley/title split + search
    cleaned = clean_title(rec_title)
    parts = [p for p in medley_split(cleaned) if not looks_like_non_song(p)]
    out = []
    if not parts and not looks_like_non_song(cleaned) and cleaned:
        parts = [cleaned]
    for p in parts:
        try:
            res = mb_call(mb.search_works, query=f'work:"{p}" AND artist:{artist_name}', limit=1)
        except Exception:
            res = {"work-count": 0}
        if res.get("work-count", 0) > 0:
            w = res["work-list"][0]
            out.append((w.get("id"), w.get("title") or p))
        else:
            out.append(("WORK-FALLBACK", p))
        time.sleep(0.15)
    return out

# ---------- venue resolution ----------
def resolve_venue_from_recording(mb, rec_id: str) -> Optional[URIRef]:
    """Try to get a Place from recording rels; fallback via Event -> Place."""
    r = mb_call(mb.get_recording_by_id, rec_id, includes=["place-rels","event-rels"])
    rec = r.get("recording", {}) or {}

    # Walk over relation lists
    for key in list(rec.keys()):
        if not key.endswith("-relation-list"):
            continue
        for rel in rec.get(key, []) or []:
            # direct Place
            plc = rel.get("place")
            if plc and plc.get("name"):
                iri = map_place_name_to_ex(plc["name"])
                if iri: return iri
            # via Event -> Place
            ev = rel.get("event")
            if ev and ev.get("id"):
                try:
                    evd = mb_call(mb.get_event_by_id, ev["id"], includes=["place-rels"])
                    evobj = evd.get("event", {}) or {}
                    for k2 in list(evobj.keys()):
                        if not k2.endswith("-relation-list"):
                            continue
                        for rel2 in evobj.get(k2, []) or []:
                            plc2 = rel2.get("place")
                            if plc2 and plc2.get("name"):
                                iri = map_place_name_to_ex(plc2["name"])
                                if iri: return iri
                except Exception:
                    pass
    return None

def resolve_venue_for_artist_tracks(mb, rec_ids: List[str]) -> Optional[URIRef]:
    """Check a few recordings for a venue; return the first match."""
    for rec_id in rec_ids[:3]:  # sample first few to keep it quick
        iri = resolve_venue_from_recording(mb, rec_id)
        if iri:
            return iri
    return None

# ================= Main =================
def main():
    artists_g = load_graph(ARTISTS_TTL)
    perf_g    = load_graph(PERF_TTL)
    rw_g      = load_graph(RECWORK_TTL)

    mb = mb_client()

    # 1) Pull release — prefer TRACK titles for recording names
    rel = mb_call(mb.get_release_by_id, RELEASE_MBID, includes=["recordings", "artist-credits"])
    media = rel["release"].get("medium-list", []) or []

    # Group tracks by credited artist
    artist_tracks: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    rec_titles: Dict[str, str] = {}
    rec_track_titles: Dict[str, str] = {}

    pos = 0
    for medium in media:
        for tr in medium.get("track-list", []):
            pos += 1
            rec = tr.get("recording", {}) or {}
            rec_id = rec.get("id")
            if not rec_id:
                continue

            # track title preferred for user-facing recording name
            track_title = tr.get("title") or rec.get("title") or ""
            rec_titles[rec_id] = rec.get("title") or ""
            rec_track_titles[rec_id] = track_title

            # artist credit
            artist_mbid, artist_name = None, None
            ac = rec.get("artist-credit") or tr.get("artist-credit") or []
            for piece in ac:
                if isinstance(piece, dict) and "artist" in piece:
                    a = piece["artist"]
                    artist_mbid = a.get("id")
                    artist_name = a.get("name")
                    break
            if not artist_name:
                artist_name = "Unknown"

            artist_tracks[(artist_mbid or artist_name, artist_name)].append(
                {"pos": pos, "recording_id": rec_id, "recording_title": track_title}
            )

    print(f"[release] tracks scanned: {pos}, artists: {len(artist_tracks)}")

    # 2) For each recording, resolve works (songs) robustly
    rec_to_works: Dict[str, List[Tuple[str, str]]] = {}
    for (artist_key, artist_name), items in artist_tracks.items():
        for it in items:
            rec_id = it["recording_id"]
            if rec_id in rec_to_works:
                continue
            works = resolve_recording_works(mb, rec_id, it["recording_title"], artist_name)
            # Filter out obvious non-song placeholders
            works = [(wid, wtitle) for (wid, wtitle) in works if not looks_like_non_song(wtitle)]
            # Fallback: if nothing remains, leave empty -> we will still keep the recording
            rec_to_works[rec_id] = works
            time.sleep(0.1)

    # 3) Emit recordings + works (with clean names)
    for rec_id, works in rec_to_works.items():
        track_title = clean_title(rec_track_titles.get(rec_id, "") or rec_titles.get(rec_id, ""))
        rec_node = rec_iri(rec_id, track_title)

        rw_g.add((rec_node, RDF.type, MM.Recording))
        if track_title:
            rw_g.add((rec_node, RDFS.label, Literal(track_title)))
            rw_g.add((rec_node, SCHEMA.name, Literal(track_title)))
        rw_g.add((rec_node, OWL.sameAs, URIRef(f"https://musicbrainz.org/recording/{rec_id}")))

        for wid, wtitle in works:
            clean_w = clean_title(wtitle)
            work_node = work_iri(clean_w, wid if wid and not wid.startswith("WORK-FALLBACK") else None)

            rw_g.add((work_node, RDF.type, SCHEMA.MusicComposition))
            rw_g.add((work_node, RDF.type, MM.MusicEntity))
            if clean_w:
                rw_g.add((work_node, RDFS.label, Literal(clean_w)))
                rw_g.add((work_node, SCHEMA.name, Literal(clean_w)))
            if wid and not wid.startswith("WORK-FALLBACK"):
                rw_g.add((work_node, OWL.sameAs, URIRef(f"https://musicbrainz.org/work/{wid}")))
            # Music Meta native link + schema compatibility
            rw_g.add((work_node, MM.hasRecording, rec_node))
            rw_g.add((rec_node, SCHEMA.recordingOf, work_node))

    # 4) Emit performances + ordered setlists per artist (with isPartOf + location)
    for (artist_key, artist_name), tracks in artist_tracks.items():
        is_person_guess = True
        artist_mbid = artist_key if re.fullmatch(r"[0-9a-f-]{36}", str(artist_key) or "") else None
        artist_iri = ensure_artist_node(artists_g, artist_name, artist_mbid or pascal_slug(artist_name), is_person=is_person_guess)

        artist_tail = str(artist_iri).split("#")[-1]
        perf = perf_iri(artist_tail)
        setlist = setlist_iri(artist_tail)

        perf_g.add((perf, RDF.type, MM.LivePerformance))
        perf_g.add((perf, RDF.type, SCHEMA.Event))
        perf_g.add((perf, SCHEMA.performer, artist_iri))
        perf_g.add((perf, SCHEMA.isPartOf, LIVE_AID))

        # try to resolve venue (Wembley vs JFK) from a few of this artist's recordings
        rec_ids = [it["recording_id"] for it in tracks]
        venue_iri = resolve_venue_for_artist_tracks(mb, rec_ids)
        if venue_iri:
            perf_g.add((perf, SCHEMA.location, venue_iri))

        # set list container
        perf_g.add((setlist, RDF.type, SCHEMA.ItemList))
        perf_g.add((setlist, SCHEMA.name, Literal(f"{artist_name} – Live Aid set")))

        # position by ascending track order, skip non-song items
        i = 0
        for item in sorted(tracks, key=lambda x: x["pos"]):
            rec_id = item["recording_id"]
            rec_node = rec_iri(rec_id, clean_title(rec_track_titles.get(rec_id, "") or rec_titles.get(rec_id, "")))
            works = rec_to_works.get(rec_id, [])
            if not works:
                continue
            for wid, wtitle in works:
                clean_w = clean_title(wtitle)
                work_node = work_iri(clean_w, wid if wid and not wid.startswith("WORK-FALLBACK") else None)

                # Performance ↔ Song ↔ Recording links
                perf_g.add((perf, MM.performedWork, work_node))
                perf_g.add((work_node, MM.isRealisedBy, perf))
                perf_g.add((perf, MM.recordedAs, rec_node))

                # ordered list element
                i += 1
                li = BNode()
                perf_g.add((li, RDF.type, SCHEMA.ListItem))
                perf_g.add((li, SCHEMA.position, Literal(i, datatype=XSD.integer)))
                perf_g.add((li, SCHEMA.item, work_node))
                perf_g.add((setlist, SCHEMA.itemListElement, li))

    # 5) Save
    save_graph(artists_g, ARTISTS_TTL)
    save_graph(perf_g,    PERF_TTL)
    save_graph(rw_g,      RECWORK_TTL)
    print("[done] performances + recordings/works emitted (isPartOf + location).")

if __name__ == "__main__":
    main()
