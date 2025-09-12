#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, re, time, unicodedata
from typing import Optional, List, Dict, Set, Tuple
import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, DC

SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
SONGFACTS_BASE = "https://www.songfacts.com/facts/"
UA = "Wembrewind-KG/1.0"
TIMEOUT = 12
DELAY = 0.3

def http_get(url: str):
    return requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)

def slugify(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+","-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

CLEAN_PATS = [
    r"\([^)]*\)", r"\[[^\]]*\]", r"\{[^}]*\}",
    r"\s+-\s+.*$", r"(remaster(ed)?|live|mono|stereo|edit|version)\b.*", r"\b\d{4}\b.*",
]
def clean_title(t: str) -> str:
    out = t
    for pat in CLEAN_PATS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE).strip()
    out = out.strip(" '\"\u2018\u2019\u201c\u201d").strip()
    return out or t

def build_song_urls(artist_slug: str, title: str) -> List[str]:
    raw = slugify(title)
    cleaned = slugify(clean_title(title))
    urls = [f"{SONGFACTS_BASE}{artist_slug}/{raw}"]
    if cleaned != raw:
        urls.append(f"{SONGFACTS_BASE}{artist_slug}/{cleaned}")
    # de-dupe, keep order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def valid_songfacts(url: str) -> bool:
    try:
        r = http_get(url)
        if not (200 <= r.status_code < 400): return False
        if not r.url.startswith(SONGFACTS_BASE): return False
        html = r.text.lower()
        if "<title>" in html and "songfacts" in html: return True
        return any(re.search(p, html) for p in [r'class=".*?facts.*?"', r'id="facts"', r">facts<", r"songfacts\u00ae", r"songfacts®"])
    except requests.RequestException:
        return False

def first_literal(g: Graph, s: URIRef, preds) -> Optional[str]:
    for p in preds:
        for _,_,o in g.triples((s, p, None)):
            if isinstance(o, Literal):
                v = str(o).strip()
                if v: return v
    return None

def load_artist_slugs(artists_songfacts_ttl: str) -> Dict[URIRef, str]:
    """Map artist URI -> Songfacts artist slug (from sameAs URL tail)."""
    g = Graph().parse(artists_songfacts_ttl, format="turtle")
    slugs = {}
    for s,p,o in g:
        if isinstance(o, URIRef) and "songfacts.com/facts/" in str(o):
            tail = str(o).rstrip("/").split("/")[-1]
            slugs[s] = tail
    return slugs

def load_artist_work_pairs(setlists_ttl: str) -> List[tuple]:
    """Directly pair each schema1/schema performer with each mm:performedWork."""
    g = Graph().parse(setlists_ttl, format="turtle")
    pairs = []

    # find performances
    performances = set()
    perf_artists = {}  # perf -> set(artist)
    # schema1:performer (some files alias schema1->http schema)
    for perf, _, artist in g.triples((None, SCHEMA.performer, None)):
        performances.add(perf); perf_artists.setdefault(perf, set()).add(artist)
    for perf, _, artist in g.triples((None, SCHEMA_HTTPS.performer, None)):
        performances.add(perf); perf_artists.setdefault(perf, set()).add(artist)

    # if mm:performer is also used, include it
    for perf, _, artist in g.triples((None, MM.performer, None)):
        performances.add(perf); perf_artists.setdefault(perf, set()).add(artist)

    # for each performance, collect direct mm:performedWork objects
    perf_works_map = {}
    for perf in performances:
        works = set(o for _,_,o in g.triples((perf, MM.performedWork, None)))
        if works:
            perf_works_map[perf] = works

    # build (artist, work) pairs
    for perf, artists in perf_artists.items():
        if perf in perf_works_map:
            for a in artists:
                for w in perf_works_map[perf]:
                    pairs.append((a, w))

    print(f"[debug] performances: {len(performances)}")
    print(f"[debug] performances with mm:performedWork: {len(perf_works_map)}")
    print(f"[debug] (artist, work) pairs from setlists: {len(pairs)}")
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists-songfacts", default="20_artists_songfacts.ttl")
    ap.add_argument("--setlists", default="40_setlists_songs.ttl")
    ap.add_argument("--works", default="33_recordings_works.ttl")
    ap.add_argument("--out-ttl", default="33_works_songfacts.ttl")
    ap.add_argument("--log", default="songfacts_works_log.csv")
    ap.add_argument("--no-verify", action="store_true", help="Do not HTTP-check song URLs; just emit them")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    artist_slug = load_artist_slugs(args.artists_songfacts)
    print(f"[debug] Songfacts artist slugs loaded: {len(artist_slug)}")

    pairs = load_artist_work_pairs(args.setlists)

    g_works = Graph().parse(args.works, format="turtle")
    name_preds = [SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label, DC.title]

    outg = Graph(); outg.bind("schema", SCHEMA); outg.bind("mm", MM)
    rows = []
    kept = skipped = tried = 0

    for artist, work in pairs:
        if args.limit and kept + skipped >= args.limit: break
        if artist not in artist_slug:
            rows.append([str(artist), "", str(work), "", "", "artist_not_in_songfacts"]); continue

        title = first_literal(g_works, work, name_preds)
        if not title:
            rows.append([str(artist), artist_slug[artist], str(work), "", "", "no_title"]); continue

        urls = build_song_urls(artist_slug[artist], title)

        ok_url = None
        if args.no_verify:
            ok_url = urls[-1]  # favor cleaned slug when not verifying
        else:
            for u in urls:
                tried += 1
                if valid_songfacts(u):
                    ok_url = u
                    break
                time.sleep(DELAY)

        if ok_url:
            outg.add((work, SCHEMA.sameAs, URIRef(ok_url)))
            rows.append([str(artist), artist_slug[artist], str(work), title, ok_url, "added"])
            kept += 1
        else:
            rows.append([str(artist), artist_slug[artist], str(work), title, urls[0], "url_invalid"])
            skipped += 1
        time.sleep(DELAY)

    outg.serialize(destination=args.out_ttl, format="turtle")

    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["artist_uri","artist_slug","work_uri","work_title","songfacts_url","status"]); w.writerows(rows)

    print(f"[debug] URL candidates tried: {tried}")
    print(f"Works added: {kept}")
    print(f"Skipped: {skipped}")
    print(f"Wrote TTL: {args.out_ttl}")
    print(f"Wrote log: {args.log}")

if __name__ == "__main__":
    import unicodedata
    main()
