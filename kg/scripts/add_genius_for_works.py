#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, re, time, unicodedata
from typing import Optional, List, Dict, Set
import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, DC

SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

UA = "Wembrewind-KG/1.0"
TIMEOUT = 12
DELAY = 0.25

GENIUS_BASE = "https://genius.com/"

# ---------- text helpers ----------
def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def title_tokens(s: str) -> List[str]:
    # split on non-alnum, keep letters/numbers; drop empties
    s = ascii_fold(s)
    toks = re.split(r"[^A-Za-z0-9]+", s)
    return [t for t in toks if t]

def to_genius_slug_case(words: List[str]) -> str:
    # Capitalize each token the way Genius tends to do (AC/DC -> AC, DC => AC-DC)
    cap = []
    for w in words:
        if w.isupper():
            cap.append(w)
        elif len(w) <= 3 and w.isalpha():
            cap.append(w.upper())  # acronyms like NME, REM
        else:
            cap.append(w.capitalize())
    return "-".join(cap)

# strip noisy qualifiers from titles before slugging
CLEAN_PATS = [
    r"\([^)]*\)", r"\[[^\]]*\]", r"\{[^}]*\}",
    r"\s+-\s+.*$",                           # " - Live", " - Remaster 2011"
    r"(remaster(ed)?|live|mono|stereo|edit|version)\b.*",
    r"\b\d{4}\b.*",
]
def clean_title(t: str) -> str:
    out = t
    for pat in CLEAN_PATS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE).strip()
    return out.strip(" '\"\u2018\u2019\u201c\u201d").strip() or t

def genius_artist_slug(artist_name: str) -> List[str]:
    """Return candidate artist slugs (e.g., 'Queen', 'AC-DC', 'Guns-N-Roses')."""
    base_tokens = title_tokens(artist_name)
    if not base_tokens:
        return []
    slugs = [to_genius_slug_case(base_tokens)]

    # variant: replace & with 'and'
    if any(t.lower() == "and" for t in base_tokens) or "&" in artist_name:
        toks_and = [("and" if t in {"&"} else t) for t in re.split(r"([^A-Za-z0-9]+)", ascii_fold(artist_name))]
        toks_and = title_tokens(" ".join(toks_and))
        if toks_and:
            slugs.append(to_genius_slug_case(toks_and))

    # de-dupe
    seen, out = set(), []
    for s in slugs:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

def genius_song_slug(title: str) -> List[str]:
    """Return candidate song slugs (Title-Case variants)"""
    raw = title_tokens(title)
    cleaned = title_tokens(clean_title(title))
    slugs = []
    if raw: slugs.append(to_genius_slug_case(raw))
    if cleaned and cleaned != raw: slugs.append(to_genius_slug_case(cleaned))
    # de-dupe
    seen, out = set(), []
    for s in slugs:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

# ---------- HTTP ----------
def http_get(url: str):
    return requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT, allow_redirects=True)

def valid_genius(url: str) -> bool:
    try:
        r = http_get(url)
        if not (200 <= r.status_code < 400):
            return False
        if not r.url.startswith(GENIUS_BASE):
            return False
        # should end with -lyrics or be a canonical lyrics page after redirect
        if "-lyrics" not in r.url.lower():
            return False
        html = r.text.lower()
        return "<title>" in html and "genius" in html
    except requests.RequestException:
        return False

# ---------- RDF helpers ----------
def first_literal(g: Graph, s: URIRef, preds) -> Optional[str]:
    for p in preds:
        for _,_,o in g.triples((s, p, None)):
            if isinstance(o, Literal):
                v = str(o).strip()
                if v:
                    return v
    return None

def load_artist_names(artists_ttl: str) -> Dict[URIRef, str]:
    """artist URI -> preferred name"""
    g = Graph().parse(artists_ttl, format="turtle")
    names = {}
    for s, p, o in g.triples((None, SCHEMA.name, None)):
        if isinstance(o, Literal):
            names[s] = str(o)
    # backfill with https schema & rdfs:label
    for s, p, o in g.triples((None, SCHEMA_HTTPS.name, None)):
        if isinstance(o, Literal) and s not in names:
            names[s] = str(o)
    for s, p, o in g.triples((None, RDFS.label, None)):
        if isinstance(o, Literal) and s not in names:
            names[s] = str(o)
    return names

def collect_artist_work_pairs(setlists_ttl: str) -> List[tuple]:
    """Pair each performer with each mm:performedWork (your data shape)."""
    g = Graph().parse(setlists_ttl, format="turtle")
    pairs = []
    performances = set()
    perf_artists = {}

    for perf, _, artist in g.triples((None, SCHEMA.performer, None)):
        performances.add(perf); perf_artists.setdefault(perf, set()).add(artist)
    for perf, _, artist in g.triples((None, SCHEMA_HTTPS.performer, None)):
        performances.add(perf); perf_artists.setdefault(perf, set()).add(artist)
    for perf, _, artist in g.triples((None, MM.performer, None)):
        performances.add(perf); perf_artists.setdefault(perf, set()).add(artist)

    perf_works = {}
    for perf in performances:
        ws = set(o for _,_,o in g.triples((perf, MM.performedWork, None)))
        if ws:
            perf_works[perf] = ws

    for perf, artists in perf_artists.items():
        if perf in perf_works:
            for a in artists:
                for w in perf_works[perf]:
                    pairs.append((a, w))

    print(f"[debug] performances: {len(performances)} | with mm:performedWork: {len(perf_works)} | pairs: {len(pairs)}")
    return pairs

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", default="20_artists.ttl")
    ap.add_argument("--setlists", default="40_setlists_songs.ttl")
    ap.add_argument("--works", default="33_recordings_works.ttl")
    ap.add_argument("--out-ttl", default="33_works_genius.ttl")
    ap.add_argument("--log", default="genius_works_log.csv")
    ap.add_argument("--no-verify", action="store_true", help="Skip HTTP validation (faster; may include 404s)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    artist_name = load_artist_names(args.artists)
    pairs = collect_artist_work_pairs(args.setlists)
    g_works = Graph().parse(args.works, format="turtle")
    name_preds = [SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label, DC.title]

    outg = Graph(); outg.bind("schema", SCHEMA); outg.bind("mm", MM)
    rows = []
    kept = skipped = tried = 0

    for artist, work in pairs:
        if args.limit and kept + skipped >= args.limit:
            break

        a_name = artist_name.get(artist)
        if not a_name:
            rows.append([str(artist), "", str(work), "", "", "no_artist_name"]); continue

        w_title = first_literal(g_works, work, name_preds)
        if not w_title:
            rows.append([str(artist), a_name, str(work), "", "", "no_work_title"]); continue

        artist_slugs = genius_artist_slug(a_name)
        song_slugs = genius_song_slug(w_title)
        if not artist_slugs or not song_slugs:
            rows.append([str(artist), a_name, str(work), w_title or "", "", "slug_build_failed"]); continue

        ok_url = None
        # try combinations; keep it small (first 3 variants each)
        for a_slug in artist_slugs[:3]:
            for s_slug in song_slugs[:3]:
                url = f"{GENIUS_BASE}{a_slug}-{s_slug}-lyrics"
                if args.no_verify:
                    ok_url = url
                    break
                tried += 1
                if valid_genius(url):
                    ok_url = url
                    break
                time.sleep(DELAY)
            if ok_url:
                break

        if ok_url:
            outg.add((work, SCHEMA.sameAs, URIRef(ok_url)))
            rows.append([str(artist), a_name, str(work), w_title, ok_url, "added"])
            kept += 1
        else:
            rows.append([str(artist), a_name, str(work), w_title, "", "url_invalid"])
            skipped += 1

    outg.serialize(destination=args.out_ttl, format="turtle")
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["artist_uri","artist_name","work_uri","work_title","genius_url","status"]); w.writerows(rows)

    print(f"[debug] URL candidates tried: {tried}")
    print(f"Works added: {kept}")
    print(f"Skipped: {skipped}")
    print(f"Wrote TTL: {args.out_ttl}")
    print(f"Wrote log: {args.log}")

if __name__ == "__main__":
    import unicodedata
    main()
