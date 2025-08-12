#!/usr/bin/env python3
import argparse, json, time, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

# Namespaces
SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX = Namespace("http://wembrewind.live/ex#")
EX_BASE = "http://wembrewind.live/ex#"


# ---------- tiny utils ----------
def pascal_slug(text: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", (text or "").strip())
    slug = "".join(p.capitalize() for p in parts if p)
    if not slug: slug = "Entity"
    if slug[0].isdigit(): slug = "A" + slug
    return slug


def fresh_graph() -> Graph:
    g = Graph()
    g.bind("ex", EX);
    g.bind("schema", SCHEMA);
    g.bind("mm", MM)
    g.bind("rdfs", RDFS);
    g.bind("owl", OWL);
    g.bind("xsd", XSD)
    return g


def load_or_new(path: Path) -> Graph:
    g = Graph()
    if path.exists():
        g.parse(path, format="turtle")
    # normalize https->http schema.org
    to_add, to_del = [], []
    for s, p, o in g:
        s2, p2, o2 = s, p, o
        if isinstance(p, URIRef) and str(p).startswith("https://schema.org/"):
            p2 = URIRef(str(p).replace("https://", "http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o2 = URIRef(str(o).replace("https://", "http://"))
        if (s2, p2, o2) != (s, p, o): to_del.append((s, p, o)); to_add.append((s2, p2, o2))
    for t in to_del: g.remove(t)
    for t in to_add: g.add(t)
    g.bind("ex", EX);
    g.bind("schema", SCHEMA);
    g.bind("mm", MM)
    g.bind("rdfs", RDFS);
    g.bind("owl", OWL);
    g.bind("xsd", XSD)
    return g


def save_clean(g_in: Graph, path: Path):
    out = fresh_graph()
    for s, p, o in g_in:
        if isinstance(p, URIRef) and str(p).startswith("https://schema.org/"):
            p = URIRef(str(p).replace("https://", "http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o = URIRef(str(o).replace("https://", "http://"))
        out.add((s, p, o))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=str(path), format="turtle")
    print(f"[write] {path} (triples: {len(out)})")


def iri(curie_or_iri: str) -> URIRef:
    return URIRef(EX_BASE + curie_or_iri[3:]) if curie_or_iri.startswith("ex:") else URIRef(curie_or_iri)


# ---------- MB client ----------
def mb_client(app: str, contact: str):
    import musicbrainzngs as mb
    mb.set_useragent(app or "WembrewindKG/1.0", version="1.0", contact=contact)
    mb.set_rate_limit(True)
    return mb


def fetch_artist(mb, mbid: str) -> Dict:
    # returns dict {"artist": {...}}
    return mb.get_artist_by_id(mbid, includes=["tags"])["artist"]


# ---------- genre resolution (from existing 20_genres.ttl) ----------
class GenreIndex:
    def __init__(self, genres_g: Graph):
        # label (lower) -> node
        self.by_label: Dict[str, URIRef] = {}
        # ex:mbGenreId literal -> node
        self.by_mb_id: Dict[str, URIRef] = {}
        # MB URL in owl:sameAs -> node
        self.by_mb_url: Dict[str, URIRef] = {}

        ex_mbGenreId = URIRef(EX_BASE + "mbGenreId")

        for s in genres_g.subjects(RDF.type, MM.MusicGenre):
            # label
            for _, _, lab in genres_g.triples((s, RDFS.label, None)):
                if isinstance(lab, Literal):
                    self.by_label[str(lab).strip().lower()] = s
            # mbGenreId literal
            for _, _, mbid in genres_g.triples((s, ex_mbGenreId, None)):
                if isinstance(mbid, Literal):
                    self.by_mb_id[str(mbid).strip().lower()] = s
            # owl:sameAs MB URL
            for _, _, same in genres_g.triples((s, OWL.sameAs, None)):
                if isinstance(same, URIRef):
                    u = str(same)
                    if "musicbrainz.org/genre/" in u:
                        self.by_mb_url[u.lower()] = s

    def resolve(self, name_or_id: str) -> Optional[URIRef]:
        if not name_or_id:
            return None
        key = name_or_id.strip().lower()
        # try by label
        if key in self.by_label:
            return self.by_label[key]
        # try by raw MB id
        if key in self.by_mb_id:
            return self.by_mb_id[key]
        # try by MB URL
        if key.startswith("http"):
            if key in self.by_mb_url:
                return self.by_mb_url[key]
        return None


# ---------- harvest artists + mbids from KG ----------
def extract_artist_mbids(artists_g: Graph) -> List[Tuple[URIRef, str]]:
    results = []
    # Preferred: literal ex:mbid
    for s, _, v in artists_g.triples((None, URIRef(EX_BASE + "mbid"), None)):
        if isinstance(v, Literal):
            results.append((s, str(v)))
    # Fallback: owl:sameAs to MB URL
    for s, _, u in artists_g.triples((None, OWL.sameAs, None)):
        if isinstance(u, URIRef) and "musicbrainz.org/artist/" in str(u):
            mbid = str(u).rsplit("/", 1)[-1]
            if (s, URIRef(EX_BASE + "mbid"), None) not in artists_g:
                results.append((s, mbid))
    # dedup by subject
    dedup = {}
    for s, mb in results:
        dedup.setdefault(s, mb)
    pairs = list(dedup.items())
    print(f"[scan] artists with MBIDs: {len(pairs)}")
    return pairs


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kg")
    ap.add_argument("--app", default="WembrewindKG/1.0")
    ap.add_argument("--contact", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Cache MB responses")
    ap.add_argument("--min-tag-count", type=int, default=3, help="threshold when falling back to tags")
    ap.add_argument("--emit-tags", action="store_true", help="also emit non-genre tags into 21_artist_tags.ttl")
    ap.add_argument("--create-missing", action="store_true",
                    help="mint a new mm:MusicGenre if not found in 20_genres.ttl")
    args = ap.parse_args()

    out = Path(args.out)
    artists_ttl = out / "20_artists.ttl"
    genres_ttl = out / "20_genres.ttl"
    tags_ttl = out / "21_artist_tags.ttl"
    cache_dir = out / "enrichment" / "cache" / "artists"

    artists_g = load_or_new(artists_ttl)
    genres_g = load_or_new(genres_ttl)
    tags_g = load_or_new(tags_ttl) if args.emit_tags else fresh_graph()

    # Build canonical genre index from 20_genres.ttl
    gindex = GenreIndex(genres_g)
    if not gindex.by_label and not gindex.by_mb_id and not gindex.by_mb_url:
        print("[warn] No mm:MusicGenre found in 20_genres.ttl; nothing to link to.")

    # collect artists
    subjects = set(artists_g.subjects(RDF.type, SCHEMA.Person)) | set(artists_g.subjects(RDF.type, SCHEMA.MusicGroup))
    pairs = [p for p in extract_artist_mbids(artists_g) if p[0] in subjects]
    if not pairs:
        print("[done] no artists with MBIDs found.")
        return

    # MB client
    mb = mb_client(args.app, args.contact)

    mb.set_useragent(
        "WembleyRewind",  # your app name
        "0.1.0",  # version
        "mailto:you@example.com"  # contact (or project URL)
    )
    mb.set_hostname("musicbrainz.org")
    mb.set_rate_limit(True)

    def load_cache(mbid: str) -> Optional[dict]:
        p = cache_dir / f"{mbid}.json"
        return json.loads(p.read_text("utf-8")) if args.resume and p.exists() else None

    def save_cache(mbid: str, data: dict):
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{mbid}.json").write_text(json.dumps(data), encoding="utf-8")

    enriched = 0
    misses = 0

    for artist_iri, mbid in pairs:
        data = load_cache(mbid)
        if data is None:
            data = fetch_artist(mb, mbid)
            save_cache(mbid, data)
            time.sleep(1.0)  # polite

        print(f"data : {data}")
        # curated genre-list + crowd tag-list
        tag_list = data.get("tag-list", []) or []

        # Normalize into {name -> count} (max count)
        merged: Dict[str, int] = {}
        for it in tag_list:
            name = (it.get("name") or "").strip()
            if not name:
                continue
            try:
                count = int(it.get("count")) if it.get("count") is not None else None
            except Exception:
                count = None
            key = name.lower()
            prev = merged.get(key, 0)
            merged[key] = max(prev, count or 0)

        # Decide which are genres:
        picked_keys = set()
        # Always accept curated genre-list names
        for it in tag_list:
            nm = (it.get("name") or "").strip().lower()
            if nm: picked_keys.add(nm)
        # Accept high-confidence tags (by count threshold)
        for key, cnt in merged.items():
            if cnt >= args.min_tag_count:
                picked_keys.add(key)

        # Resolve picked names to existing genre nodes
        for key in sorted(picked_keys):
            node = gindex.resolve(key)
            if node is None:
                # try a MusicBrainz genre URL pattern if input already looks like an ID/URL (rare here)
                # else log miss
                misses += 1
                continue  # skip linking to unknown genres

            # Link artist → genre
            artists_g.add((artist_iri, MM.hasGenre, node))

        # Optionally keep non-genre tags (as Tag + TagUse nodes)
        if args.emit_tags:
            non_genre = set(merged.keys()) - set(picked_keys)
            for key in sorted(non_genre):
                tag_node = URIRef(EX_BASE + pascal_slug(key))
                tags_g.add((tag_node, RDF.type, URIRef(EX_BASE + "Tag")))
                tags_g.set((tag_node, RDFS.label, Literal(key)))
                use = URIRef(EX_BASE + f"{pascal_slug(str(artist_iri).split('#')[-1])}Tag{pascal_slug(key)}")
                tags_g.add((use, RDF.type, URIRef(EX_BASE + "TagUse")))
                tags_g.add((use, URIRef(EX_BASE + "tag"), tag_node))
                tags_g.add((use, URIRef(EX_BASE + "forArtist"), artist_iri))
                count_val = merged.get(key, 0)
                tags_g.add((use, URIRef(EX_BASE + "tagCount"), Literal(count_val, datatype=XSD.integer)))

        enriched += 1
        if enriched % 10 == 0:
            print(f"[progress] {enriched}/{len(pairs)} artists processed")

    print(f"[summary] artists processed: {enriched}, genre misses (not found in 20_genres): {misses}")

    if args.dry_run:
        print("[dry-run] no files written.")
        return

    save_clean(artists_g, artists_ttl)
    save_clean(genres_g, genres_ttl)
    if args.emit_tags:
        save_clean(tags_g, tags_ttl)
    print("[done] genre enrichment complete.")


if __name__ == "__main__":
    main()
