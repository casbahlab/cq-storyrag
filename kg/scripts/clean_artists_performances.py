#!/usr/bin/env python3
import time, random, re
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

from rdflib import Graph, Namespace, URIRef, BNode, RDF, RDFS, OWL, Literal

# ---- Config ----
SCHEMA = Namespace("http://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX = Namespace("http://wembrewind.live/ex#")

KG_DIR = Path("kg")
ARTISTS_IN = KG_DIR / "20_artists.ttl"
PERF_IN    = KG_DIR / "23_liveaid_setlists.ttl"
RECW_IN    = KG_DIR / "24_recordings_works.ttl"

ARTISTS_OUT = KG_DIR / "20_artists.mainonly.ttl"
PERF_OUT    = KG_DIR / "23_liveaid_setlists.mainonly.ttl"
RECW_OUT    = KG_DIR / "24_recordings_works.mainonly.ttl"

LIVE_AID = EX.LiveAid1985
WEMBLEY_EVENT_MBID = "08657c50-71c6-4a3d-b7c4-3db6efbf07fd"
PHILA_EVENT_MBID   = "e4ea9ddd-3fbb-47d0-b570-add507bf0c27"

APP_NAME = "WembleyRewind"
APP_VER  = "0.1.0"
CONTACT  = "mailto:you@example.com"  # set me

# relation types on MB events we consider as “main performers”
PERFORMER_REL_TYPES = {"performers", "main performer", "performer"}  # be permissive
EXCLUDE_REL_TYPES   = {"host", "presenter", "dj", "mc"}

# ---- IO helpers ----
def load_graph(p: Path) -> Graph:
    g = Graph()
    if p.exists():
        g.parse(p, format="turtle")
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("owl", OWL)
    return g

def save_graph(g: Graph, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(p), format="turtle")
    print(f"[write] {p} (triples: {len(g)})")

# ---- MusicBrainz client ----
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

def performers_from_event(mb, event_mbid: str) -> Set[str]:
    """Return set of artist MBIDs that are performers on a MusicBrainz event."""
    data = mb_call(mb.get_event_by_id, event_mbid, includes=["artist-rels"])
    rels = data.get("event", {}).get("artist-relation-list", []) or []
    keep: Set[str] = set()
    for rel in rels:
        rtype = (rel.get("type") or "").lower()
        if rtype in EXCLUDE_REL_TYPES:
            continue
        if (rtype in PERFORMER_REL_TYPES) or ("perform" in rtype):
            a = rel.get("artist") or {}
            if a.get("id"):
                keep.add(a["id"])
    return keep

# ---- Mapping MBID -> KG IRI (artists) ----
def harvest_artist_by_mbid(g: Graph) -> Dict[str, URIRef]:
    """Index artists by ex:mbid literal and owl:sameAs MB URL."""
    ex_mbid = EX.mbid
    idx: Dict[str, URIRef] = {}
    # literal
    for s, _, lit in g.triples((None, ex_mbid, None)):
        if isinstance(lit, Literal):
            idx[str(lit)] = s
    # sameAs URL
    for s, _, u in g.triples((None, OWL.sameAs, None)):
        if isinstance(u, URIRef) and "musicbrainz.org/artist/" in str(u):
            mbid = str(u).rstrip("/").rsplit("/", 1)[-1]
            idx.setdefault(mbid, s)
    return idx

# ---- Filtering utilities ----
def subgraph_by_subjects(g: Graph, subjects: Set[URIRef]) -> Graph:
    out = Graph(); [out.bind(pfx, ns) for pfx, ns in g.namespaces()]
    for s, p, o in g:
        if s in subjects:
            out.add((s, p, o))
    return out

def closure_from_perf(perf_g: Graph, keep_perfs: Set[URIRef]) -> Tuple[Set[URIRef], Set[URIRef], Set[BNode]]:
    """Return (works, recordings, listitem_bnodes) reachable from kept performances."""
    works: Set[URIRef] = set()
    recs: Set[URIRef]  = set()
    listitems: Set[BNode] = set()
    # works via mm:performedWork and setlist items
    for perf in keep_perfs:
        for w in perf_g.objects(perf, MM.performedWork):
            if isinstance(w, URIRef): works.add(w)
        for r in perf_g.objects(perf, MM.recordedAs):
            if isinstance(r, URIRef): recs.add(r)
    # setlist items (schema:itemListElement -> ListItem -> schema:item -> work)
    for setlist in set(perf_g.objects(None, SCHEMA.name)):  # just to force iteration
        pass  # (we’ll scan globally instead)
    for setlist in perf_g.subjects(RDF.type, SCHEMA.ItemList):
        # include only setlists that belong to kept performers by checking any link from perf
        # (setlists are named per artist in your build, so we’ll keep all ItemLists referenced by kept perfs)
        for li in perf_g.objects(setlist, SCHEMA.itemListElement):
            if isinstance(li, BNode):
                listitems.add(li)
                w = next(perf_g.objects(li, SCHEMA.item), None)
                if isinstance(w, URIRef): works.add(w)
    return works, recs, listitems

def filter_performance_graph(perf_g: Graph, artist_set: Set[URIRef]) -> Graph:
    """Keep performances (and their setlists) for artists in artist_set and isPartOf LiveAid."""
    keep_perfs: Set[URIRef] = set()
    keep_setlists: Set[URIRef] = set()
    keep_listitems: Set[BNode] = set()

    # select performances
    for perf in perf_g.subjects(RDF.type, MM.LivePerformance):
        ok = (perf_g.value(perf, SCHEMA.isPartOf) == LIVE_AID) and \
             (isinstance(perf_g.value(perf, SCHEMA.performer), URIRef) and perf_g.value(perf, SCHEMA.performer) in artist_set)
        if ok:
            keep_perfs.add(perf)

    # gather closure (works, recs, list items)
    works, recs, listitems = closure_from_perf(perf_g, keep_perfs)
    keep_listitems |= listitems

    # collect setlists that mention kept listitems
    for setlist in perf_g.subjects(RDF.type, SCHEMA.ItemList):
        for li in perf_g.objects(setlist, SCHEMA.itemListElement):
            if isinstance(li, BNode) and li in keep_listitems:
                keep_setlists.add(setlist)

    # build output
    out = Graph(); [out.bind(pfx, ns) for pfx, ns in perf_g.namespaces()]
    to_keep_nodes: Set = set().union(keep_perfs, keep_setlists, keep_listitems)
    for s, p, o in perf_g:
        if s in to_keep_nodes or (isinstance(s, BNode) and s in keep_listitems):
            out.add((s, p, o))
    return out, works, recs, keep_perfs

def filter_recwork_graph(rw_g: Graph, keep_works: Set[URIRef], keep_recs: Set[URIRef]) -> Graph:
    out = Graph(); [out.bind(pfx, ns) for pfx, ns in rw_g.namespaces()]
    # keep any triple whose subject is a kept work or kept recording
    for s, p, o in rw_g:
        if s in keep_works or s in keep_recs:
            out.add((s, p, o))
    # also keep recordings that are linked from kept works (safety)
    for w in keep_works:
        for r in rw_g.objects(w, MM.hasRecording):
            if isinstance(r, URIRef):
                for t in rw_g.triples((r, None, None)):
                    out.add(t)
    return out

def main():
    # load graphs
    artists_g = load_graph(ARTISTS_IN)
    perf_g    = load_graph(PERF_IN)
    rw_g      = load_graph(RECW_IN)

    # get performer MBIDs from both events
    mb = mb_client()
    keep_mbids = performers_from_event(mb, WEMBLEY_EVENT_MBID) | performers_from_event(mb, PHILA_EVENT_MBID)
    print(f"[mb] performer MBIDs: {len(keep_mbids)}")

    # map MBID -> artist IRI in your KG
    idx = harvest_artist_by_mbid(artists_g)
    keep_artists: Set[URIRef] = {idx[m] for m in keep_mbids if m in idx}
    print(f"[kg] matched artists in KG: {len(keep_artists)}")

    # filter artists graph (subjects only)
    artists_out = subgraph_by_subjects(artists_g, keep_artists)
    save_graph(artists_out, ARTISTS_OUT)

    # filter performances graph (and gather referenced works/recordings)
    perf_out, keep_works, keep_recs, keep_perfs = filter_performance_graph(perf_g, keep_artists)
    print(f"[perf] kept performances: {len(keep_perfs)}, works: {len(keep_works)}, recordings: {len(keep_recs)}")
    save_graph(perf_out, PERF_OUT)

    # filter recordings/works graph
    rw_out = filter_recwork_graph(rw_g, keep_works, keep_recs)
    save_graph(rw_out, RECW_OUT)

    print("[done] filtered to main Live Aid performers.")

if __name__ == "__main__":
    main()
