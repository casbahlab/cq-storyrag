#!/usr/bin/env python3
import argparse, re
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from rdflib import Graph, Namespace, URIRef, BNode, RDF, RDFS, OWL, Literal

SCHEMA_HTTP  = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
SCHEMA = Namespace(SCHEMA_HTTP)
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

ITEMLIST         = URIRef(SCHEMA_HTTP + "ItemList")
LISTITEM         = URIRef(SCHEMA_HTTP + "ListItem")
ITEMLISTELEMENT  = URIRef(SCHEMA_HTTP + "itemListElement")
ITEM             = URIRef(SCHEMA_HTTP + "item")
POSITION         = URIRef(SCHEMA_HTTP + "position")
NAME             = URIRef(SCHEMA_HTTP + "name")
# tolerate https predicates, too
ITEMLISTELEMENT_S = URIRef(SCHEMA_HTTPS + "itemListElement")
ITEM_S            = URIRef(SCHEMA_HTTPS + "item")
POSITION_S        = URIRef(SCHEMA_HTTPS + "position")
NAME_S            = URIRef(SCHEMA_HTTPS + "name")

def load_graph(p: Path) -> Graph:
    g = Graph()
    g.parse(p, format="turtle")
    # normalize https->http schema.org
    fixes = []
    for s,p,o in g:
        p2, o2 = p, o
        if isinstance(p, URIRef) and str(p).startswith(SCHEMA_HTTPS):
            p2 = URIRef(str(p).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if isinstance(o, URIRef) and str(o).startswith(SCHEMA_HTTPS):
            o2 = URIRef(str(o).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if (p2,o2) != (p,o):
            fixes.append(((s,p,o),(s,p2,o2)))
    for (s,p,o),(s2,p2,o2) in fixes:
        g.remove((s,p,o))
        g.add((s2,p2,o2))
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("ex", EX); g.bind("rdfs", RDFS); g.bind("owl", OWL)
    return g

def save_graph(g: Graph, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(p), format="turtle")

def label_of(g: Graph, u: URIRef) -> str:
    for o in g.objects(u, NAME): return str(o)
    for o in g.objects(u, RDFS.label): return str(o)
    s = str(u);  return s.split("#")[-1] if "#" in s else s.rsplit("/",1)[-1]

def mb_work_id(g: Graph, w: URIRef) -> Optional[str]:
    for o in g.objects(w, OWL.sameAs):
        m = re.search(r"/work/([0-9a-f-]{36})", str(o), re.I)
        if m: return m.group(1).lower()
    return None

def find_itemlists(g: Graph) -> List[URIRef]:
    return [s for s in g.subjects(RDF.type, ITEMLIST) if isinstance(s, URIRef)]

def collect_items(g: Graph, setlist: URIRef) -> List[Tuple[Optional[int], URIRef, BNode]]:
    items = []
    for _, _, li in g.triples((setlist, ITEMLISTELEMENT, None)):
        if not isinstance(li, BNode): continue
        w = next(g.objects(li, ITEM), None)
        if not isinstance(w, URIRef): continue
        pos_lit = next(g.objects(li, POSITION), None)
        pos = None
        try:
            if isinstance(pos_lit, Literal):
                pos = int(str(pos_lit))
        except Exception:
            pos = None
        items.append((pos, w, li))
    return items

def clean_setlist(g_set: Graph, g_ref: Optional[Graph], setlist: URIRef, use_mbid: bool) -> int:
    items = collect_items(g_set, setlist)
    if not items:
        return 0

    # Build canonical key per work (MBID if available and requested; else IRI)
    def key_for(w: URIRef) -> Tuple[str, str]:
        if use_mbid:
            mbid = mb_work_id(g_ref or g_set, w)
            if mbid:
                return ("mbid", mbid)
        return ("iri", str(w))

    # Keep the earliest position per key; if no pos, record as large number to sort later
    first_by_key: Dict[Tuple[str,str], Tuple[int, URIRef]] = {}
    for pos, w, _ in items:
        k = key_for(w)
        norm_pos = pos if pos is not None else 10**9
        if k not in first_by_key or norm_pos < first_by_key[k][0]:
            first_by_key[k] = (norm_pos, w)

    # Order by recorded position (then by label for stability)
    ordered = sorted(
        [(p if p != 10**9 else None, w) for (p,w) in first_by_key.values()],
        key=lambda t: (t[0] is None, t[0] if t[0] is not None else 10**9, label_of(g_ref or g_set, t[1]).lower())
    )

    # 1) Remove all existing listitem bnodes for this set
    to_drop = []
    for _, _, li in list(g_set.triples((setlist, ITEMLISTELEMENT, None))):
        if isinstance(li, BNode):
            to_drop.append(li)
    for li in to_drop:
        for t in list(g_set.triples((li, None, None))):
            g_set.remove(t)
        g_set.remove((setlist, ITEMLISTELEMENT, li))

    # 2) Rebuild with unique items and fresh 1..N positions
    for idx, (_, w) in enumerate(ordered, start=1):
        li = BNode()
        g_set.add((setlist, ITEMLISTELEMENT, li))
        g_set.add((li, RDF.type, LISTITEM))
        g_set.add((li, ITEM, w))
        g_set.add((li, POSITION, Literal(idx)))

    # return how many items now
    return len(ordered)

def main():
    ap = argparse.ArgumentParser(description="Deduplicate setlist ListItems and renumber positions.")
    ap.add_argument("--setlists-in",  default="kg/40_setlists_songs.ttl")
    ap.add_argument("--setlists-out", default=None)
    ap.add_argument("--ref-graph",    default="kg/33_recordings_works.ttl",
                    help="Reference graph used to read owl:sameAs MBIDs for deduping (optional but recommended).")
    ap.add_argument("--use-mbid",     action="store_true",
                    help="Treat works with the same MB Work ID as the same song.")
    ap.add_argument("--inplace",      action="store_true",
                    help="Overwrite input (writes a .bak).")
    args = ap.parse_args()

    set_in = Path(args.setlists_in)
    set_out = Path(args.setlists_out) if args.setlists_out else set_in.with_suffix(".setclean.ttl")

    g_set = load_graph(set_in)
    g_ref = None
    if args.ref_graph and Path(args.ref_graph).exists():
        g_ref = load_graph(Path(args.ref_graph))

    sets = find_itemlists(g_set)
    total_before = 0
    total_after  = 0
    for s in sets:
        before = len(collect_items(g_set, s))
        after  = clean_setlist(g_set, g_ref, s, args.use_mbid)
        total_before += before
        total_after  += after

    if args.inplace:
        bak = set_in.with_suffix(set_in.suffix + ".bak")
        bak.write_text(g_set.serialize(format="turtle"), encoding="utf-8")  # backup current state
        save_graph(g_set, set_in)
        out_path = set_in
    else:
        save_graph(g_set, set_out)
        out_path = set_out

    print(f"[setlists] ItemLists: {len(sets)} | items before: {total_before} → after: {total_after}")
    print(f"[write] {out_path}")

if __name__ == "__main__":
    main()
