#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

SCHEMA_HTTP  = "http://schema.org/"
SCHEMA_HTTPS = "https://schema.org/"
SCHEMA = Namespace(SCHEMA_HTTP)
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

# Predicates (handle both http/https when reading)
IS_PART_OF = [URIRef(SCHEMA_HTTP + "isPartOf"), URIRef(SCHEMA_HTTPS + "isPartOf")]
PERFORMER  = [URIRef(SCHEMA_HTTP + "performer"), URIRef(SCHEMA_HTTPS + "performer")]
LOCATION   = [URIRef(SCHEMA_HTTP + "location"),  URIRef(SCHEMA_HTTPS + "location")]
NAME       = URIRef(SCHEMA_HTTP + "name")

def load_graph(p: Path) -> Graph:
    g = Graph()
    g.parse(p, format="turtle")
    # normalize https->http for schema.org
    changes = []
    for s, p_, o in g:
        p2, o2 = p_, o
        if isinstance(p_, URIRef) and str(p_).startswith(SCHEMA_HTTPS):
            p2 = URIRef(str(p_).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if isinstance(o, URIRef) and str(o).startswith(SCHEMA_HTTPS):
            o2 = URIRef(str(o).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if (p2, o2) != (p_, o):
            changes.append(((s, p_, o), (s, p2, o2)))
    for (s, p_, o), (s2, p2, o2) in changes:
        g.remove((s, p_, o))
        g.add((s2, p2, o2))
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("ex", EX); g.bind("rdfs", RDFS)
    return g

def first_obj_any(g: Graph, s: URIRef, props) -> Optional[URIRef]:
    for p in props:
        for o in g.objects(s, p):
            if isinstance(o, URIRef):
                return o
    return None

def label_of(g: Graph, u: URIRef) -> str:
    for o in g.objects(u, NAME):
        return str(o)
    for o in g.objects(u, RDFS.label):
        return str(o)
    # fallback to fragment/localname
    su = str(u)
    return su.split("#")[-1] if "#" in su else su.rsplit("/", 1)[-1]

def build_perf_name(g: Graph, perf: URIRef, append_location: bool) -> str:
    # Prefer existing rdfs:label if present
    for o in g.objects(perf, RDFS.label):
        return str(o)

    artist   = first_obj_any(g, perf, PERFORMER)
    event    = first_obj_any(g, perf, IS_PART_OF)
    loc      = first_obj_any(g, perf, LOCATION)

    a = label_of(g, artist) if artist else "Unknown Artist"
    e = label_of(g, event)  if event  else "Unknown Event"

    if append_location and loc is not None:
        l = label_of(g, loc)
        return f"{a} – {e} ({l})"
    return f"{a} – {e} performance"

def process_file(src: Path, dst: Optional[Path], inplace: bool, force: bool, append_location: bool):
    g = load_graph(src)

    # candidates: explicit LivePerformance OR nodes that have both event+artist
    candidates = set(s for s in g.subjects(RDF.type, MM.LivePerformance) if isinstance(s, URIRef))
    for s in g.subjects(IS_PART_OF[0], None):  # http version
        if isinstance(s, URIRef):
            if first_obj_any(g, s, PERFORMER):
                candidates.add(s)
    for s in g.subjects(IS_PART_OF[1], None):  # https (already normalized, but just in case)
        if isinstance(s, URIRef):
            if first_obj_any(g, s, PERFORMER):
                candidates.add(s)

    added = updated = skipped = 0
    for perf in sorted(candidates, key=str):
        has_name = (perf, NAME, None) in g
        if has_name and not force:
            skipped += 1
            continue
        new_val = build_perf_name(g, perf, append_location)
        if has_name and force:
            # remove existing names to avoid duplicates
            for o in list(g.objects(perf, NAME)):
                g.remove((perf, NAME, o))
            updated += 1
        else:
            added += 1
        g.add((perf, NAME, Literal(new_val)))

    # write
    if inplace:
        bak = src.with_suffix(src.suffix + ".bak")
        bak.write_text(g.serialize(format="turtle"), encoding="utf-8")  # backup current content
        g.serialize(destination=str(src), format="turtle")
        out_path = src
    else:
        out_path = dst or src.with_suffix(".perfnamed.ttl")
        g.serialize(destination=str(out_path), format="turtle")

    print(f"[file] {src.name}: performances={len(candidates)}  added={added}  updated={updated}  skipped={skipped}")
    print(f"[write] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Add schema:name to mm:LivePerformance nodes.")
    ap.add_argument("inputs", nargs="+", help="Input TTL files")
    ap.add_argument("-o", "--out", help="Single output file (only if one input). Else writes *.perfnamed.ttl")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input file(s) (writes a .bak)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing schema:name")
    ap.add_argument("--append-location", action="store_true", help="Append (Location) to the generated name when available")
    args = ap.parse_args()

    if args.out and len(args.inputs) != 1:
        ap.error("--out can only be used with a single input file")

    for i, p in enumerate(args.inputs):
        src = Path(p)
        dst = Path(args.out) if (i == 0 and args.out) else None
        process_file(src, dst, args.inplace, args.force, args.append_location)

if __name__ == "__main__":
    main()
