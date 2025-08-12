#!/usr/bin/env python3
import argparse
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, RDF

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

def load_graph(p: Path) -> Graph:
    g = Graph()
    if p.exists():
        g.parse(p, format="turtle")
    # normalize https->http for schema.org if any crept in
    fixes = []
    for s, p_, o in g:
        s2, p2, o2 = s, p_, o
        if isinstance(p_, URIRef) and str(p_).startswith("https://schema.org/"):
            p2 = URIRef(str(p_).replace("https://", "http://"))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o2 = URIRef(str(o).replace("https://", "http://"))
        if (s2, p2, o2) != (s, p_, o):
            fixes.append(((s, p_, o), (s2, p2, o2)))
    for (s, p_, o), (s2, p2, o2) in fixes:
        g.remove((s, p_, o))
        g.add((s2, p2, o2))
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM)
    return g

def main():
    ap = argparse.ArgumentParser(
        description="Filter artists' mm:hasGenre triples to only those genres present in the genre file."
    )
    ap.add_argument("--artists-in",  default="kg/20_artists.ttl", help="Input artists TTL")
    ap.add_argument("--genres-in",   default="kg/11_genre.ttl",   help="Input genres TTL")
    ap.add_argument("--artists-out", default="kg/20_artists.genrefiltered.ttl", help="Output artists TTL")
    ap.add_argument(
        "--require-type", "--require_type",
        dest="require_type",
        action="store_true",
        help="Keep only objects typed as mm:MusicGenre (default: any subject present in genre file)."
    )
    args = ap.parse_args()

    artists_g = load_graph(Path(args.artists_in))
    genres_g  = load_graph(Path(args.genres_in))

    # Allowlist of valid genre nodes
    typed_nodes = set(genres_g.subjects(RDF.type, MM.MusicGenre))
    if args.require_type and typed_nodes:
        valid_genres = typed_nodes
    else:
        # any subject defined in the genre file counts as 'present'
        valid_genres = {s for s, _, _ in genres_g}

    total, kept, dropped = 0, 0, 0
    missing_objs = set()

    # Rebuild artists graph, skipping invalid mm:hasGenre triples
    out = Graph()
    for pfx, ns in artists_g.namespaces():
        out.bind(pfx, ns)

    for s, p, o in artists_g:
        if p == MM.hasGenre:
            total += 1
            if isinstance(o, URIRef) and o in valid_genres:
                out.add((s, p, o)); kept += 1
            else:
                dropped += 1
                if isinstance(o, URIRef):
                    missing_objs.add(o)
        else:
            out.add((s, p, o))

    out_path = Path(args.artists_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=str(out_path), format="turtle")

    print(f"[genres] valid nodes in genre file: {len(valid_genres)} "
          f"(typed mm:MusicGenre: {len(typed_nodes)})")
    print(f"[artists] mm:hasGenre total={total}, kept={kept}, dropped={dropped}")
    if missing_objs:
        print(f"[warn] {len(missing_objs)} referenced genres not present in genre file (first 10):")
        for u in list(sorted(map(str, missing_objs)))[:10]:
            print("  -", u)
    print(f"[write] {out_path}")

if __name__ == "__main__":
    main()
