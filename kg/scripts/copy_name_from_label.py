#!/usr/bin/env python3
import argparse
from pathlib import Path
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS

SCHEMA = Namespace("http://schema.org/")

def process_file(src: Path, dst: Path, inplace: bool) -> int:
    g = Graph()
    g.parse(src, format="turtle")
    g.bind("schema", SCHEMA)

    added = 0
    for s, l in g.subject_objects(RDFS.label):
        if (s, SCHEMA.name, l) not in g:
            g.add((s, SCHEMA.name, l))
            added += 1

    if inplace:
        # simple backup then overwrite
        bak = src.with_suffix(src.suffix + ".bak")
        bak.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        g.serialize(destination=str(src), format="turtle")
        print(f"[write] {src} (+{added} schema:name)  [backup: {bak.name}]")
    else:
        dst = dst or src.with_suffix(src.suffix.replace(".ttl", "") + ".named.ttl")
        g.serialize(destination=str(dst), format="turtle")
        print(f"[write] {dst} (+{added} schema:name)")

    return added

def main():
    ap = argparse.ArgumentParser(description="Copy rdfs:label -> schema:name in Turtle files (idempotent).")
    ap.add_argument("inputs", nargs="+", help="Input .ttl file(s)")
    ap.add_argument("-o", "--out", help="Single output file (only when one input). Otherwise writes *.named.ttl")
    ap.add_argument("-i", "--inplace", action="store_true", help="Overwrite input file(s) (writes a .bak)")
    args = ap.parse_args()

    if args.out and len(args.inputs) != 1:
        ap.error("--out can be used only with a single input file")

    total = 0
    for p in args.inputs:
        src = Path(p)
        dst = Path(args.out) if args.out else None
        total += process_file(src, dst, args.inplace)

    print(f"[done] Added {total} schema:name triple(s) total.")

if __name__ == "__main__":
    main()
