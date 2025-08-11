#!/usr/bin/env python3
import argparse, datetime
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, URIRef

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")
XSD    = Namespace("http://www.w3.org/2001/XMLSchema#")

PRESETS = {
    "liveaid": {
        # file-stem : {allowed parent types}
        "10_core_entities": {SCHEMA.Event, SCHEMA.Place, SCHEMA.Organization},
        "20_artists": {SCHEMA.Person, SCHEMA.MusicGroup, MM.Musician, MM.MusicArtist},
        "21_artist_labels": {SCHEMA.Person, SCHEMA.MusicGroup, MM.Musician, MM.MusicArtist},
        "30_performances": {MM.LivePerformance, EX.SongPerformance},
        "31_performance_labels": {MM.LivePerformance, EX.SongPerformance},
        "32_performance_labels_rdfs": {MM.LivePerformance, EX.SongPerformance},
        "40_setlists_songs": {SCHEMA.MusicComposition, SCHEMA.MusicRecording, SCHEMA.ItemList},
        "41_song_labels": {SCHEMA.MusicComposition, SCHEMA.MusicRecording},
        "50_instruments": {EX.MusicInstrument, SCHEMA.MusicInstrument},
        "51_instrument_labels": {EX.MusicInstrument, SCHEMA.MusicInstrument},
        "60_reviews": {SCHEMA.Review},
        "70_conditions": {SCHEMA.MedicalCondition},
    }
}

def backup(path: Path):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    b = path.with_suffix(path.suffix + f".bak_{ts}")
    b.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[backup] {b.name}")
    return b

def normalize_schema_http_term(t):
    if isinstance(t, URIRef):
        s = str(t)
        if s.startswith("https://schema.org/"):
            return URIRef(s.replace("https://schema.org/", "http://schema.org/"))
    return t

def expand_allowed_with_subclasses(master: Graph, allowed_types: set) -> set:
    """Include all subclasses of the allowed parent types (so Event keeps MusicEvent, Festival, etc.)."""
    closure = set(allowed_types)
    fringe = list(allowed_types)
    seen = set(fringe)
    while fringe:
        sup = fringe.pop()
        for sub in master.subjects(RDFS.subClassOf, sup):
            if isinstance(sub, URIRef) and sub not in seen:
                seen.add(sub)
                closure.add(sub)
                fringe.append(sub)
    return closure

def compute_keep_subjects(master: Graph, module_graph: Graph, allowed_types: set) -> set:
    """Decide which subjects in the module to keep based on their rdf:type in the master graph."""
    allowed = expand_allowed_with_subclasses(master, allowed_types)
    keep = set()
    for s in set(module_graph.subjects()):
        if not isinstance(s, URIRef):
            continue
        subj_types = set(master.objects(s, RDF.type))
        if subj_types & allowed:
            keep.add(s)
    return keep

def fresh_graph_with_prefixes() -> Graph:
    g = Graph()
    # Bind ONLY the prefixes we want; nothing else
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    g.bind("mm", MM)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    return g

def prune_file(master: Graph, module_path: Path, allowed_types: set, dry_run: bool, force: bool):
    mod = Graph().parse(str(module_path), format="turtle")

    keep_subjects = compute_keep_subjects(master, mod, allowed_types)
    drop_subjects = {s for s in set(mod.subjects()) if isinstance(s, URIRef) and s not in keep_subjects}

    # Safety rail: if we’re removing >80% of subjects and not forced, abort unless dry-run
    pct = (len(drop_subjects) / max(1, len(set(mod.subjects())))) * 100.0
    if pct > 80 and not (dry_run or force):
        print(f"[ABORT] Would remove {pct:.1f}% of subjects from {module_path.name}. Use --force or --dry-run first.")
        return

    # Build a NEW graph and copy only kept subjects, normalizing schema IRIs as we copy.
    out = fresh_graph_with_prefixes()
    removed_triples = 0
    kept_triples = 0
    for s in keep_subjects:
        for p, o in mod.predicate_objects(s):
            s2 = normalize_schema_http_term(s)
            p2 = normalize_schema_http_term(p)
            o2 = normalize_schema_http_term(o)
            out.add((s2, p2, o2))
            kept_triples += 1

    if dry_run:
        print(f"[dry-run] {module_path.name}: would keep {len(keep_subjects)} subjects, write {kept_triples} triples; drop {len(drop_subjects)} subjects.")
        return

    backup(module_path)
    out.serialize(destination=str(module_path), format="turtle")
    print(f"[prune] {module_path.name}: kept {len(keep_subjects)} subjects / wrote {kept_triples} triples (dropped {len(drop_subjects)} subjects).")
    # Bonus: tell you if schema1 leaked (it shouldn't)
    txt = Path(module_path).read_text(encoding="utf-8")
    if "schema1:" in txt:
        print("[warn] schema1 detected in output (unexpected).")
    else:
        print("[ok] prefixes clean (schema, mm, ex).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, help="Path to master TTL (type ground truth)")
    ap.add_argument("--preset", choices=PRESETS.keys(), help="Use preset allowlist by filename stem")
    ap.add_argument("--file", action="append", help="Module TTL to prune (repeatable)")
    ap.add_argument("--allow", action="append", help="Allowed rdf:type IRI (repeatable; only without --preset)")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write changes")
    ap.add_argument("--force", action="store_true", help="Allow large removals (>80%)")
    args = ap.parse_args()

    master = Graph().parse(args.kg, format="turtle")

    if args.preset:
        mapping = PRESETS[args.preset]
        if not args.file:
            raise SystemExit("[ERR] With --preset provide one or more --file paths to prune.")
        for f in args.file:
            p = Path(f)
            allowed = mapping.get(p.stem)
            if not allowed:
                print(f"[skip] No preset allowlist for {p.stem} (skipped)")
                continue
            prune_file(master, p, allowed, dry_run=args.dry_run, force=args.force)
    else:
        if not args.file or not args.allow:
            raise SystemExit("[ERR] Without --preset, use --file (repeatable) and --allow (repeatable).")
        allowed = set(URIRef(a) for a in args.allow)
        for f in args.file:
            prune_file(master, Path(f), allowed, dry_run=args.dry_run, force=args.force)

if __name__ == "__main__":
    main()
