#!/usr/bin/env python3
"""
One-click KG release (self-contained, strict vocab validation).

Steps:
  1) Merge modules -> kg/liveaid_instances_master.ttl
  2) STRICT vocabulary validation (inline):
       - Loads ontologies from kg/schema/: schemaorg.ttl, musicmeta.owl, liveaid_schema.ttl
       - Builds allowed class/property sets + allows known namespaces
       - Fails release if any unknown class/property is detected
  3) Run CQ coverage if kg/run_cq_coverage.py exists
  4) Freeze a versioned snapshot via freeze_master_snapshot.py (with --no-strict to avoid re-check)

Usage:
  python kg/scripts/release_one_click.py
  python kg/scripts/release_one_click.py --label "Baseline after schema lock"
  python kg/scripts/release_one_click.py --major
  python kg/scripts/release_one_click.py --patch
"""

import argparse, sys, subprocess
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCHEMA = ROOT / "schema"
CQS_DIR = ROOT / "cqs"
MASTER = ROOT / "liveaid_instances_master.ttl"

ALLOWED_NAMESPACES = [
    "http://schema.org/",
    "https://w3id.org/polifonia/ontology/music-meta/",
    "http://wembrewind.live/ex#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
]

def run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(f"[ABORT] command failed: {' '.join(str(c) for c in cmd)}")

def normalize_schema_iri(iri: str) -> str:
    return iri.replace("https://schema.org/", "http://schema.org/")

def load_ontology_terms(path: Path):
    g = Graph()
    fmt = "xml" if path.suffix in (".owl", ".rdf", ".xml") else "turtle"
    g.parse(str(path), format=fmt)
    classes, properties = set(), set()
    for s, _, _ in g.triples((None, RDF.type, OWL.Class)): classes.add(str(s))
    for s, _, _ in g.triples((None, RDF.type, RDFS.Class)): classes.add(str(s))
    for s, _, _ in g.triples((None, RDF.type, RDF.Property)): properties.add(str(s))
    for s, _, _ in g.triples((None, RDF.type, OWL.ObjectProperty)): properties.add(str(s))
    for s, _, _ in g.triples((None, RDF.type, OWL.DatatypeProperty)): properties.add(str(s))
    for s, _, _ in g.triples((None, RDF.type, OWL.AnnotationProperty)): properties.add(str(s))
    # normalize schema.org IRIs
    classes = {normalize_schema_iri(x) for x in classes}
    properties = {normalize_schema_iri(x) for x in properties}
    return classes, properties

def strict_vocab_validate(master_path: Path):
    schema_ttl = SCHEMA / "schemaorg.ttl"
    mm_owl     = SCHEMA / "musicmeta.owl"
    custom_ttl = SCHEMA / "liveaid_schema.ttl"
    missing = [p for p in (schema_ttl, mm_owl, custom_ttl) if not p.exists()]
    if missing:
        raise SystemExit("[ABORT] Missing local ontology files for strict validation:\n - " + "\n - ".join(map(str, missing)))

    known_classes, known_props = set(), set()
    for p in (schema_ttl, mm_owl, custom_ttl):
        cset, pset = load_ontology_terms(p)
        known_classes |= cset
        known_props |= pset

    data = Graph().parse(str(master_path), format="turtle")

    used_classes, used_props = set(), set()
    for _, _, o in data.triples((None, RDF.type, None)):
        used_classes.add(normalize_schema_iri(str(o)))
    for _, p, _ in data.triples((None, None, None)):
        used_props.add(normalize_schema_iri(str(p)))

    def allowed_ns(iri: str) -> bool:
        iri = normalize_schema_iri(iri)
        return any(iri.startswith(ns) for ns in ALLOWED_NAMESPACES)

    unknown_classes = sorted(x for x in used_classes if (x not in known_classes and not allowed_ns(x)))
    unknown_props   = sorted(x for x in used_props   if (x not in known_props   and not allowed_ns(x)))

    if unknown_classes or unknown_props:
        print("[FAIL] Unknown vocabulary detected.")
        if unknown_classes:
            print("  Unknown classes:")
            for c in unknown_classes: print("   -", c)
        if unknown_props:
            print("  Unknown properties:")
            for p in unknown_props: print("   -", p)
        raise SystemExit("[ABORT] Strict vocab validation failed. Define new terms in kg/schema/liveaid_schema.ttl or adjust data.")

    print("[ok] Strict vocab validation passed.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="Optional label for the snapshot")
    bump = ap.add_mutually_exclusive_group()
    bump.add_argument("--major", action="store_true")
    bump.add_argument("--minor", action="store_true")
    bump.add_argument("--patch", action="store_true")
    args = ap.parse_args()

    # 1) Merge modules
    merge_py = SCRIPTS / "merge_all_ttls.py"
    if merge_py.exists():
        run([sys.executable, str(merge_py)])
    else:
        # Inline simple merge if helper missing
        modules = [
            "10_core_entities.ttl","20_artists.ttl","21_artist_labels.ttl",
            "30_performances.ttl","31_performance_labels.ttl","32_performance_labels_rdfs.ttl",
            "40_setlists_songs.ttl","41_song_labels.ttl",
            "50_instruments.ttl","51_instrument_labels.ttl",
            "60_reviews.ttl","70_conditions.ttl",
            "80_provenance.ttl","81_links_sameAs.ttl","82_external_ids_artists.ttl",
            "83_external_ids_songs.ttl","84_external_links_performances.ttl","85_artist_mbids.ttl",
        ]
        g = Graph()
        for m in modules:
            p = ROOT / m if m.startswith("10_") else ROOT / m  # simple join
            p = ROOT / m  # modules live under kg/
            if p.exists():
                g.parse(str(p), format="turtle")
        g.serialize(str(MASTER), format="turtle")
        print("[merge] Wrote", MASTER)

    # 2) Strict vocab validation (inline)
    strict_vocab_validate(MASTER)

    # 3) CQ coverage (optional)
    coverage_runner = SCRIPTS / "run_cq_coverage.py"
    if coverage_runner.exists():
        kg_file = ROOT / "liveaid_instances_master.ttl"
        cq_file = ROOT / "cqs" / "cqs_queries_template_filled_in.rq"
        out_file = ROOT / "coverage_summary.csv"

        run([
            sys.executable, str(coverage_runner),
            "--kg", str(kg_file),
            "--input", str(cq_file),
            "--out", str(out_file)
        ])
    else:
        print("[warn] CQ runner not found, skipping:", coverage_runner)

    # 4) Freeze snapshot (avoid duplicate validation by passing --no-strict)
    freezer = SCRIPTS / "freeze_master_snapshot.py"
    if not freezer.exists():
        raise SystemExit(f"[ERR] missing: {freezer}")
    bump_flag = "--minor"
    if args.major: bump_flag = "--major"
    elif args.patch: bump_flag = "--patch"
    cmd = [sys.executable, str(freezer), bump_flag]
    if args.label:
        cmd += ["--label", args.label]
    run(cmd)

    print("[DONE] One-click release completed.")

if __name__ == "__main__":
    main()
