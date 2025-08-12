#!/usr/bin/env python3
"""
One-click KG release (self-contained, strict vocab validation).

Pipeline:
  1) Merge modules -> kg/liveaid_instances_master.ttl
  2) Normalize schema prefixes (schema1: -> schema:, https->http)
  3) STRICT vocabulary validation (inline) against kg/schema/{schemaorg.ttl, musicmeta.owl, liveaid_schema.ttl}
  4) Run SHACL data-quality checks (pySHACL) if shapes present
  5) Run minimal SPARQL tests
  6) Run CQ coverage (if kg/cqs/run_cq_coverage.py exists)
  7) Freeze a versioned snapshot via freeze_master_snapshot.py
"""

import argparse, sys, subprocess, re
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCHEMA = ROOT / "schema"
CQS_DIR = ROOT / "cqs"
TESTS_DIR = ROOT / "tests"
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

def normalize_schema_prefixes(ttl_path: Path):
    """Ensure schema.org uses 'schema:' + HTTP IRIs; strip any 'schema1:' usage."""
    import re
    txt = ttl_path.read_text(encoding="utf-8")
    txt = txt.replace("https://schema.org/", "http://schema.org/")
    if "@prefix schema:" not in txt and "prefix schema:" not in txt:
        txt = txt.replace(
            "@prefix ex:",
            "@prefix schema: <http://schema.org/> .\n@prefix ex:"
        )
    txt = re.sub(r"(?im)^\s*@?prefix\s+schema1:\s*<https?://schema\.org/>\s*\.\s*$", "", txt)
    txt = txt.replace("schema1:", "schema:")
    ttl_path.write_text(txt, encoding="utf-8")
    print(f"[normalize] schema prefixes fixed in {ttl_path.name}")

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
        modules =  [
            #"10_core_entities.ttl",
            "11_genre.ttl", "12_event_broadcast_entities.ttl" , "13_city_country_venue.ttl" ,
            "14_organizations.ttl", "15_creativeevents.ttl" , "16_audience.ttl" , "17_miscellaneaous.ttl" ,
            "20_artists.ttl",
            "27_external_links_only.ttl",
            "27_removed_nonlink_content.ttl",
            # "21_solo_artists.ttl",
            # "22_music_groups.ttl",
            # "30_performances.ttl","31_songs.ttl",
            "32_albums.ttl", "33_recordings.ttl",
            "40_setlists_songs.ttl",
            "50_instruments.ttl",
            "60_reviews.ttl","70_conditions.ttl",
            "80_provenance.ttl","81_links_sameAs.ttl","82_external_ids_artists.ttl",
            "83_external_ids_songs.ttl","84_external_links_performances.ttl",
            "90_iconic_performances.ttl",
        ]
        g = Graph()
        for m in modules:
            p = ROOT / m
            if p.exists():
                g.parse(str(p), format="turtle")
        g.serialize(str(MASTER), format="turtle")
        print("[merge] Wrote", MASTER)

    # 2) Normalize schema prefixes
    normalize_schema_prefixes(MASTER)

    # 3) Strict vocab validation
    strict_vocab_validate(MASTER)

    # 4) SHACL data-quality (optional but recommended)
    shacl_shapes = SCHEMA / "91_shacl_data_quality.ttl"
    shacl_runner = SCRIPTS / "run_shacl.py"
    if shacl_shapes.exists() and shacl_runner.exists():
        report = ROOT / "validation" / "shacl_report.txt"
        (ROOT / "validation").mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(shacl_runner),
             "--data", str(MASTER),
             "--shapes", str(shacl_shapes),
             "--report", str(report)])
    else:
        print("[warn] SHACL skipped (missing shapes or runner)")

    # 5) Minimal SPARQL tests
    tests_runner = TESTS_DIR / "run_minimal_tests.py"
    if tests_runner.exists():
        run([sys.executable, str(tests_runner), "--kg", str(MASTER)])
    else:
        print("[warn] Minimal tests skipped (runner not found)")

    # 6) CQ coverage (optional)
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

    # 7) Freeze snapshot
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
