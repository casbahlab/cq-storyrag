#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
no_shapes_validator.py  —  SHACL-free validator with multi-level inheritance and fix checklists.

What it does
- Validates DATA triples (ABox) against SCHEMA (TBox) using:
  * rdfs:subPropertyOf* (super-properties)
  * rdfs:subClassOf* (deep inheritance)
  * owl:equivalentClass (on either side)
  * owl:unionOf in domain/range (collects all members)
  * schema:domainIncludes / schema:rangeIncludes
- Emits a detailed CSV of issues (errors/warnings/info).
- Optional: emits aggregated "fix checklist" CSVs grouped by property and error type.

Usage
  python no_shapes_validator.py \
    --data /path/to/data_dir \
    --schemas /path/to/schemas_dir \
    --file-mask "*.ttl" \
    --out validation_report.csv \
    --checklist-prefix fixcheck

Outputs when --checklist-prefix is provided:
  fixcheck_overview.csv              # count of each error/warn/info per property
  fixcheck_domain_mismatches.csv     # domain failures: subject, types, expected domains
  fixcheck_range_mismatches_iri.csv  # range failures for IRI objects
  fixcheck_range_mismatches_lit.csv  # range failures for literal objects
  fixcheck_untyped_nodes.csv         # subject/object untyped warnings
  fixcheck_missing_schema.csv        # properties lacking domain/range declarations
"""

import argparse
import csv
from urllib.parse import urlparse
from rdflib import Graph, URIRef, BNode, Literal, Namespace, RDF, RDFS, XSD
from rdflib.namespace import OWL
import pathlib
from collections import defaultdict, Counter

SCHEMA = Namespace("http://schema.org/")

def parse_args():
    ap = argparse.ArgumentParser(description="Validate RDF against ontology domains/ranges without SHACL.")
    ap.add_argument("--data", required=True, help="Path to data directory or a single RDF file.")
    ap.add_argument("--schemas", required=True, help="Path to schema/ontology directory or a single RDF file.")
    ap.add_argument("--out", default="validation_report.csv", help="Output CSV path.")
    ap.add_argument("--format", default=None, help="Force rdflib parse format (ttl, xml, nt, json-ld, ...)")
    ap.add_argument("--file-mask", default=None, help="Glob mask like *.ttl to limit loaded files.")
    ap.add_argument("--checklist-prefix", default=None, help="Prefix for aggregated 'fix checklist' CSV outputs.")
    return ap.parse_args()

def iter_files(path, mask=None):
    p = pathlib.Path(path)
    if p.is_file():
        yield p
        return
    pattern = mask or "*"
    for ext in ["", ".ttl", ".rdf", ".owl", ".nt", ".nq", ".jsonld", ".trig", ".n3"]:
        for f in p.rglob(pattern if mask else f"*{ext}"):
            if f.is_file():
                yield f

def load_graph(path, mask=None, force_format=None):
    g = Graph()
    for f in iter_files(path, mask):
        tried = []
        if force_format:
            try:
                g.parse(f.as_posix(), format=force_format)
                continue
            except Exception as e:
                tried.append((force_format, str(e)))
        for fmt in ["turtle", "xml", "n3", "nt", "trig", "json-ld"]:
            try:
                g.parse(f.as_posix(), format=fmt)
                break
            except Exception as e:
                tried.append((fmt, str(e)))
        else:
            print(f"[WARN] Failed to parse {f} with formats: {tried}")
    return g

def ask_subclass_or_equal(schema_g, c1, c2):
    """True if c1 ≡/⊑* c2 using rdfs:subClassOf*, owl:equivalentClass on either side."""
    if c1 == c2:
        return True
    q = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    ASK {
      {
        ?c1 rdfs:subClassOf* ?c2 .
      }
      UNION
      {
        ?c1 (owl:equivalentClass|^owl:equivalentClass)/rdfs:subClassOf* ?c2 .
      }
    }
    """
    return bool(schema_g.query(q, initBindings={"c1": c1, "c2": c2}).askAnswer)

def q_schema_domains_ranges(schema_g, p):
    """Return (domains, ranges) for property p, including super-properties and owl:unionOf collections."""
    q = """
    PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema:<http://schema.org/>
    PREFIX owl:   <http://www.w3.org/2002/07/owl#>
    PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?d ?r WHERE {
      ?p rdfs:subPropertyOf* ?sp .

      # domains
      OPTIONAL { ?sp rdfs:domain ?d . }
      OPTIONAL { ?sp schema:domainIncludes ?d . }
      OPTIONAL {
        ?sp rdfs:domain [ owl:unionOf ?L ] .
        ?L rdf:rest*/rdf:first ?d .
      }
      OPTIONAL {
        ?sp schema:domainIncludes [ owl:unionOf ?L2 ] .
        ?L2 rdf:rest*/rdf:first ?d .
      }

      # ranges
      OPTIONAL { ?sp rdfs:range ?r . }
      OPTIONAL { ?sp schema:rangeIncludes ?r . }
      OPTIONAL {
        ?sp rdfs:range [ owl:unionOf ?L3 ] .
        ?L3 rdf:rest*/rdf:first ?r .
      }
      OPTIONAL {
        ?sp schema:rangeIncludes [ owl:unionOf ?L4 ] .
        ?L4 rdf:rest*/rdf:first ?r .
      }
    }
    """
    domains, ranges = set(), set()
    for d, r in schema_g.query(q, initBindings={"p": p}):
        if d: domains.add(d)
        if r: ranges.add(r)
    return domains, ranges

def q_data_types(data_g, node):
    return {t for _, _, t in data_g.triples((node, RDF.type, None))}

def literal_looks_like_url(lit: Literal):
    try:
        s = str(lit)
        u = urlparse(s)
        return bool(u.scheme and u.netloc)
    except Exception:
        return False

def validate_triple(schema_g, data_g, s, p, o):
    issues = []
    domains, ranges = q_schema_domains_ranges(schema_g, p)

    # DOMAIN
    if domains:
        s_types = q_data_types(data_g, s)
        if not s_types:
            issues.append({"severity":"warn","code":"subject_untyped",
                           "property":str(p),"subject":str(s),"object":str(o),
                           "expected_domain_any_of":"|".join(sorted(map(str, domains))),
                           "note":"Subject has no rdf:type in data; cannot verify domain."})
        else:
            ok = any(ask_subclass_or_equal(schema_g, st, d) for st in s_types for d in domains)
            if not ok:
                issues.append({"severity":"error","code":"domain_mismatch",
                               "property":str(p),"subject":str(s),"object":str(o),
                               "subject_types":"|".join(sorted(map(str, s_types))),
                               "expected_domain_any_of":"|".join(sorted(map(str, domains))),
                               "note":"No subject type compatible with any declared domain (via rdfs:subClassOf*/equiv)."})
    else:
        issues.append({"severity":"info","code":"domain_unknown",
                       "property":str(p),"subject":str(s),"object":str(o),
                       "note":"No domain/domainIncludes found for property (or its super-properties)."})
    # RANGE
    if ranges:
        if isinstance(o, Literal):
            dt = o.datatype or XSD.string
            def lit_ok():
                for r in ranges:
                    if r == RDFS.Literal:
                        return True
                    if isinstance(r, URIRef):
                        if str(r).startswith(str(XSD)):
                            if r == dt:
                                return True
                        elif r == SCHEMA.Text:
                            if dt in (XSD.string, None):
                                return True
                        elif r == SCHEMA.URL:
                            if dt == XSD.anyURI or literal_looks_like_url(o):
                                return True
                return False
            if not lit_ok():
                issues.append({"severity":"error","code":"range_mismatch_literal",
                               "property":str(p),"subject":str(s),"object":str(o),
                               "object_datatype":str(dt),
                               "expected_range_any_of":"|".join(sorted(map(str, ranges))),
                               "note":"Literal datatype not compatible with declared range."})
        else:
            if isinstance(o, BNode):
                issues.append({"severity":"warn","code":"blank_node_object",
                               "property":str(p),"subject":str(s),"object":str(o),
                               "expected_range_any_of":"|".join(sorted(map(str, ranges))),
                               "note":"Object is a blank node; type may be unknown."})
            else:
                o_types = q_data_types(data_g, o)
                if not o_types:
                    issues.append({"severity":"warn","code":"object_untyped",
                                   "property":str(p),"subject":str(s),"object":str(o),
                                   "expected_range_any_of":"|".join(sorted(map(str, ranges))),
                                   "note":"Object has no rdf:type in data; cannot verify range."})
                else:
                    ok = any(ask_subclass_or_equal(schema_g, ot, r) for ot in o_types for r in ranges)
                    if not ok:
                        issues.append({"severity":"error","code":"range_mismatch_iri",
                                       "property":str(p),"subject":str(s),"object":str(o),
                                       "object_types":"|".join(sorted(map(str, o_types))),
                                       "expected_range_any_of":"|".join(sorted(map(str, ranges))),
                                       "note":"No object type compatible with declared range (via rdfs:subClassOf*/equiv)."})
    else:
        issues.append({"severity":"info","code":"range_unknown",
                       "property":str(p),"subject":str(s),"object":str(o),
                       "note":"No range/rangeIncludes found for property (or its super-properties)."})
    return issues

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def emit_checklists(prefix, rows):
    # Overview counts per property and code
    ctr = Counter((r.get("property",""), r.get("code","")) for r in rows)
    overview = [
        {"property": p, "code": c, "count": n}
        for (p, c), n in sorted(ctr.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]
    write_csv(f"{prefix}_overview.csv", overview, ["property","code","count"])

    # Domain mismatches
    dom = [
        {"property": r.get("property",""),
         "subject": r.get("subject",""),
         "subject_types": r.get("subject_types",""),
         "expected_domains": r.get("expected_domain_any_of","")}
        for r in rows if r.get("code") == "domain_mismatch"
    ]
    write_csv(f"{prefix}_domain_mismatches.csv", dom,
              ["property","subject","subject_types","expected_domains"])

    # Range mismatches (IRI)
    rng_iri = [
        {"property": r.get("property",""),
         "object": r.get("object",""),
         "object_types": r.get("object_types",""),
         "expected_ranges": r.get("expected_range_any_of","")}
        for r in rows if r.get("code") == "range_mismatch_iri"
    ]
    write_csv(f"{prefix}_range_mismatches_iri.csv", rng_iri,
              ["property","object","object_types","expected_ranges"])

    # Range mismatches (literal)
    rng_lit = [
        {"property": r.get("property",""),
         "object": r.get("object",""),
         "object_datatype": r.get("object_datatype",""),
         "expected_ranges": r.get("expected_range_any_of","")}
        for r in rows if r.get("code") == "range_mismatch_literal"
    ]
    write_csv(f"{prefix}_range_mismatches_lit.csv", rng_lit,
              ["property","object","object_datatype","expected_ranges"])

    # Untyped nodes
    untyped = [
        {"code": r.get("code",""),
         "property": r.get("property",""),
         "who": "subject" if r.get("code")=="subject_untyped" else "object",
         "node": r.get("subject","") if r.get("code")=="subject_untyped" else r.get("object",""),
         "expected": r.get("expected_domain_any_of","") if r.get("code")=="subject_untyped" else r.get("expected_range_any_of","")}
        for r in rows if r.get("code") in ("subject_untyped","object_untyped")
    ]
    write_csv(f"{prefix}_untyped_nodes.csv", untyped,
              ["code","property","who","node","expected"])

    # Missing schema declarations
    missing = [
        {"property": r.get("property",""),
         "missing": "domain" if r.get("code")=="domain_unknown" else "range",
         "note": r.get("note","")}
        for r in rows if r.get("code") in ("domain_unknown","range_unknown")
    ]
    write_csv(f"{prefix}_missing_schema.csv", missing, ["property","missing","note"])

def main():
    args = parse_args()
    print("[load] SCHEMA graph...")
    schema_g = load_graph(args.schemas, mask=args.file_mask, force_format=args.format)
    schema_g.bind("schema", SCHEMA)

    print("[load] DATA graph...")
    data_g = load_graph(args.data, mask=args.file_mask, force_format=args.format)

    print("[scan] Validating triples...")
    rows = []
    count = 0
    for s, p, o in data_g:
        if p in (RDF.type, RDFS.subClassOf, RDFS.subPropertyOf, RDFS.domain, RDFS.range):
            continue
        for it in validate_triple(schema_g, data_g, s, p, o):
            rows.append(it)
        count += 1
        if count % 5000 == 0:
            print(f"  processed {count} triples...")

    print(f"[write] {args.out} with {len(rows)} rows")
    fieldnames = [
        "severity","code","property","subject","object",
        "subject_types","object_types","object_datatype",
        "expected_domain_any_of","expected_range_any_of","note"
    ]
    write_csv(args.out, rows, fieldnames)

    if args.checklist_prefix:
        print(f"[write] checklists with prefix '{args.checklist_prefix}_*.csv'")
        emit_checklists(args.checklist_prefix, rows)

if __name__ == "__main__":
    main()
