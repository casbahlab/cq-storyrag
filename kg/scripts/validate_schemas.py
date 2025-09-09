#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_schemas.py

Validates instance RDF against OWL/Turtle schemas.

Now supports:
  --schema-dir <DIR>   # load ALL schema files from a folder (e.g., core.owl, musicmeta.owl, liveaid_schema.ttl)
  --schemas <files...> # or list specific schema files
  --instances <files...> and/or --instances-dir <DIR>

Checks:
  - UNKNOWN_PROPERTY (unless whitelisted with --allow-external)
  - PROPERTY_TYPE_MISMATCH (object vs datatype)
  - DOMAIN_*/RANGE_* issues (with subClass/subProperty inheritance)
  - OWL restrictions on classes (some/all/cardinalities)
  - owl:FunctionalProperty violations

Outputs CSV report + console summary. Exits 1 with --strict if any ERRORs.

Usage:
  python validate_schemas.py --schema-dir /path/to/schemas \
    --instances /path/to/liveaid_instances_master.ttl \
    --report /path/to/validation_report.csv \
    --allow-external http://schema.org/ --strict
"""
import argparse, csv, sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Set, Tuple, Iterable, Optional, List

from rdflib import Graph, URIRef, BNode, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

Issue = Tuple[str, str, str, str, str]

# ---------- utils ----------
def guess_format(path: str) -> str:
    p = path.lower()
    if p.endswith((".ttl", ".turtle")): return "turtle"
    if p.endswith((".nt", ".ntriples")): return "nt"
    if p.endswith((".nq", ".nquads")): return "nquads"
    if p.endswith((".trig",)): return "trig"
    return "xml"  # RDF/XML & OWL

def to_qname(g: Graph, node) -> str:
    try:
        return g.namespace_manager.normalizeUri(node) if isinstance(node, URIRef) else str(node)
    except Exception:
        return str(node)

def is_literal(n): return isinstance(n, Literal)
def is_iri(n):     return isinstance(n, URIRef)
def is_bnode(n):   return isinstance(n, BNode)

def gather_files_from_dir(d: Path) -> List[Path]:
    exts = (".ttl",".turtle",".owl",".rdf",".xml",".nt",".ntriples",".trig",".nq",".nquads")
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

# ---------- ontology helpers ----------
def subclass_closure(g: Graph) -> Dict[URIRef, Set[URIRef]]:
    parents = defaultdict(set)
    for c, _, sup in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(c,(URIRef,BNode)) and isinstance(sup,URIRef):
            parents[c].add(sup)
    clos = defaultdict(set)
    def dfs(c):
        if c in clos: return clos[c]
        sup = set([c]) if isinstance(c,URIRef) else set()
        for p in parents.get(c, []): sup |= dfs(p)
        clos[c] = sup; return sup
    for c in list(parents.keys()): dfs(c)
    return clos

def subprop_closure(g: Graph) -> Dict[URIRef, Set[URIRef]]:
    parents = defaultdict(set)
    for p, _, sup in g.triples((None, RDFS.subPropertyOf, None)):
        if isinstance(p,URIRef) and isinstance(sup,URIRef):
            parents[p].add(sup)
    clos = defaultdict(set)
    def dfs(p):
        if p in clos: return clos[p]
        sup = set([p])
        for sp in parents.get(p, []): sup |= dfs(sp)
        clos[p] = sup; return sup
    for p in list(parents.keys()): dfs(p)
    return clos

def prop_kinds_and_functional(g: Graph):
    kinds = defaultdict(set)
    for p in g.subjects(RDF.type, OWL.ObjectProperty):   kinds[p].add("object")
    for p in g.subjects(RDF.type, OWL.DatatypeProperty): kinds[p].add("datatype")
    for p in g.subjects(RDF.type, OWL.AnnotationProperty): kinds[p].add("annotation")
    for p in g.subjects(RDF.type, RDF.Property): kinds[p].add("rdf")
    functional = set(p for p in g.subjects(RDF.type, OWL.FunctionalProperty))
    return kinds, functional

def effective_domains_ranges(g: Graph, spc: Dict[URIRef, Set[URIRef]]):
    doms = defaultdict(set); rngs = defaultdict(set)
    for p, _, d in g.triples((None, RDFS.domain, None)):
        if isinstance(p,URIRef) and isinstance(d,URIRef): doms[p].add(d)
    for p, _, r in g.triples((None, RDFS.range, None)):
        if isinstance(p,URIRef) and isinstance(r,URIRef): rngs[p].add(r)
    eff_d = defaultdict(set); eff_r = defaultdict(set)
    for p in set(list(doms.keys()) + list(rngs.keys()) + list(spc.keys())):
        for sp in spc.get(p,{p}):
            eff_d[p] |= doms.get(sp,set())
            eff_r[p] |= rngs.get(sp,set())
    return eff_d, eff_r

def types_with_supers(g: Graph, s, scc) -> Set[URIRef]:
    out = set()
    for _,_,t in g.triples((s, RDF.type, None)):
        if isinstance(t,URIRef):
            out.add(t); out |= scc.get(t,set())
    return out

def class_restrictions(g: Graph):
    restr = defaultdict(list)
    for cls, _, r in g.triples((None, RDFS.subClassOf, None)):
        if not isinstance(cls, URIRef): continue
        if (r, RDF.type, OWL.Restriction) not in g and not list(g.triples((r, OWL.onProperty, None))):
            continue
        p = next((o for _,_,o in g.triples((r, OWL.onProperty, None)) if isinstance(o,URIRef)), None)
        if p is None: continue
        entry = {"onProperty": p}
        for (pred, key) in [(OWL.someValuesFrom,"some"), (OWL.allValuesFrom,"all"),
                            (OWL.cardinality,"card"), (OWL.minCardinality,"min"),
                            (OWL.maxCardinality,"max")]:
            val = next((o for _,_,o in g.triples((r, pred, None))), None)
            if val is not None:
                entry[key] = int(str(val)) if key in {"card","min","max"} else val
        restr[cls].append(entry)
    return restr

# ---------- validation ----------
def validate(g_schema: Graph, g_data: Graph, report_csv: Path, allow_external: Set[str], strict: bool) -> int:
    issues: List[Issue] = []

    scc = subclass_closure(g_schema)
    spc = subprop_closure(g_schema)
    kinds, functional = prop_kinds_and_functional(g_schema)
    eff_dom, eff_rng = effective_domains_ranges(g_schema, spc)
    restr = class_restrictions(g_schema)

    declared_props = set(kinds.keys()) | set(eff_dom.keys()) | set(eff_rng.keys()) \
                     | set(p for p,_,_ in g_schema.triples((None, RDF.type, RDF.Property)))

    for s,p,o in g_data:
        if not isinstance(p, URIRef): continue
        # Unknown property (unless whitelisted)
        if p not in declared_props and not any(str(p).startswith(pref) for pref in allow_external):
            issues.append(("WARNING","UNKNOWN_PROPERTY", to_qname(g_data,s), to_qname(g_data,p),
                           f"Not declared in provided schemas: {p}"))
        # Property type
        pk = set()
        for q in spc.get(p,{p}): pk |= kinds.get(q,set())
        if "object" in pk and "datatype" not in pk and is_literal(o):
            issues.append(("ERROR","PROPERTY_TYPE_MISMATCH", to_qname(g_data,s), to_qname(g_data,p),
                           f"ObjectProperty used with Literal: {repr(o)}"))
        if "datatype" in pk and "object" not in pk and not is_literal(o):
            issues.append(("ERROR","PROPERTY_TYPE_MISMATCH", to_qname(g_data,s), to_qname(g_data,p),
                           f"DatatypeProperty used with non-Literal: {to_qname(g_data,o)}"))
        # Domain
        doms = eff_dom.get(p,set())
        if doms:
            st = types_with_supers(g_data, s, scc)
            if not st:
                issues.append(("WARNING","DOMAIN_UNTYPED", to_qname(g_data,s), to_qname(g_data,p),
                               f"No rdf:type on subject; expected {', '.join(to_qname(g_data,d) for d in doms)}"))
            elif not any(any(d in scc.get(t,{t}) or d==t for t in st) for d in doms):
                issues.append(("ERROR","DOMAIN_VIOLATION", to_qname(g_data,s), to_qname(g_data,p),
                               f"Subject types {', '.join(to_qname(g_data,t) for t in st)} not in domain {', '.join(to_qname(g_data,d) for d in doms)}"))
        # Range
        rngs = eff_rng.get(p,set())
        if rngs:
            if is_literal(o):
                dt = o.datatype or XSD.string
                if not (RDFS.Literal in rngs or dt in rngs):
                    issues.append(("ERROR","RANGE_DT_VIOLATION", to_qname(g_data,s), to_qname(g_data,p),
                                   f"Literal datatype {to_qname(g_data,dt)} not in range {', '.join(to_qname(g_data,r) for r in rngs)}"))
            else:
                ot = types_with_supers(g_data, o, scc)
                if not ot:
                    issues.append(("WARNING","RANGE_UNTYPED", to_qname(g_data,s), to_qname(g_data,p),
                                   f"No rdf:type on object; expected {', '.join(to_qname(g_data,r) for r in rngs)}"))
                elif not any(any(r in scc.get(t,{t}) or r==t for t in ot) for r in rngs):
                    issues.append(("ERROR","RANGE_VIOLATION", to_qname(g_data,s), to_qname(g_data,p),
                                   f"Object types {', '.join(to_qname(g_data,t) for t in ot)} not in range {', '.join(to_qname(g_data,r) for r in rngs)}"))

    # Functional properties
    for p in functional:
        vals_by_s = defaultdict(set)
        for s,_,o in g_data.triples((None,p,None)):
            vals_by_s[s].add(o)
        for s,vals in vals_by_s.items():
            if len(vals) > 1:
                issues.append(("ERROR","FUNCTIONAL_VIOLATION", to_qname(g_data,s), to_qname(g_data,p),
                               f"Functional property has {len(vals)} values."))

    # Restrictions
    inst_types = defaultdict(set)
    for s,_,t in g_data.triples((None,RDF.type,None)):
        if isinstance(t,URIRef): inst_types[s].add(t)
    for s in list(inst_types.keys()):
        for t in list(inst_types[s]): inst_types[s] |= scc.get(t,set())

    # accumulate restrictions from supertypes
    restr_inh = defaultdict(list)
    for cls, rs in restr.items():
        acc = list(rs)
        for sup in scc.get(cls,set()): acc.extend(restr.get(sup,[]))
        restr_inh[cls] = acc

    for inst, tset in inst_types.items():
        rs = []
        for c in tset: rs.extend(restr_inh.get(c,[]))
        for r in rs:
            p = r.get("onProperty"); vals = list(g_data.objects(inst, p))
            if "some" in r:
                svf = r["some"]
                if not vals:
                    issues.append(("ERROR","RESTRICTION_SOME_MISSING", to_qname(g_data,inst), to_qname(g_data,p),
                                   f"Requires someValuesFrom {to_qname(g_data,svf)} but property is missing."))
                else:
                    ok = False
                    if vals and is_literal(vals[0]):
                        for v in vals:
                            dt = v.datatype or XSD.string
                            if svf == RDFS.Literal or dt == svf: ok = True; break
                    else:
                        for v in vals:
                            vtypes = types_with_supers(g_data, v, scc)
                            if svf in vtypes: ok = True; break
                    if not ok:
                        issues.append(("ERROR","RESTRICTION_SOME_TYPE_MISSING", to_qname(g_data,inst), to_qname(g_data,p),
                                       f"Has values but none of required type {to_qname(g_data,svf)}."))
            if "all" in r and vals:
                avf = r["all"]
                if vals and is_literal(vals[0]):
                    for v in vals:
                        dt = v.datatype or XSD.string
                        if not (avf == RDFS.Literal or dt == avf):
                            issues.append(("ERROR","RESTRICTION_ALL_VIOLATION", to_qname(g_data,inst), to_qname(g_data,p),
                                           f"Literal datatype {to_qname(g_data,dt)} not {to_qname(g_data,avf)}"))
                else:
                    for v in vals:
                        if avf not in types_with_supers(g_data, v, scc):
                            issues.append(("ERROR","RESTRICTION_ALL_VIOLATION", to_qname(g_data,inst), to_qname(g_data,p),
                                           f"Value {to_qname(g_data,v)} not of type {to_qname(g_data,avf)}"))
            for key,msg in [("card","exactly"),("min","at least"),("max","at most")]:
                if key in r:
                    n = r[key]
                    count = len(vals)
                    bad = (key=="card" and count!=n) or (key=="min" and count<n) or (key=="max" and count>n)
                    if bad:
                        issues.append(("ERROR","CARDINALITY_VIOLATION", to_qname(g_data,inst), to_qname(g_data,p),
                                       f"Expected {msg} {n} values, found {count}."))

    # Report
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["severity","code","subject","predicate","message"]); w.writerows(issues)

    counts = Counter(code for (_,code,_,_,_) in issues)
    sev_counts = Counter(sev for (sev,_,_,_,_) in issues)
    print("Validation summary"); print("------------------")
    for sev,c in sev_counts.items(): print(f"{sev:7s}: {c}")
    for code,c in counts.items():    print(f"{code:28s}: {c}")
    print(f"\nReport written to: {report_csv}")
    return 1 if strict and any(sev=="ERROR" for (sev,_,_,_,_) in issues) else 0

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Validate instance RDF against OWL/Turtle schemas.")
    ap.add_argument("--schema-dir", type=str, help="Directory containing schema files (e.g., core.owl, musicmeta.owl, liveaid_schema.ttl)")
    ap.add_argument("--schemas", nargs="*", default=[], help="Specific schema files")
    ap.add_argument("--instances-dir", type=str, help="Directory containing instance files")
    ap.add_argument("--instances", nargs="*", default=[], help="Specific instance files")
    ap.add_argument("--report", required=True, help="CSV report output path")
    ap.add_argument("--allow-external", nargs="*", default=["http://schema.org/"], help="Prefixes to allow as external properties")
    ap.add_argument("--strict", action="store_true", help="Exit code 1 if any ERRORs are found")
    args = ap.parse_args()

    schema_files = []
    if args.schema_dir:
        d = Path(args.schema_dir)
        if not d.is_dir(): print(f"--schema-dir not a directory: {d}", file=sys.stderr); sys.exit(2)
        schema_files += [str(p) for p in gather_files_from_dir(d)]
    schema_files += args.schemas
    if not schema_files:
        print("Provide --schema-dir or --schemas", file=sys.stderr); sys.exit(2)

    instance_files = []
    if args.instances_dir:
        d = Path(args.instances_dir)
        if not d.is_dir(): print(f"--instances-dir not a directory: {d}", file=sys.stderr); sys.exit(2)
        instance_files += [str(p) for p in gather_files_from_dir(d)]
    instance_files += args.instances
    if not instance_files:
        print("Provide --instances-dir or --instances", file=sys.stderr); sys.exit(2)

    g_schema = Graph(); g_schema.bind("schema", SCHEMA); g_schema.bind("core", CORE); g_schema.bind("mm", MM)
    for f in schema_files:
        g_schema.parse(f, format=guess_format(f))

    g_data = Graph(); g_data.bind("schema", SCHEMA); g_data.bind("core", CORE); g_data.bind("mm", MM)
    for f in instance_files:
        g_data.parse(f, format=guess_format(f))

    exit_code = validate(g_schema, g_data, Path(args.report), set(args.allow_external or []), args.strict)
    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
