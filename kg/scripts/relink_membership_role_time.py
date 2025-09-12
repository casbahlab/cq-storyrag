#!/usr/bin/env python3
# relink_membership_role_time.py
import argparse
from rdflib import Graph, URIRef, RDF, Namespace

MM   = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
CORE = Namespace("https://w3id.org/polifonia/ontology/core/")
EX   = Namespace("http://wembrewind.live/ex#")

def load_graphs(paths, forced_format=None):
    g = Graph()
    for p in paths:
        if forced_format:
            g.parse(p, format=forced_format)
            continue
        ok = False
        for fmt in ["turtle","xml","n3","nt","trig","json-ld"]:
            try:
                g.parse(p, format=fmt)
                ok = True
                break
            except Exception:
                pass
        if not ok:
            raise RuntimeError(f"Failed to parse {p}")
    return g

def main():
    ap = argparse.ArgumentParser(description="Relink memberships to role/timeInterval based on URI prefix patterns.")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input RDF files (TTL/RDF/JSON-LD).")
    ap.add_argument("--out", required=True, help="Output TTL with relinking triples.")
    ap.add_argument("--format", default=None, help="Force parse format for inputs (ttl, xml, nt, json-ld).")
    ap.add_argument("--prop-role", dest="prop_role", default=str(EX.hasRole), help="IRI for membership->role linking predicate.")
    ap.add_argument("--prop-time", dest="prop_time", default=str(EX.hasTimeInterval), help="IRI for membership->timeInterval linking predicate.")
    args = ap.parse_args()

    g = load_graphs(args.inputs, forced_format=args.format)
    mem_cls = MM.MusicEnsembleMembership
    role_cls = CORE.Role
    ti_cls   = CORE.TimeInterval

    memberships = set(s for s in g.subjects(RDF.type, mem_cls))
    roles = set(s for s in g.subjects(RDF.type, role_cls))
    tis   = set(s for s in g.subjects(RDF.type, ti_cls))

    prop_role = URIRef(args.prop_role)
    prop_time = URIRef(args.prop_time)

    out = Graph()
    out.bind("mm", MM)
    out.bind("core", CORE)
    out.bind("ex", EX)

    added_role = added_time = 0

    role_by_prefix = {}
    for r in roles:
        s = str(r)
        if "_role_" in s:
            base = s.split("_role_")[0]
            role_by_prefix.setdefault(base, []).append(r)

    ti_by_prefix = {}
    for t in tis:
        s = str(t)
        if s.endswith("_interval"):
            base = s[: -len("_interval")]
            ti_by_prefix.setdefault(base, []).append(t)

    for m in memberships:
        ms = str(m)
        for r in role_by_prefix.get(ms, []):
            if (m, prop_role, r) not in g:
                out.add((m, prop_role, r))
                added_role += 1
        for t in ti_by_prefix.get(ms, []):
            if (m, prop_time, t) not in g:
                out.add((m, prop_time, t))
                added_time += 1

    out.serialize(destination=args.out, format="turtle")
    print(f"[link] memberships: {len(memberships)} roles: {len(roles)} timeIntervals: {len(tis)}")
    print(f"[link] added membership -> role links: {added_role}")
    print(f"[link] added membership -> timeInterval links: {added_time}")
    print(f"[write] {args.out}")

if __name__ == "__main__":
    main()
