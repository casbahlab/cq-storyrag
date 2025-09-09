#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD
from calendar import monthrange
import argparse, re
from datetime import datetime

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def parse_span(lit: Literal):
    if not isinstance(lit, Literal):
        return None, None
    v = str(lit)
    # xsd:dateTime
    if lit.datatype == XSD.dateTime:
        try:
            dt = datetime.fromisoformat(v.replace("Z","+00:00")).replace(microsecond=0)
            s = dt.isoformat()
            return s, s
        except: pass
    # xsd:date or bare yyyy-mm-dd
    if lit.datatype == XSD.date or re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        try:
            y,m,d = map(int, v.split("-"))
            return f"{y:04d}-{m:02d}-{d:02d}T00:00:00", f"{y:04d}-{m:02d}-{d:02d}T23:59:59"
        except: pass
    # xsd:gYearMonth or bare yyyy-mm
    if lit.datatype == XSD.gYearMonth or re.fullmatch(r"\d{4}-\d{2}", v):
        try:
            y,m = map(int, v.split("-"))
            last = monthrange(y,m)[1]
            return f"{y:04d}-{m:02d}-01T00:00:00", f"{y:04d}-{m:02d}-{last:02d}T23:59:59"
        except: pass
    # xsd:gYear or bare yyyy
    if lit.datatype == XSD.gYear or re.fullmatch(r"\d{4}", v):
        try:
            y = int(v)
            return f"{y:04d}-01-01T00:00:00", f"{y:04d}-12-31T23:59:59"
        except: pass
    try:
        dt = datetime.fromisoformat(v).replace(microsecond=0)
        return dt.isoformat(), dt.isoformat()
    except:
        return None, None

def slugify(txt: str):
    return re.sub(r"[^a-z0-9]+","-", txt.strip().lower()).strip("-")

def main():
    ap = argparse.ArgumentParser(description="Translate mm:MusicEnsembleMemberships into core:TimeIndexedRole + core:Role.")
    ap.add_argument("--in",  dest="in_path",  required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--keep-schema-dates", action="store_true",
                    help="Keep schema:startDate/endDate on membership")
    ap.add_argument("--keep-roleName", action="store_true",
                    help="Keep schema:roleName on membership after creating core:Role nodes")
    ap.add_argument("--split-roles-on", default=",",
                    help="Delimiter for splitting schema:roleName (default: ,)")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_path, format="turtle")
    g.bind("schema", SCHEMA)
    g.bind("core", CORE)
    g.bind("mm", MM)

    memberships = set(g.subjects(RDF.type, MM.MusicEnsembleMembership))
    made_roles = 0
    typed = 0
    made_intervals = 0

    for m in memberships:
        # 1) Type as TimeIndexedRole
        if (m, RDF.type, CORE.TimeIndexedRole) not in g:
            g.add((m, RDF.type, CORE.TimeIndexedRole))
            typed += 1

        # 2) Agent (artist)
        for a in g.objects(m, MM.involvesMemberOfMusicEnsemble):
            g.add((m, CORE.hasAgent, a))

        # 3) Role(s) from schema:roleName
        role_names = []
        for rn in g.objects(m, SCHEMA.roleName):
            if isinstance(rn, Literal) and str(rn).strip():
                parts = [p.strip() for p in str(rn).split(args.split_roles_on) if p.strip()]
                role_names.extend(parts)

        # Also derive coarse roles from instruments if no roleName
        if not role_names:
            for inst in g.objects(m, SCHEMA.instrument):
                if isinstance(inst, URIRef):
                    # Use the localname as a role label proposal
                    role_names.append(inst.split("/")[-1])

        # Create one core:Role per unique label
        seen = set()
        for label in role_names:
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)

            # Make a stable URI if membership is a URIRef
            if isinstance(m, URIRef):
                ruri = URIRef(str(m) + "_role_" + slugify(label))
            else:
                ruri = BNode()

            g.add((ruri, RDF.type, CORE.Role))
            g.add((ruri, RDFS.label, Literal(label)))
            g.add((m, CORE.hasRole, ruri))
            made_roles += 1

        # 4) Time interval (reuse existing or create from schema dates)
        interval = next(g.objects(m, CORE.hasTimeInterval), None)
        if interval is None:
            # try to build from schema dates
            s_lit = next(g.objects(m, SCHEMA.startDate), None)
            e_lit = next(g.objects(m, SCHEMA.endDate), None)
            s_iso = e_iso = None
            if s_lit:
                s_iso, _ = parse_span(s_lit)
            if e_lit:
                _, e_iso = parse_span(e_lit)

            if s_iso or e_iso:
                interval = URIRef(str(m) + "_interval") if isinstance(m, URIRef) else BNode()
                g.add((m, CORE.hasTimeInterval, interval))
                g.add((interval, RDF.type, CORE.TimeInterval))
                if s_iso:
                    g.add((interval, CORE.startTime, Literal(s_iso, datatype=XSD.dateTime)))
                if e_iso:
                    g.add((interval, CORE.endTime,   Literal(e_iso, datatype=XSD.dateTime)))
                made_intervals += 1

        # 5) Optionally remove schema date + roleName literals
        if not args.keep_schema_dates:
            for x in list(g.objects(m, SCHEMA.startDate)):
                g.remove((m, SCHEMA.startDate, x))
            for x in list(g.objects(m, SCHEMA.endDate)):
                g.remove((m, SCHEMA.endDate, x))
        if not args.keep_roleName:
            for x in list(g.objects(m, SCHEMA.roleName)):
                g.remove((m, SCHEMA.roleName, x))

        # 6) Ensemble → we can optionally attach a named Role **about** the ensemble
        # If you want, you can add inverse link core:isRoleOf, but typically we attach
        # the ensemble via domain predicates already present (keep original mm triple).

    g.serialize(destination=args.out_path, format="turtle")
    print(f"Memberships: {len(memberships)} | typed as core:TimeIndexedRole: {typed}")
    print(f"Roles created: {made_roles} | Intervals created: {made_intervals}")
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
