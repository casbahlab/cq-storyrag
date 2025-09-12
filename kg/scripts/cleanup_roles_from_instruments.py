#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup script: treat instruments as Roles for MusicEnsembleMemberships.
- For each mm:MusicEnsembleMembership:
  * ensure rdf:type core:TimeIndexedRole
  * for each schema:instrument, create/link a core:Role node with a sensible label
  * attach the instrument to that role via schema:instrument
  * optionally use schema:roleName to refine the role label (lead, rhythm, background)
  * optionally drop instruments on the membership, and drop the textual roleName
  * optionally create a core:TimeInterval from schema:startDate/endDate if present
Idempotent: will not duplicate roles if a role with the same label already exists for the membership.
"""
import argparse
import csv
import re
from calendar import monthrange
from datetime import datetime
from typing import Optional, Tuple

from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD, OWL

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

# Basic heuristic mapping from instrument label -> role label
# You can extend or override with --mapping-csv (columns: instrument_label,role_label)
INSTR_TO_ROLE = {
    "drum": "drummer",
    "drums": "drummer",
    "drum set": "drummer",
    "percussion": "percussionist",
    "guitar": "guitarist",
    "lead guitar": "lead guitarist",
    "rhythm guitar": "rhythm guitarist",
    "bass guitar": "bassist",
    "bass": "bassist",
    "keyboard": "keyboardist",
    "keyboards": "keyboardist",
    "piano": "pianist",
    "synth": "synthesist",
    "synthesizer": "synthesist",
    "voice": "vocalist",
    "vocal": "vocalist",
    "vocals": "vocalist",
    "background vocals": "background vocalist",
    "backing vocals": "background vocalist",
    "bgv": "background vocalist",
    "violin": "violinist",
    "cello": "cellist",
    "trumpet": "trumpeter",
    "sax": "saxophonist",
    "saxophone": "saxophonist",
    "harmonica": "harmonicist",
}

QUALIFIERS = {
    "lead": "lead",
    "rhythm": "rhythm",
    "background": "background",
    "backing": "background",
    "co-": "co",
    "co ": "co",
    "guest": "guest",
}


def slugify(txt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", txt.strip().lower()).strip("-")


def localname(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]


def load_mapping_csv(path: Optional[str]):
    if not path:
        return {}
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # expects: instrument_label,role_label
        for row in reader:
            il = (row.get("instrument_label") or "").strip().lower()
            rl = (row.get("role_label") or "").strip()
            if il and rl:
                mapping[il] = rl
    return mapping


def pick_role_label(instr_label: str, role_name_text: Optional[str], mapping_override: dict) -> str:
    base = instr_label.strip().lower()
    # direct override first
    if base in mapping_override:
        label = mapping_override[base]
    else:
        # heuristic dictionary
        best = None
        for key, val in INSTR_TO_ROLE.items():
            if key in base:
                best = val
                break
        label = best or base

    # refine with qualifiers from roleName text if present
    if role_name_text:
        rn = role_name_text.lower()
        for q, norm in QUALIFIERS.items():
            if q in rn and norm not in label:
                # inject qualifier ahead of head noun when makes sense
                # e.g., "guitarist" -> "lead guitarist"
                if "vocalist" in label and norm == "background":
                    label = "background vocalist"
                elif "guitarist" in label and norm in {"lead", "rhythm"}:
                    label = f"{norm} guitarist"
                elif norm == "guest":
                    label = f"guest {label}"
                elif norm == "co":
                    label = f"co-{label}"
                # else leave as is if no clean injection point
    return label


def parse_span(lit: Literal) -> Tuple[Optional[str], Optional[str]]:
    v = str(lit)
    if lit.datatype == XSD.dateTime:
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00")).replace(microsecond=0)
            s = dt.isoformat()
            return s, s
        except Exception:
            return None, None
    if lit.datatype == XSD.date or re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        y, m, d = map(int, v.split("-"))
        return f"{y:04d}-{m:02d}-{d:02d}T00:00:00", f"{y:04d}-{m:02d}-{d:02d}T23:59:59"
    if lit.datatype == XSD.gYearMonth or re.fullmatch(r"\d{4}-\d{2}", v):
        y, m = map(int, v.split("-"))
        last = monthrange(y, m)[1]
        return f"{y:04d}-{m:02d}-01T00:00:00", f"{y:04d}-{m:02d}-{last:02d}T23:59:59"
    if lit.datatype == XSD.gYear or re.fullmatch(r"\d{4}", v):
        y = int(v)
        return f"{y:04d}-01-01T00:00:00", f"{y:04d}-12-31T23:59:59"
    try:
        dt = datetime.fromisoformat(v).replace(microsecond=0)
        return dt.isoformat(), dt.isoformat()
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Clean memberships by turning instruments into core:Role nodes and linking them.")
    ap.add_argument("--in",  dest="in_path",  required=True, help="Input TTL")
    ap.add_argument("--out", dest="out_path", required=True, help="Output TTL")
    ap.add_argument("--drop-membership-instruments", action="store_true", help="Remove schema:instrument from membership after copying to roles")
    ap.add_argument("--drop-rolename", action="store_true", help="Remove schema:roleName literal after creating roles")
    ap.add_argument("--create-intervals", action="store_true", help="Create core:TimeInterval from schema:startDate/endDate if not present")
    ap.add_argument("--mapping-csv", help="CSV with columns instrument_label,role_label to override heuristics")
    args = ap.parse_args()

    mapping_override = load_mapping_csv(args.mapping_csv)

    g = Graph()
    g.parse(args.in_path, format="turtle")
    g.bind("schema", SCHEMA)
    g.bind("core", CORE)
    g.bind("mm", MM)

    memberships = set(g.subjects(RDF.type, MM.MusicEnsembleMembership))

    roles_created = 0
    intervals_created = 0
    typed_as_tir = 0

    for m in memberships:
        # type as TimeIndexedRole
        if (m, RDF.type, CORE.TimeIndexedRole) not in g:
            g.add((m, RDF.type, CORE.TimeIndexedRole))
            typed_as_tir += 1

        # grab one roleName text to detect qualifiers
        role_name_text = None
        for rn in g.objects(m, SCHEMA.roleName):
            if isinstance(rn, Literal) and str(rn).strip():
                role_name_text = str(rn)
                break

        # handle instruments -> roles
        instruments = list(g.objects(m, SCHEMA.instrument))
        for inst in instruments:
            # find a label for instrument
            inst_label = None
            if isinstance(inst, URIRef):
                lbl = next(g.objects(inst, RDFS.label), None)
                inst_label = str(lbl) if isinstance(lbl, Literal) else localname(inst)
            elif isinstance(inst, Literal):
                inst_label = str(inst)
            else:
                inst_label = "instrument"

            role_label = pick_role_label(inst_label, role_name_text, mapping_override)

            # check if a role with same label already linked
            existing = False
            for r in g.objects(m, CORE.hasRole):
                if (r, RDF.type, CORE.Role) in g:
                    lab = next(g.objects(r, RDFS.label), None)
                    if isinstance(lab, Literal) and lab.value.strip().lower() == role_label.strip().lower():
                        existing = True
                        # still ensure instrument is attached to the role
                        g.add((r, SCHEMA.instrument, inst))
                        break

            if existing:
                continue

            # make a role URI
            if isinstance(m, URIRef):
                ruri = URIRef(str(m) + "_role_" + slugify(role_label))
            else:
                ruri = BNode()

            # declare and link
            g.add((ruri, RDF.type, CORE.Role))
            g.add((ruri, RDFS.label, Literal(role_label)))
            g.add((m, CORE.hasRole, ruri))
            g.add((ruri, SCHEMA.instrument, inst))
            roles_created += 1

        # optionally drop membership instruments
        if args.drop_membership_instruments:
            for inst in instruments:
                g.remove((m, SCHEMA.instrument, inst))

        # optionally drop textual roleName
        if args.drop_rolename:
            for rn in list(g.objects(m, SCHEMA.roleName)):
                g.remove((m, SCHEMA.roleName, rn))

        # optional: create interval from schema:startDate/endDate if missing
        if args.create_intervals and not list(g.objects(m, CORE.hasTimeInterval)):
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
                    g.add((interval, CORE.EndTime, Literal(e_iso, datatype=XSD.dateTime)))  # NOTE: fix to CORE.endTime if your ontology uses lowercase 'e'
                intervals_created += 1

    g.serialize(destination=args.out_path, format="turtle")

    print(f"Memberships processed: {len(memberships)}")
    print(f"Typed as core:TimeIndexedRole: {typed_as_tir}")
    print(f"Roles created: {roles_created}")
    print(f"TimeIntervals created: {intervals_created}")
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()
