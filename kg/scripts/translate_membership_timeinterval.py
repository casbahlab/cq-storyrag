#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, XSD
from calendar import monthrange
import argparse
import re
from datetime import datetime

SCHEMA = Namespace("http://schema.org/")
CORE   = Namespace("https://w3id.org/polifonia/ontology/core/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def parse_xsd_temporal(lit: Literal):
    """
    Accepts xsd:gYear, xsd:gYearMonth, xsd:date, xsd:dateTime (as strings).
    Returns (start_dt_iso, end_dt_iso) both as xsd:dateTime strings.
    """
    if not isinstance(lit, Literal):
        return None, None
    val = str(lit)

    # dateTime full
    if lit.datatype == XSD.dateTime:
        # For a single dateTime, use same moment for start/end with sensible end bump
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            start = dt.replace(microsecond=0).isoformat()
            # use same instant; if you prefer a tiny bump, uncomment next line
            # end = (dt + timedelta(seconds=0)).replace(microsecond=0).isoformat()
            end = start
            return start, end
        except Exception:
            pass

    # date (YYYY-MM-DD)
    if lit.datatype == XSD.date or re.fullmatch(r"\d{4}-\d{2}-\d{2}", val):
        try:
            y, m, d = map(int, val.split("-"))
            start = f"{y:04d}-{m:02d}-{d:02d}T00:00:00"
            end   = f"{y:04d}-{m:02d}-{d:02d}T23:59:59"
            return start, end
        except Exception:
            pass

    # gYearMonth (YYYY-MM)
    if lit.datatype == XSD.gYearMonth or re.fullmatch(r"\d{4}-\d{2}", val):
        try:
            y, m = map(int, val.split("-"))
            last_day = monthrange(y, m)[1]
            start = f"{y:04d}-{m:02d}-01T00:00:00"
            end   = f"{y:04d}-{m:02d}-{last_day:02d}T23:59:59"
            return start, end
        except Exception:
            pass

    # gYear (YYYY)
    if lit.datatype == XSD.gYear or re.fullmatch(r"\d{4}", val):
        try:
            y = int(val)
            start = f"{y:04d}-01-01T00:00:00"
            end   = f"{y:04d}-12-31T23:59:59"
            return start, end
        except Exception:
            pass

    # Fallback: try to coerce as date
    try:
        dt = datetime.fromisoformat(val)
        start = dt.replace(microsecond=0).isoformat()
        end   = start
        return start, end
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description="Translate schema:startDate/endDate on memberships into core:TimeInterval.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input TTL file (e.g., /mnt/data/20_artists.ttl)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output TTL file")
    ap.add_argument("--keep-schema-dates", action="store_true", help="Keep schema:startDate/endDate after translation")
    ap.add_argument("--interval-suffix", default="_interval", help="Suffix for generated interval node local names")
    args = ap.parse_args()

    g = Graph()
    g.parse(args.in_path, format="turtle")

    g.bind("schema", SCHEMA)
    g.bind("core", CORE)
    g.bind("mm", MM)

    # Collect memberships
    memberships = set(g.subjects(RDF.type, MM.MusicEnsembleMembership))

    created = 0
    updated = 0

    for m in memberships:
        # Get dates if present
        sdates = list(g.objects(m, SCHEMA.startDate))
        edates = list(g.objects(m, SCHEMA.endDate))

        if not sdates and not edates:
            # Nothing to translate; maybe already has core:hasTimeInterval
            continue

        # Prefer the first literal if multiples
        start_lit = sdates[0] if sdates else None
        end_lit   = edates[0] if edates else None

        start_iso = end_iso = None

        if start_lit:
            s_start, s_end = parse_xsd_temporal(start_lit)
            # For startDate, take the start of the parsed span
            start_iso = s_start
        if end_lit:
            e_start, e_end = parse_xsd_temporal(end_lit)
            # For endDate, take the end of the parsed span
            end_iso = e_end

        # If one side missing, still create interval with one bound
        # Find or create interval node
        existing_intervals = list(g.objects(m, CORE.hasTimeInterval))
        if existing_intervals:
            interval = existing_intervals[0]
            updated += 1
        else:
            # Make a deterministic URI if m is a URIRef, else a blank node
            if isinstance(m, URIRef):
                interval = URIRef(str(m) + args.interval_suffix)
            else:
                interval = BNode()
            g.add((m, CORE.hasTimeInterval, interval))
            g.add((interval, RDF.type, CORE.TimeInterval))
            created += 1

        # Write bounds
        if start_iso:
            g.set((interval, CORE.startTime, Literal(start_iso, datatype=XSD.dateTime)))
        if end_iso:
            g.set((interval, CORE.endTime, Literal(end_iso, datatype=XSD.dateTime)))

        # Optionally remove original schema date predicates
        if not args.keep_schema_dates:
            for x in sdates:
                g.remove((m, SCHEMA.startDate, x))
            for x in edates:
                g.remove((m, SCHEMA.endDate, x))

    g.serialize(destination=args.out_path, format="turtle")

    print(f"Memberships found: {len(memberships)}")
    print(f"Intervals created: {created}, updated: {updated}")
    print(f"Wrote: {args.out_path}")

if __name__ == "__main__":
    main()
