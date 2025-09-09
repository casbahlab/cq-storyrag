#!/usr/bin/env python3
# extract_schema_url.py
# Collect values mapped via schema:url from TTL/RDF files.
#
# What it does:
# - Parses files/directories with rdflib
# - Finds triples ?s schema:url ?o (supports http:// and https:// bases)
# - Accepts IRI objects and URL-looking literal objects (incl. xsd:anyURI)
# - Dedupes and writes a plain text list; optional CSV with subject+object
#
# Usage:
#   python extract_schema_url.py --data ./kg --mask "*.ttl" \
#     --out-urls schema_url_values.txt --out-csv schema_url_context.csv
#
# Filters:
#   --only-iri / --only-literal         (default: both)
#   --exclude-domain example.com,foo.org  (repeat or comma-separate; subdomains included)
#   --exclude-prefix https://my/site     (repeat or comma-separate)
#
import argparse
import re
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Optional, Iterable

from rdflib import Graph, URIRef, Literal, XSD

SCHEMA_URL_HTTP  = URIRef("http://schema.org/url")
SCHEMA_URL_HTTPS = URIRef("https://schema.org/url")
URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>'\")]+")

def iter_files(paths: List[Path], mask: Optional[str]) -> Iterable[Path]:
    for p in paths:
        if p.is_file():
            yield p
        else:
            patt = mask or "*"
            for f in p.rglob(patt):
                if f.is_file():
                    yield f

def parse_multi(values: Optional[List[str]]) -> List[str]:
    out = []
    if not values:
        return out
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    return out

def domain_match(host: str, doms: List[str]) -> bool:
    if not host: return False
    h = host.lstrip(".").lower()
    for d in doms:
        d = d.lstrip(".").lower()
        if h == d or h.endswith("." + d):
            return True
    return False

def hostname(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def should_keep(url: str, exclude_prefixes: List[str], exclude_domains: List[str]) -> bool:
    for p in exclude_prefixes:
        if url.startswith(p):
            return False
    if exclude_domains and domain_match(hostname(url), exclude_domains):
        return False
    return True

def url_from_literal(lit: Literal) -> Optional[str]:
    if lit.datatype == XSD.anyURI:
        return str(lit)
    m = URL_RE.search(str(lit))
    return m.group(0) if m else None

def collect_schema_urls(paths: List[Path], mask: Optional[str], only_iri: bool, only_literal: bool,
                        exclude_prefixes: List[str], exclude_domains: List[str]) -> List[tuple]:
    rows = []  # (subject, url_string)
    for file in iter_files(paths, mask):
        g = Graph()
        parsed = False
        for fmt in ["turtle","xml","n3","nt","trig","json-ld"]:
            try:
                g.parse(file.as_posix(), format=fmt)
                parsed = True
                break
            except Exception:
                continue
        if not parsed:
            continue

        for s, p, o in g:
            if p != SCHEMA_URL_HTTP and p != SCHEMA_URL_HTTPS:
                continue
            if isinstance(o, URIRef):
                if only_literal: 
                    continue
                so = str(o)
                if so.startswith(("http://","https://")) and should_keep(so, exclude_prefixes, exclude_domains):
                    rows.append((str(s), so))
            elif isinstance(o, Literal):
                if only_iri:
                    continue
                so = url_from_literal(o)
                if so and should_keep(so, exclude_prefixes, exclude_domains):
                    rows.append((str(s), so))
    return rows

def write_outputs(rows: List[tuple], out_urls: Path, out_csv: Optional[Path]):
    # Dedup by URL string, keep first occurrence
    seen = set()
    urls = []
    for s, u in rows:
        if u not in seen:
            seen.add(u)
            urls.append(u)
    out_urls.write_text("\n".join(urls) + ("\n" if urls else ""), encoding="utf-8")
    if out_csv:
        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject","schema_url_value"])
            for s, u in rows:
                w.writerow([s, u])

def main():
    ap = argparse.ArgumentParser(description="Extract values of schema:url from TTL/RDF files")
    ap.add_argument("--data", nargs="+", required=True, help="TTL/RDF files or directories")
    ap.add_argument("--mask", default="*.ttl", help="Glob when scanning directories")
    ap.add_argument("--out-urls", required=True, help="Write deduped URL list here")
    ap.add_argument("--out-csv", default=None, help="Also write (subject, value) CSV for validation")
    ap.add_argument("--only-iri", action="store_true", help="Only keep IRI objects")
    ap.add_argument("--only-literal", action="store_true", help="Only keep literal objects that look like URLs")
    ap.add_argument("--exclude-prefix", action="append", default=None, help="Exclude URLs with these prefix(es) (repeat or comma-separate)")
    ap.add_argument("--exclude-domain", action="append", default=None, help="Exclude URLs with these domain(s) (repeat or comma-separate)")
    args = ap.parse_args()

    if args.only_iri and args.only_literal:
        ap.error("Cannot use --only-iri and --only-literal together")

    paths = [Path(p) for p in args.data]
    exclude_prefixes = parse_multi(args.exclude_prefix)
    exclude_domains  = parse_multi(args.exclude_domain)

    rows = collect_schema_urls(paths, args.mask, args.only_iri, args.only_literal, exclude_prefixes, exclude_domains)
    write_outputs(rows, Path(args.out_urls), Path(args.out_csv) if args.out_csv else None)
    print(f"[extract] schema:url values: {len(rows)} (deduped -> {len(set(u for _,u in rows))})")
    print(f"[write] URLs -> {args.out_urls}")
    if args.out_csv:
        print(f"[write] CSV  -> {args.out_csv}")

if __name__ == "__main__":
    main()
