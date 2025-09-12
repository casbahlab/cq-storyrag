# slice_external_links.py (enhanced with optional descriptions)
import argparse
from collections import defaultdict
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, Literal, BNode

SCHEMA = Namespace("http://schema.org/")
EXBASE = "http://wembrewind.live/ex#"

KEEP_PROPS = {
    SCHEMA.sameAs, OWL.sameAs,
    SCHEMA.image, SCHEMA.logo,
    SCHEMA.video, SCHEMA.url, SCHEMA.embedUrl, SCHEMA.provider,
    SCHEMA.identifier, SCHEMA.propertyID, SCHEMA.value, SCHEMA.name,
    # schema:description is handled conditionally below
}

DEFAULT_DESC_TYPES = {
    SCHEMA.Person, SCHEMA.MusicGroup, SCHEMA.MusicComposition, SCHEMA.Event, SCHEMA.Place
}

def is_external(u: URIRef) -> bool:
    s = str(u)
    return s.startswith("http") and not s.startswith(EXBASE)

def normalize_schema(g: Graph) -> Graph:
    out = Graph()
    for pfx, ns in g.namespaces(): out.bind(pfx, ns)
    for s,p,o in g:
        if isinstance(p, URIRef) and str(p).startswith("https://schema.org/"):
            p = URIRef(str(p).replace("https://", "http://", 1))
        if isinstance(o, URIRef) and str(o).startswith("https://schema.org/"):
            o = URIRef(str(o).replace("https://", "http://", 1))
        out.add((s,p,o))
    out.bind("schema", SCHEMA); out.bind("owl", OWL); out.bind("rdfs", RDFS)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default="kg/27_external_links_only.ttl")
    ap.add_argument("--removed-out", default="kg/27_removed_nonlink_content.ttl")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--keep-descriptions", action="store_true",
                    help="Keep schema:description on selected types")
    ap.add_argument("--desc-types", default="Person,MusicGroup,MusicComposition,Event,Place",
                    help="Comma-separated schema:* local names")
    ap.add_argument("--desc-lang", default="en", help="Language tag to keep (e.g., en)")
    ap.add_argument("--desc-max", type=int, default=280, help="Max description length")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g = Graph(); g.parse(args.infile, format="turtle"); g = normalize_schema(g)

    # Build allowed description types from CLI
    allowed_desc_types = set()
    for name in [s.strip() for s in args.desc_types.split(",") if s.strip()]:
        allowed_desc_types.add(getattr(SCHEMA, name, None))
    allowed_desc_types = {t for t in allowed_desc_types if t is not None} or DEFAULT_DESC_TYPES

    kept = Graph();    [kept.bind(p,n) for p,n in g.namespaces()]
    removed = Graph(); [removed.bind(p,n) for p,n in g.namespaces()]

    keep_nodes = set()

    # Discover nodes to keep for external links
    for s,p,o in g:
        if p in (SCHEMA.sameAs, OWL.sameAs) and isinstance(o, URIRef) and is_external(o):
            keep_nodes.add(s)
        if p in (SCHEMA.image, SCHEMA.logo) and isinstance(o, URIRef) and is_external(o):
            keep_nodes.add(s)
        if p == SCHEMA.video and isinstance(o, (URIRef, BNode)):
            keep_nodes.add(s); keep_nodes.add(o)
        if p == SCHEMA.identifier and isinstance(o, (URIRef, BNode)):
            keep_nodes.add(s); keep_nodes.add(o)

    def subject_has_allowed_type(s) -> bool:
        for t in allowed_desc_types:
            if (s, RDF.type, t) in g:
                return True
        return False

    def keep_description_triple(s, p, o) -> bool:
        if not args.keep_descriptions: return False
        if p != SCHEMA.description:    return False
        if not subject_has_allowed_type(s): return False
        if isinstance(o, Literal):
            if args.desc_lang and (o.language or "").lower() != args.desc_lang.lower():
                return False
            if args.desc_max and len(str(o)) > args.desc_max:
                return False
            return True
        return False

    def should_keep(s, p, o) -> bool:
        if p in KEEP_PROPS and ((s in keep_nodes) or (o in keep_nodes)):
            return True
        if keep_description_triple(s, p, o):
            return True
        return False

    kept_count = removed_count = 0
    for s,p,o in g:
        if should_keep(s,p,o):
            kept.add((s,p,o)); kept_count += 1
        else:
            removed.add((s,p,o)); removed_count += 1

    # Minimal typing for kept nodes
    for s,_,v in kept.triples((None, SCHEMA.video, None)):
        if (v, RDF.type, SCHEMA.VideoObject) not in kept:
            kept.add((v, RDF.type, SCHEMA.VideoObject))
    for s,_,pv in kept.triples((None, SCHEMA.identifier, None)):
        if (pv, RDF.type, SCHEMA.PropertyValue) not in kept:
            kept.add((pv, RDF.type, SCHEMA.PropertyValue))

    if args.dry_run:
        print(f"[dry-run] kept_triples={kept_count} | removed_triples={removed_count}")
        return

    kept.serialize(args.out, format="turtle")
    removed.serialize(args.removed_out, format="turtle")
    print(f"[write] kept  → {args.out} (triples: {len(kept)})")
    print(f"[write] removed → {args.removed_out} (triples: {len(removed)})")

    if args.report_out:
        import csv
        from collections import defaultdict
        removed_by_s = defaultdict(lambda: {"count":0, "preds": set()})
        kept_by_s    = defaultdict(int)
        for s,_,_ in kept:    kept_by_s[s] += 1
        for s,p,_ in removed:
            removed_by_s[s]["count"] += 1
            removed_by_s[s]["preds"].add(str(p))
        with open(args.report_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject","kept_triples","removed_triples","removed_predicates"])
            all_subjects = set(removed_by_s.keys()) | set(kept_by_s.keys())
            for s in sorted(all_subjects, key=lambda x: str(x)):
                w.writerow([str(s), kept_by_s.get(s,0), removed_by_s.get(s,{"count":0})["count"],
                            "; ".join(sorted(removed_by_s.get(s,{"preds":set()})["preds"]))])
        print(f"[write] report → {args.report_out}")
