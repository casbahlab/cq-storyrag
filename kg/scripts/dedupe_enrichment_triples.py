#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deduplicate enrichment triples across TTLs:
- normalizes https://schema.org -> http://schema.org
- collapses duplicate schema:identifier PropertyValues per (subject, propertyID, value)
- collapses duplicate schema:video VideoObjects per (subject, YouTubeID)
- removes exact duplicate triples automatically

Outputs a single cleaned TTL.

Py3.9-friendly. Requires: rdflib
"""

import argparse, re
from pathlib import Path
from typing import Dict, Tuple
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL, BNode, Literal
from rdflib.namespace import XSD

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

SCHEMA_HTTPS = "https://schema.org/"
SCHEMA_HTTP  = "http://schema.org/"

def normalize_schema(g: Graph) -> Graph:
    out = Graph()
    for pfx, ns in g.namespaces():
        out.bind(pfx, ns)
    for s,p,o in g:
        p2, o2 = p, o
        if isinstance(p, URIRef) and str(p).startswith(SCHEMA_HTTPS):
            p2 = URIRef(str(p).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        if isinstance(o, URIRef) and str(o).startswith(SCHEMA_HTTPS):
            o2 = URIRef(str(o).replace(SCHEMA_HTTPS, SCHEMA_HTTP, 1))
        out.add((s,p2,o2))
    out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)
    return out

def load_merged(paths):
    g = Graph()
    for p in paths:
        gp = Graph(); gp.parse(p, format="turtle")
        g += normalize_schema(gp)
    g.bind("schema", SCHEMA); g.bind("mm", MM); g.bind("owl", OWL); g.bind("rdfs", RDFS); g.bind("xsd", XSD)
    return g

def yt_id_from_node(g: Graph, v) -> str:
    """Extract a YouTube video id from schema:url/embedUrl values if present."""
    urls = []
    for u in g.objects(v, SCHEMA.url):       urls.append(str(u))
    for u in g.objects(v, SCHEMA.embedUrl):  urls.append(str(u))
    for u in urls:
        s = str(u)
        # patterns: full and short
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", s)
        if not m:
            m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", s)
        if m:
            return m.group(1)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input TTL files to dedupe")
    ap.add_argument("--out", default="kg/27_enrichment_dedup.ttl")
    args = ap.parse_args()

    g = load_merged(args.inputs)

    # ——— Pass 1: plan canonical nodes ———
    pv_canon: Dict[Tuple[URIRef,str,str], URIRef] = {}  # (subject, propID, value) -> canonical PV node
    pv_map: Dict[URIRef, URIRef] = {}                   # old PV -> canonical PV

    vid_canon: Dict[Tuple[URIRef,str], URIRef] = {}     # (subject, yt_id) -> canonical VideoObject
    vid_map: Dict[URIRef, URIRef] = {}                  # old vid -> canonical vid

    # Collect all property value nodes linked via schema:identifier
    for s, _, pv in g.triples((None, SCHEMA.identifier, None)):
        if not isinstance(pv, (URIRef, BNode)):
            continue
        if (pv, RDF.type, SCHEMA.PropertyValue) not in g:
            continue
        propid = next((str(o) for o in g.objects(pv, SCHEMA.propertyID)), None)
        val    = next((str(o) for o in g.objects(pv, SCHEMA.value)), None)
        if not propid or not val:
            continue
        key = (s, propid, val)
        # canonical PV IRI (stable)
        if key not in pv_canon:
            pv_canon[key] = URIRef(str(s) + f"_prop_{propid}_{abs(hash(val)) % 10**8}")
        pv_map[pv] = pv_canon[key]

    # Collect all video nodes linked via schema:video
    for s, _, v in g.triples((None, SCHEMA.video, None)):
        if not isinstance(v, (URIRef, BNode)):
            continue
        vid = yt_id_from_node(g, v)
        if not vid:
            # keep non-YouTube videos as-is
            continue
        key = (s, vid)
        if key not in vid_canon:
            vid_canon[key] = URIRef(str(s) + "_YT_" + vid)
        vid_map[v] = vid_canon[key]

    # ——— Pass 2: rewrite into a fresh graph ———
    out = Graph(); out.bind("schema", SCHEMA); out.bind("mm", MM); out.bind("owl", OWL); out.bind("rdfs", RDFS); out.bind("xsd", XSD)

    def remap(node):
        # collapse PVs and Videos to canonical nodes
        if isinstance(node, (URIRef, BNode)):
            node = pv_map.get(node, node)
            node = vid_map.get(node, node)
        return node

    # Rewrite all triples with mapping applied
    for s,p,o in g:
        s2 = remap(s); o2 = remap(o)
        out.add((s2, p, o2))

    # Ensure canonical VideoObjects have minimal consistent metadata
    for (subj, vid), vcanon in vid_canon.items():
        # Type
        out.add((vcanon, RDF.type, SCHEMA.VideoObject))
        # Ensure url/embedUrl present (derive from id if missing)
        has_url     = any(True for _ in out.triples((vcanon, SCHEMA.url, None)))
        has_embed   = any(True for _ in out.triples((vcanon, SCHEMA.embedUrl, None)))
        if not has_url:
            out.add((vcanon, SCHEMA.url, URIRef(f"https://www.youtube.com/watch?v={vid}")))
        if not has_embed:
            out.add((vcanon, SCHEMA.embedUrl, URIRef(f"https://www.youtube.com/watch?v={vid}")))
        # Provider
        if (vcanon, SCHEMA.provider, None) not in out:
            out.add((vcanon, SCHEMA.provider, URIRef("https://www.youtube.com/")))
        # Link from subject
        if (subj, SCHEMA.video, vcanon) not in out:
            out.add((subj, SCHEMA.video, vcanon))

    # Ensure canonical PropertyValues have the core triples
    for (subj, propid, val), pcanon in pv_canon.items():
        out.add((pcanon, RDF.type, SCHEMA.PropertyValue))
        out.add((pcanon, SCHEMA.propertyID, Literal(propid)))
        out.add((pcanon, SCHEMA.value, Literal(val)))
        if (subj, SCHEMA.identifier, pcanon) not in out:
            out.add((subj, SCHEMA.identifier, pcanon))

    # Write
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")

    # Stats
    collapsed_pv = len(set(pv_map.values())) - len(set(k for k in pv_map.keys() if pv_map[k] == k))
    collapsed_vid = len(set(vid_map.values())) - len(set(k for k in vid_map.keys() if vid_map[k] == k))
    print(f"[write] {args.out} (triples: {len(out)})")
    print(f"[stats] identifiers collapsed: {len(pv_map)} → {len(set(pv_map.values()))} | videos collapsed: {len(vid_map)} → {len(set(vid_map.values()))}")

if __name__ == "__main__":
    main()
