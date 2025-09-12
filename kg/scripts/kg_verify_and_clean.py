#!/usr/bin/env python3
"""
kg_verify_and_clean.py

Verify and (optionally) clean artist/group entries in a KG.

It does two things:
1) VERIFY: builds a CSV that shows, for each Musician and MusicGroup:
   - whether it has a membership (via mm:MusicEnsembleMembership)
   - whether it has a direct performance link to a LivePerformance
2) CLEAN (optional): writes a cleaned copy of the artists TTL where any
   Musician or MusicGroup that has NEITHER a membership NOR a performance
   link (considering all loaded files) is removed.

USAGE EXAMPLES

# 1) Verify only: produce CSV report
python kg_verify_and_clean.py \
  --artists /path/to/20_artists.ttl \
  --extra /path/to/40_setlists_songs.ttl \
  --verify-out /path/to/kg_membership_performance_verification.csv

# 2) Verify + Clean: also write cleaned artists TTL
python kg_verify_and_clean.py \
  --artists /path/to/20_artists.ttl \
  --extra /path/to/40_setlists_songs.ttl \
  --verify-out /path/to/kg_membership_performance_verification.csv \
  --clean-out /path/to/20_artists.cleaned.ttl

# 3) If you don’t have the setlists file yet, omit --extra
python kg_verify_and_clean.py \
  --artists /path/to/20_artists.ttl \
  --verify-out /path/to/report.csv \
  --clean-out /path/to/20_artists.cleaned.ttl
"""

import argparse
import csv
from collections import defaultdict
from typing import Iterable, Set, Dict

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal

SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
MM_HTTP = Namespace("http://w3id.org/polifonia/ontology/music-meta/")

def lname(u) -> str:
    s = str(u)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def get_label(g: Graph, node: URIRef) -> str:
    for p in (SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label):
        val = g.value(node, p)
        if isinstance(val, Literal):
            return str(val)
    s = str(node)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def parse_any(g: Graph, path: str):
    # Try turtle first, then let rdflib guess
    try:
        g.parse(path, format="turtle")
    except Exception:
        g.parse(path)

def discover_classes(g: Graph):
    classes_used = set(o for _, _, o in g.triples((None, RDF.type, None)))
    musician_classes, group_classes, performance_classes = set(), set(), set()

    for c in classes_used:
        ln = lname(c).lower()
        if "musician" in ln or "musicartist" in ln:
            musician_classes.add(c)
        if "musicgroup" in ln or (ln.endswith("group") and "music" in ln):
            group_classes.add(c)
        if "performance" in ln:
            performance_classes.add(c)

    # add known URIs if present
    for c in (MM.Musician, MM_HTTP.Musician):
        if c in classes_used: musician_classes.add(c)
    for c in (SCHEMA.MusicGroup, SCHEMA_HTTPS.MusicGroup):
        if c in classes_used: group_classes.add(c)
    for c in (MM.LivePerformance, MM_HTTP.LivePerformance):
        if c in classes_used: performance_classes.add(c)

    return musician_classes, group_classes, performance_classes

def collect_instances(g: Graph, classes: Iterable[URIRef]) -> Set[URIRef]:
    out = set()
    for cls in classes:
        for s in g.subjects(RDF.type, cls):
            out.add(s)
    return out

def infer_memberships(g: Graph,
                      musicians: Set[URIRef],
                      groups: Set[URIRef]):
    membership_nodes = set(g.subjects(RDF.type, MM.MusicEnsembleMembership)) | \
                       set(g.subjects(RDF.type, MM_HTTP.MusicEnsembleMembership))

    member_pred_candidates = set()
    ensemble_pred_candidates = set()

    # From actual edges
    for m in membership_nodes:
        for p, o in g.predicate_objects(m):
            if o in musicians:
                member_pred_candidates.add(p)
            if o in groups:
                ensemble_pred_candidates.add(p)

    # Heuristics if not discovered
    if not member_pred_candidates:
        for m in membership_nodes:
            for p, _ in g.predicate_objects(m):
                if "member" in lname(p).lower():
                    member_pred_candidates.add(p)

    if not ensemble_pred_candidates:
        for m in membership_nodes:
            for p, _ in g.predicate_objects(m):
                l = lname(p).lower()
                if "ensemble" in l or "group" in l:
                    ensemble_pred_candidates.add(p)

    musician_to_groups: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    group_to_musicians: Dict[URIRef, Set[URIRef]] = defaultdict(set)

    for m in membership_nodes:
        linked_musicians = set()
        linked_groups = set()
        for p in member_pred_candidates:
            for o in g.objects(m, p):
                if o in musicians:
                    linked_musicians.add(o)
        for p in ensemble_pred_candidates:
            for o in g.objects(m, p):
                if o in groups:
                    linked_groups.add(o)
        for mu in linked_musicians:
            for gr in linked_groups:
                musician_to_groups[mu].add(gr)
                group_to_musicians[gr].add(mu)

    return membership_nodes, member_pred_candidates, ensemble_pred_candidates, musician_to_groups, group_to_musicians

def infer_performer_links(g: Graph,
                          performances: Set[URIRef],
                          entities: Set[URIRef]):
    performer_pred_candidates = set()
    # Known IRIs
    for cand in (SCHEMA.performer, SCHEMA_HTTPS.performer, MM.hasPerformer, MM_HTTP.hasPerformer):
        if any(True for _ in g.triples((None, cand, None))):
            performer_pred_candidates.add(cand)

    # Heuristic
    if not performer_pred_candidates:
        for perf in performances:
            for p, _ in g.predicate_objects(perf):
                if "perform" in lname(p).lower():
                    performer_pred_candidates.add(p)

    entity_to_perfs: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for perf in performances:
        for p in performer_pred_candidates:
            for ent in g.objects(perf, p):
                if ent in entities:
                    entity_to_perfs[ent].add(perf)

    return performer_pred_candidates, entity_to_perfs

def write_report(csv_path: str,
                 g: Graph,
                 musicians: Set[URIRef],
                 groups: Set[URIRef],
                 musician_to_groups: Dict[URIRef, Set[URIRef]],
                 group_to_musicians: Dict[URIRef, Set[URIRef]],
                 entity_to_perfs: Dict[URIRef, Set[URIRef]]):
    fieldnames = [
        "entity_uri","entity_name","entity_type",
        "has_membership","membership_groups","membership_members",
        "has_performance_link","performance_uris"
    ]
    rows = []

    for mu in sorted(musicians, key=lambda x: get_label(g, x).lower()):
        rows.append({
            "entity_uri": str(mu),
            "entity_name": get_label(g, mu),
            "entity_type": "Musician",
            "has_membership": "YES" if musician_to_groups.get(mu) else "NO",
            "membership_groups": "; ".join(sorted(get_label(g, gr) for gr in musician_to_groups.get(mu, []))),
            "membership_members": "",
            "has_performance_link": "YES" if entity_to_perfs.get(mu) else "NO",
            "performance_uris": ";".join(sorted(str(p) for p in entity_to_perfs.get(mu, set())))
        })

    for gr in sorted(groups, key=lambda x: get_label(g, x).lower()):
        rows.append({
            "entity_uri": str(gr),
            "entity_name": get_label(g, gr),
            "entity_type": "MusicGroup",
            "has_membership": "YES" if group_to_musicians.get(gr) else "NO",
            "membership_groups": "",
            "membership_members": "; ".join(sorted(get_label(g, mu) for mu in group_to_musicians.get(gr, []))),
            "has_performance_link": "YES" if entity_to_perfs.get(gr) else "NO",
            "performance_uris": ";".join(sorted(str(p) for p in entity_to_perfs.get(gr, set())))
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def clean_artists(artists_path: str,
                  out_path: str,
                  g_all: Graph,
                  g_artists: Graph,
                  musicians: Set[URIRef],
                  groups: Set[URIRef],
                  musician_to_groups: Dict[URIRef, Set[URIRef]],
                  group_to_musicians: Dict[URIRef, Set[URIRef]],
                  entity_to_perfs: Dict[URIRef, Set[URIRef]]) -> int:
    # Collect URIs present in artists file
    entities_in_artists = set()
    for s in g_artists.all_nodes():
        if isinstance(s, URIRef):
            entities_in_artists.add(s)

    def should_drop(e: URIRef) -> bool:
        is_musician = e in musicians
        is_group = e in groups
        if not (is_musician or is_group):
            return False  # only drop musicians or groups
        has_membership = bool(musician_to_groups.get(e) or group_to_musicians.get(e))
        has_perf = bool(entity_to_perfs.get(e))
        return (not has_membership) and (not has_perf)

    candidates = (musicians | groups) & entities_in_artists
    to_drop = {e for e in candidates if should_drop(e)}

    removed = 0
    for e in to_drop:
        for t in list(g_artists.triples((e, None, None))):
            g_artists.remove(t); removed += 1
        for t in list(g_artists.triples((None, None, e))):
            g_artists.remove(t); removed += 1

    g_artists.serialize(destination=out_path, format="turtle")
    return len(to_drop)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", required=True, help="Path to 20_artists.ttl (base file to clean)")
    ap.add_argument("--extra", action="append", default=[], help="Additional TTL/RDF files to consider (e.g., 40_setlists_songs.ttl). Can be repeated.")
    ap.add_argument("--verify-out", required=True, help="Path to write verification CSV")
    ap.add_argument("--clean-out", help="If provided, writes a cleaned copy of --artists to this path")
    args = ap.parse_args()

    # Combined graph for verification across all files
    g_all = Graph()
    parse_any(g_all, args.artists)
    for extra in args.extra:
        parse_any(g_all, extra)

    # Separate graph for the artists file (only this will be modified if cleaning)
    g_artists = Graph()
    parse_any(g_artists, args.artists)

    musician_classes, group_classes, performance_classes = discover_classes(g_all)
    musicians = collect_instances(g_all, musician_classes)
    groups = collect_instances(g_all, group_classes)
    performances = collect_instances(g_all, performance_classes)

    membership_nodes, member_pred_candidates, ensemble_pred_candidates, musician_to_groups, group_to_musicians = \
        infer_memberships(g_all, musicians, groups)

    performer_pred_candidates, entity_to_perfs = \
        infer_performer_links(g_all, performances, musicians | groups)

    # 1) Write verification report
    write_report(args.verify_out, g_all, musicians, groups,
                 musician_to_groups, group_to_musicians, entity_to_perfs)

    print("=== Summary ===")
    print(f"Musicians: {len(musicians)}")
    print(f"MusicGroups: {len(groups)}")
    print(f"Performances: {len(performances)}")
    print(f"Membership nodes: {len(membership_nodes)}")
    print("Member predicate candidates:", ", ".join(str(p) for p in sorted(member_pred_candidates, key=str)) or "(none)")
    print("Ensemble predicate candidates:", ", ".join(str(p) for p in sorted(ensemble_pred_candidates, key=str)) or "(none)")
    print("Performer predicate candidates:", ", ".join(str(p) for p in sorted(performer_pred_candidates, key=str)) or "(none)")
    print(f"Verification CSV → {args.verify_out}")

    # 2) Optionally clean the artists file
    if args.clean_out:
        removed_entities = clean_artists(args.artists, args.clean_out,
                                         g_all, g_artists,
                                         musicians, groups,
                                         musician_to_groups, group_to_musicians,
                                         entity_to_perfs)
        print(f"Cleaned TTL → {args.clean_out}")
        print(f"Entities removed (no membership & no performance): {removed_entities}")

if __name__ == "__main__":
    main()
