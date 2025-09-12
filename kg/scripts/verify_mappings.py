#!/usr/bin/env python3
import os, sys, csv, glob
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal

SCHEMA = Namespace("http://schema.org/")
SCHEMA_HTTPS = Namespace("https://schema.org/")
MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
MM_HTTP = Namespace("http://w3id.org/polifonia/ontology/music-meta/")

def lname(u):
    s = str(u)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def get_label(g, node):
    for p in (SCHEMA.name, SCHEMA_HTTPS.name, RDFS.label):
        val = g.value(node, p)
        if isinstance(val, Literal):
            return str(val)
    s = str(node)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]

def parse_any(g, path):
    # try turtle first then let rdflib guess
    try:
        g.parse(path, format="turtle")
    except Exception:
        g.parse(path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_mappings.py <20_artists.ttl> <40_setlists_songs.ttl> [output.csv]")
        sys.exit(1)

    f1, f2 = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else "kg_membership_performance_verification.csv"

    g = Graph()
    parse_any(g, f1)
    parse_any(g, f2)

    # Discover classes actually used
    classes_used = set(o for _, _, o in g.triples((None, RDF.type, None)))

    musician_classes = set()
    group_classes = set()
    performance_classes = set()

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

    musicians = set()
    for cls in musician_classes:
        for s in g.subjects(RDF.type, cls):
            musicians.add(s)

    groups = set()
    for cls in group_classes:
        for s in g.subjects(RDF.type, cls):
            groups.add(s)

    performances = set()
    for cls in performance_classes:
        for s in g.subjects(RDF.type, cls):
            performances.add(s)

    # --- Membership inference (mm:MusicEnsembleMembership) ---
    membership_nodes = set(g.subjects(RDF.type, MM.MusicEnsembleMembership)) | \
                       set(g.subjects(RDF.type, MM_HTTP.MusicEnsembleMembership))

    member_pred_candidates = set()
    ensemble_pred_candidates = set()

    for m in membership_nodes:
        for p, o in g.predicate_objects(m):
            if o in musicians:
                member_pred_candidates.add(p)
            if o in groups:
                ensemble_pred_candidates.add(p)

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

    musician_to_groups = {m: set() for m in musicians}
    group_to_musicians = {gr: set() for gr in groups}

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
                musician_to_groups.setdefault(mu, set()).add(gr)
                group_to_musicians.setdefault(gr, set()).add(mu)

    # --- Performer predicate inference (performance → performer) ---
    performer_pred_candidates = set()
    for cand in (SCHEMA.performer, SCHEMA_HTTPS.performer, MM.hasPerformer, MM_HTTP.hasPerformer):
        if any(True for _ in g.triples((None, cand, None))):
            performer_pred_candidates.add(cand)

    if not performer_pred_candidates:
        for perf in performances:
            for p, _ in g.predicate_objects(perf):
                if "perform" in lname(p).lower():
                    performer_pred_candidates.add(p)

    entity_to_perfs = {e: set() for e in musicians | groups}
    for perf in performances:
        for p in performer_pred_candidates:
            for ent in g.objects(perf, p):
                if ent in entity_to_perfs:
                    entity_to_perfs[ent].add(perf)

    # --- Output CSV ---
    fieldnames = [
        "entity_uri","entity_name","entity_type",
        "has_membership","membership_groups","membership_members",
        "has_performance_link","performance_uris"
    ]
    rows = []

    # Musicians
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

    # Groups
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

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Console summary
    print("=== Summary ===")
    print(f"Musicians: {len(musicians)}")
    print(f"MusicGroups: {len(groups)}")
    print(f"Performances: {len(performances)}")
    print(f"Membership nodes: {len(membership_nodes)}")
    print("Member predicate candidates:", ", ".join(str(p) for p in member_pred_candidates) or "(none)")
    print("Ensemble predicate candidates:", ", ".join(str(p) for p in ensemble_pred_candidates) or "(none)")
    print("Performer predicate candidates:", ", ".join(str(p) for p in performer_pred_candidates) or "(none)")
    print(f"\nWrote report → {out_path}")

if __name__ == "__main__":
    main()
