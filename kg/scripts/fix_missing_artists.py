#!/usr/bin/env python3
import argparse, re
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD

SCHEMA = Namespace("http://schema.org/")
MM     = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = Namespace("http://wembrewind.live/ex#")

def load_ttl(p: Path) -> Graph:
    g = Graph()
    g.parse(p, format="turtle")
    g.bind("ex", EX); g.bind("schema", SCHEMA); g.bind("mm", MM)
    g.bind("rdfs", RDFS); g.bind("owl", OWL); g.bind("xsd", XSD)
    return g

def guess_label(g: Graph, node: URIRef) -> str:
    for o in g.objects(node, RDFS.label):  # prefer rdfs:label if present
        return str(o)
    for o in g.objects(node, SCHEMA.name):
        return str(o)
    s = str(node)
    return s.split("#",1)[-1] if "#" in s else s.rsplit("/",1)[-1]

def ensure_artist(artists_g: Graph, node: URIRef, label: str, is_person: bool):
    """Mint minimal artist/group node, with BOTH rdfs:label and schema:name."""
    if (node, None, None) in artists_g:
        return False
    if is_person:
        for t in (SCHEMA.Person, SCHEMA.PropertyValue, MM.Musician):
            artists_g.add((node, RDF.type, t))
    else:
        for t in (SCHEMA.MusicGroup, SCHEMA.PropertyValue, MM.MusicArtist):
            artists_g.add((node, RDF.type, t))
    literal = Literal(label)
    artists_g.add((node, RDFS.label, literal))
    artists_g.add((node, SCHEMA.name, literal))
    return True

def ensure_schema_name_for_all_labels(artists_g: Graph) -> int:
    """
    For every subject with rdfs:label but no schema:name, add schema:name (same literal).
    Returns how many schema:name triples were added.
    """
    added = 0
    subjects = set(s for s, _, _ in artists_g.triples((None, RDFS.label, None)))
    for s in subjects:
        if (s, SCHEMA.name, None) in artists_g:
            continue
        # pick one label (if multiple, use the first encountered)
        lab = next((o for o in artists_g.objects(s, RDFS.label)), None)
        if lab is not None:
            artists_g.add((s, SCHEMA.name, lab))
            added += 1
    return added

def main():
    ap = argparse.ArgumentParser(description="Ensure all artists referenced in band-membership exist in 20_artists.ttl, and mirror rdfs:label to schema:name.")
    ap.add_argument("--artists", default="kg/20_artists.ttl")
    ap.add_argument("--members", default="kg/22_band_members.ttl")
    ap.add_argument("--out",     default=None, help="output path (default: overwrite --artists after writing a .bak)")
    args = ap.parse_args()

    artists_p = Path(args.artists)
    members_p = Path(args.members)
    out_p     = Path(args.out) if args.out else artists_p

    # Load artists and keep a pre-change backup if overwriting
    original_artists_text = artists_p.read_text(encoding="utf-8")

    artists_g = load_ttl(artists_p)
    members_g = load_ttl(members_p)

    # collect referenced members and ensembles
    members   = set(members_g.objects(None, MM.involvesMemberOfMusicEnsemble))
    ensembles = set(members_g.objects(None, MM.involvesMusicEnsemble))

    # also handle direct schema:member / schema:memberOf if present
    members  |= set(members_g.objects(None, SCHEMA.member))
    ensembles|= set(members_g.objects(None, SCHEMA.memberOf))

    # URIs only
    members    = {u for u in members if isinstance(u, URIRef)}
    ensembles  = {u for u in ensembles if isinstance(u, URIRef)}

    added_people = added_groups = 0

    for u in sorted(members):
        if (u, None, None) not in artists_g:
            lab = guess_label(members_g, u)
            if ensure_artist(artists_g, u, lab, is_person=True):
                added_people += 1

    for u in sorted(ensembles):
        if (u, None, None) not in artists_g:
            lab = guess_label(members_g, u)
            if ensure_artist(artists_g, u, lab, is_person=False):
                added_groups += 1

    # Global pass: add schema:name wherever rdfs:label exists but name is missing
    mirrored = ensure_schema_name_for_all_labels(artists_g)

    # backup & write
    if out_p == artists_p and not args.out:
        bak = artists_p.with_suffix(".bak.ttl")
        bak.write_text(original_artists_text, encoding="utf-8")
        print(f"[backup] wrote {bak}")

    artists_g.serialize(destination=str(out_p), format="turtle")
    print(f"[done] added people={added_people}, groups={added_groups}, mirrored schema:name for {mirrored} nodes → {out_p}")

if __name__ == "__main__":
    main()
