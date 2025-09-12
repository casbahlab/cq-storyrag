
#!/usr/bin/env python3
"""
fix_membership_bnodes.py

Replaces anonymous mm:MusicEnsembleMembership blank nodes with
deterministic IRIs and removes duplicate triples.

Deterministic IRI pattern:
  ex:Membership_{GroupLocal}_{MemberLocal}_{YYYYMMDD-or-YYYY-or-X}
If collision risk remains, a short hash suffix is added.

Works with files that use either "schema:" or "schema1:" for http://schema.org/.

Usage:
  python fix_membership_bnodes.py \
    --in kg/20_artists.ttl \
    --out kg/20_artists_memberships_skolem.ttl

  python fix_membership_bnodes.py \
    --in kg/20_artists.ttl kg/30_memberships.ttl \
    --out kg/merged_memberships_skolem.ttl
"""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Tuple

from rdflib import Graph, Namespace, URIRef, BNode, Literal
from rdflib.namespace import RDF

SCHEMA  = Namespace("http://schema.org/")
SCHEMA1 = Namespace("http://schema.org/")
MM      = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX      = Namespace("http://wembrewind.live/ex#")

DATE_PROPS = (
    SCHEMA.startDate,
    SCHEMA1.startDate,
)

GROUP_PROP = MM.involvesMusicEnsemble
MEMBER_PROP = MM.involvesMemberOfMusicEnsemble

def local(u: URIRef) -> str:
    s = str(u)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rstrip("/").rsplit("/", 1)[-1]

def canonical_date_str(g: Graph, n) -> str:
    # prefer schema:startDate; accept any literal; canonicalize to digits
    for p in DATE_PROPS:
        for _, _, o in g.triples((n, p, None)):
            if isinstance(o, Literal):
                # YYYY-MM-DD / YYYY-MM → keep digits only for stability
                ds = re.sub(r"[^0-9]", "", str(o))
                return ds or "X"
    return "X"

def triple_fingerprint(g: Graph, node) -> str:
    rows = []
    for s, p, o in g.triples((node, None, None)):
        rows.append(("S", str(p), str(o)))
    for s, p, o in g.triples((None, None, node)):
        rows.append(("O", str(s), str(p)))
    rows.sort()
    return "|".join(f"{a}|{b}|{c}" for a,b,c in rows)

def mint_membership_iri(g: Graph, n) -> URIRef:
    # Extract ensemble and member where possible
    ens = next((o for _, _, o in g.triples((n, GROUP_PROP, None)) if isinstance(o, URIRef)), None)
    mem = next((o for _, _, o in g.triples((n, MEMBER_PROP, None)) if isinstance(o, URIRef)), None)
    ens_l = local(ens) if ens else "Ensemble"
    mem_l = local(mem) if mem else "Member"
    dstr = canonical_date_str(g, n)
    base = f"http://wembrewind.live/ex#Membership_{ens_l}_{mem_l}_{dstr}"
    iri = URIRef(base)
    # If something else already has this IRI, append a short hash of the node fingerprint
    # (or if ens/mem missing and base becomes too generic).
    fp = triple_fingerprint(g, n)
    h = hashlib.sha1(fp.encode("utf-8")).hexdigest()[:8]
    return URIRef(f"{base}_{h}")

def dedupe_graph(g: Graph) -> Graph:
    nt = g.serialize(format="nt")
    lines = sorted(set(nt.splitlines()))
    out = Graph()
    out.parse(data="\n".join(lines), format="nt")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input TTL file(s)")
    ap.add_argument("--out", required=True, help="Output TTL file")
    args = ap.parse_args()

    g = Graph()
    g.bind("schema", SCHEMA)
    g.bind("schema1", SCHEMA1)
    g.bind("mm", MM)
    g.bind("ex", EX)

    for f in args.inputs:
        g.parse(f, format="turtle")

    g = dedupe_graph(g)

    # Map only BNodes of type mm:MusicEnsembleMembership
    mapping: Dict[BNode, URIRef] = {}
    for n in list(g.subjects(RDF.type, MM.MusicEnsembleMembership)):
        if isinstance(n, BNode):
            mapping[n] = mint_membership_iri(g, n)

    # Apply mapping to all triples
    out = Graph()
    out.bind("schema", SCHEMA); out.bind("schema1", SCHEMA1)
    out.bind("mm", MM); out.bind("ex", EX)

    for s, p, o in g.triples((None, None, None)):
        s2 = mapping.get(s, s)
        o2 = mapping.get(o, o)
        out.add((s2, p, o2))

    out = dedupe_graph(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.serialize(destination=args.out, format="turtle")
    print(f"[ok] Wrote memberships-skolemized TTL: {args.out}")
    print(f"[info] Replaced {len(mapping)} anonymous memberships with IRIs.")

if __name__ == "__main__":
    main()
