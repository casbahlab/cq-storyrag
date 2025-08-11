#!/usr/bin/env python3
import rdflib
import argparse
import os

import rdflib

SCHEMA = rdflib.Namespace("http://schema.org/")
MM     = rdflib.Namespace("https://w3id.org/polifonia/ontology/music-meta/")
EX     = rdflib.Namespace("http://wembrewind.live/ex#")

def _norm_term(t):
    # Unify schema.org to http:// (prevents rdflib inventing schema1:)
    if isinstance(t, rdflib.URIRef) and str(t).startswith("https://schema.org/"):
        return rdflib.URIRef(str(t).replace("https://schema.org/", "http://schema.org/"))
    return t

def filter_and_write(g, types, output_path, keep_one_hop=True):
    subgraph = rdflib.Graph()

    # Hard-reset potential conflicting bindings, then bind preferred ones
    nm = subgraph.namespace_manager
    nm.bind("schema", None, replace=True)
    nm.bind("schema1", None, replace=True)
    subgraph.bind("ex", EX)
    subgraph.bind("schema", SCHEMA)
    subgraph.bind("mm", MM)

    subjects = {s for s in g.subjects(rdflib.RDF.type, None)
                if any((s, rdflib.RDF.type, t) in g for t in types)}

    for s in subjects:
        for p, o in g.predicate_objects(s):
            subgraph.add((_norm_term(s), _norm_term(p), _norm_term(o)))
        if keep_one_hop:
            # include both rdfs:label and schema:name as handy labels
            for o in g.objects(s, rdflib.RDFS.label):
                subgraph.add((_norm_term(s), rdflib.RDFS.label, o))
            for o in g.objects(s, SCHEMA.name):
                subgraph.add((_norm_term(s), SCHEMA.name, o))

    subgraph.serialize(output_path, format="turtle")
    print(f"[split] Wrote {output_path} ({len(subgraph)} triples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split master TTL by entity types.")
    parser.add_argument("--kg", required=True, help="Path to master TTL")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    g = rdflib.Graph()
    g.parse(args.kg, format="turtle")

    os.makedirs(args.out, exist_ok=True)



    solo_types = [rdflib.URIRef("http://schema.org/Person"), rdflib.URIRef("https://w3id.org/polifonia/ontology/music-meta/Musician")]
    group_types = [rdflib.URIRef("http://schema.org/MusicGroup")]
    performance_types = [rdflib.URIRef("https://w3id.org/polifonia/ontology/music-meta/LivePerformance"), rdflib.URIRef("http://wembrewind.live/ex#SongPerformance")]
    song_types = [rdflib.URIRef("http://schema.org/MusicComposition"), rdflib.URIRef("http://schema.org/MusicRecording")]

    filter_and_write(g, solo_types, os.path.join(args.out, "solo_artists.ttl"))
    filter_and_write(g, group_types, os.path.join(args.out, "music_groups.ttl"))
    filter_and_write(g, performance_types, os.path.join(args.out, "performances.ttl"))
    filter_and_write(g, song_types, os.path.join(args.out, "songs.ttl"))
