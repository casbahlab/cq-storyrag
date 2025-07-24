from wikidata import enrich_from_wikidata
from musicbrainz import enrich_from_musicbrainz
from wikipedia import enrich_from_wikipedia

def enrich_graph_with_external_links(triples):
    enriched_triples = []

    for s, p, o in triples:
        if "wikidata.org" in o:
            enriched_triples += enrich_from_wikidata(s, o)
        elif "musicbrainz.org" in o:
            enriched_triples += enrich_from_musicbrainz(s, o)
        elif "wikipedia.org" in o:
            enriched_triples += enrich_from_wikipedia(s, o)

    return enriched_triples
