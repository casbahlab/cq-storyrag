import json
import csv
import requests
from rdflib import Graph, Namespace, URIRef, RDFS

# ==== CONFIG ====
INPUT_KG = "data/liveaid_extended.ttl"
OUTPUT_KG = "data/enriched_kg.ttl"
LOG_FILE = "logs/artist_enrichment_log.csv"
CACHE_FILE = "logs/verified_artist_mappings.json"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")
SCHEMA = Namespace("https://schema.org/")

# ---- Load cache if exists ----
try:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        verified_cache = json.load(f)
except FileNotFoundError:
    verified_cache = {}

# ---- Extract Artists from KG ----
def extract_artists(graph):
    """Extract all artists from KG with their labels"""
    artists = []
    for s in graph.subjects(predicate=None, object=MM.MusicArtist):
        label = graph.value(s, RDFS.label)
        if label:
            artists.append({"uri": str(s), "label": str(label)})
    return artists

# ---- Lookup on Wikidata ----
def wikidata_lookup(label):
    """Robust lookup for artists (bands or humans)"""
    query = f"""
    SELECT ?item ?itemLabel ?mbid ?wikipedia
    WHERE {{
      ?item rdfs:label "{label}"@en.

      VALUES ?acceptableClass {{ wd:Q215380 wd:Q5 }}
      ?item wdt:P31/wdt:P279* ?acceptableClass.

      MINUS {{ ?item wdt:P31 wd:Q482994 }}      # album
      MINUS {{ ?item wdt:P31 wd:Q7366 }}        # song
      MINUS {{ ?item wdt:P31 wd:Q4167836 }}     # Wikimedia category

      OPTIONAL {{ ?item wdt:P434 ?mbid. }}
      OPTIONAL {{
        ?wikipedia schema:about ?item ;
                   schema:isPartOf <https://en.wikipedia.org/> .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 3
    """

    headers = {"User-Agent": "KG-Enrichment-Bot/1.0"}
    r = requests.get(WIKIDATA_ENDPOINT, params={'query': query, 'format': 'json'}, headers=headers)
    if r.status_code == 200:
        results = r.json()["results"]["bindings"]
        if results:
            results.sort(key=lambda x: "mbid" not in x)  # Prefer MusicBrainz
            return results

    # ---- Fallback: Wikidata Search API ----
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": label,
        "limit": 3
    }
    r2 = requests.get(search_url, params=params, headers=headers)
    if r2.status_code == 200:
        data = r2.json()
        candidates = []
        for entry in data.get("search", []):
            candidates.append({
                "item": {"value": f"http://www.wikidata.org/entity/{entry['id']}"},
                "wikipedia": {"value": entry.get("url", "")}
            })
        return candidates

    return []

# ---- Interactive Approval ----
def approve_candidate(artist_label, candidates):
    """Prompt user to approve a Wikidata mapping"""
    print(f"\n=== {artist_label} ===")
    for idx, cand in enumerate(candidates, start=1):
        qid = cand.get("item", {}).get("value")
        mbid = cand.get("mbid", {}).get("value", "")
        wiki = cand.get("wikipedia", {}).get("value", "")
        print(f"[{idx}] {qid} | MBID: {mbid} | Wikipedia: {wiki}")

    choice = input("Select [1..n], [s]kip, or type QID manually: ").strip()

    if choice.lower() == "s":
        return None
    elif choice.upper().startswith("Q"):
        return f"http://www.wikidata.org/entity/{choice.upper()}"
    elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
        return candidates[int(choice)-1]["item"]["value"]
    else:
        print("Invalid input, skipping.")
        return None

import requests

def fetch_wikidata_metadata(wikidata_uri):
    """Fetch MBID and Wikipedia for a Wikidata entity URI."""
    qid = wikidata_uri.split("/")[-1]
    query = f"""
    SELECT ?mbid ?wikipedia
    WHERE {{
      OPTIONAL {{ wd:{qid} wdt:P434 ?mbid. }}
      OPTIONAL {{
        ?wikipedia schema:about wd:{qid};
                   schema:isPartOf <https://en.wikipedia.org/> .
      }}
    }}
    LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "KG-Enrichment-Bot/1.0"}
    r = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)

    if r.status_code == 200:
        results = r.json()["results"]["bindings"]
        if results:
            mbid = results[0].get("mbid", {}).get("value", "")
            wikipedia = results[0].get("wikipedia", {}).get("value", "")
            return mbid, wikipedia
    return "", ""


def enrich_artist(graph, artist, candidate):
    """
    Append enrichment triples for an artist using Wikidata + MusicBrainz + Wikipedia.
    Automatically fetches MBID and Wikipedia if missing.
    """
    subj = URIRef(artist["uri"])

    # Normalize candidate
    if isinstance(candidate, str):
        wikidata_uri = candidate
        mbid, wikipedia = fetch_wikidata_metadata(wikidata_uri)
    else:
        wikidata_uri = candidate.get("wikidata") or candidate.get("item", {}).get("value")
        mbid = candidate.get("musicbrainz") or candidate.get("mbid", {}).get("value", "")
        wikipedia = candidate.get("wikipedia") or candidate.get("wikipedia", {}).get("value", "")

        # Self-heal if partial
        if wikidata_uri and (not mbid or not wikipedia):
            mbid, wikipedia = fetch_wikidata_metadata(wikidata_uri)

    # Add Wikidata sameAs
    if wikidata_uri:
        graph.add((subj, SCHEMA.sameAs, URIRef(wikidata_uri)))

    # Add MusicBrainz sameAs if MBID exists
    if mbid:
        mbid_uri = mbid if mbid.startswith("http") else f"https://musicbrainz.org/artist/{mbid}"
        graph.add((subj, SCHEMA.sameAs, URIRef(mbid_uri)))

    # Add Wikipedia as subjectOf
    if wikipedia:
        graph.add((subj, SCHEMA.subjectOf, URIRef(wikipedia)))

    return {
        "EntityURI": artist["uri"],
        "Label": artist["label"],
        "Wikidata": wikidata_uri or "",
        "MusicBrainz": mbid,
        "Wikipedia": wikipedia
    }


# ---- Main Pipeline ----
def main():
    g = Graph()
    g.parse(INPUT_KG, format="ttl")
    artists = extract_artists(g)
    print(f"Found {len(artists)} artists to enrich.")

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["EntityURI","Label","Wikidata"])
        writer.writeheader()

        for artist in artists:
            label = artist["label"]

            # Skip if already verified
            if label in verified_cache:
                print(f"Using cached mapping for {label}: {verified_cache[label]}")
                enrich_artist(g, artist, verified_cache[label])
                writer.writerow({
                    "EntityURI": artist["uri"],
                    "Label": label,
                    "Wikidata": verified_cache[label]
                })
                continue

            # Lookup candidates
            candidates = wikidata_lookup(label)
            if not candidates:
                print(f"⚠ No candidates found for {label}, skipping.")
                continue

            # Interactive approval
            selected = approve_candidate(label, candidates)
            if selected:
                enrich_artist(g, artist, selected)
                verified_cache[label] = selected
                writer.writerow({
                    "EntityURI": artist["uri"],
                    "Label": label,
                    "Wikidata": selected
                })
                print(f"Enriched with {selected}")
            else:
                print(f"  ⏭ Skipped {label}")

    # Save cache + KG
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(verified_cache, f, indent=2)

    g.serialize(OUTPUT_KG, format="ttl")
    print(f"\nEnriched TTL saved to {OUTPUT_KG}")
    print(f"Log saved to {LOG_FILE}")
    print(f"Cache updated at {CACHE_FILE}")

if __name__ == "__main__":
    main()
