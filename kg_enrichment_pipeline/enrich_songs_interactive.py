import json
import csv
import requests
from rdflib import Graph, Namespace, URIRef, RDFS

# ==== CONFIG ====
INPUT_KG = "data/enriched_kg.ttl"
OUTPUT_KG = "data/enriched_kg_songs.ttl"
LOG_FILE = "logs/song_enrichment_log.csv"
CACHE_FILE = "logs/verified_song_mappings.json"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

SCHEMA = Namespace("https://schema.org/")

# ---- Load cache if exists ----
try:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        verified_cache = json.load(f)
except FileNotFoundError:
    verified_cache = {}

from rdflib.namespace import RDF

MM = Namespace("https://w3id.org/polifonia/ontology/music-meta/")

def extract_songs(graph):
    """Extract all songs from the KG using mm:MusicalWork or schema:MusicRecording."""
    songs = []
    seen = set()

    for s in graph.subjects(RDF.type, MM.MusicalWork):
        label = graph.value(s, RDFS.label)
        if label and s not in seen:
            songs.append({"uri": str(s), "label": str(label)})
            seen.add(s)

    # Fallback: also look for schema:MusicRecording
    for s in graph.subjects(RDF.type, SCHEMA.MusicRecording):
        label = graph.value(s, RDFS.label)
        if label and s not in seen:
            songs.append({"uri": str(s), "label": str(label)})
            seen.add(s)

    return songs

# ---- Wikidata Lookup ----
def wikidata_lookup(label):
    """Lookup candidate songs on Wikidata."""
    query = f"""
    SELECT ?item ?itemLabel ?mbid ?wikipedia
    WHERE {{
      ?item rdfs:label "{label}"@en.
      VALUES ?acceptableClass {{ wd:Q7366 }}   # Song

      ?item wdt:P31/wdt:P279* ?acceptableClass.

      OPTIONAL {{ ?item wdt:P435 ?mbid. }}  # MusicBrainz Work ID
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

# ---- Self-Healing Metadata Fetch ----
def fetch_wikidata_metadata(wikidata_uri):
    """Fetch MBID and Wikipedia for a Wikidata entity URI."""
    qid = wikidata_uri.split("/")[-1]
    query = f"""
    SELECT ?mbid ?wikipedia
    WHERE {{
      OPTIONAL {{ wd:{qid} wdt:P435 ?mbid. }}
      OPTIONAL {{
        ?wikipedia schema:about wd:{qid};
                   schema:isPartOf <https://en.wikipedia.org/> .
      }}
    }}
    LIMIT 1
    """
    headers = {"User-Agent": "KG-Enrichment-Bot/1.0"}
    r = requests.get(WIKIDATA_ENDPOINT, params={'query': query, 'format': 'json'}, headers=headers)
    if r.status_code == 200:
        results = r.json()["results"]["bindings"]
        if results:
            mbid = results[0].get("mbid", {}).get("value", "")
            wikipedia = results[0].get("wikipedia", {}).get("value", "")
            return mbid, wikipedia
    return "", ""

# ---- Interactive Approval ----
def approve_candidate(song_label, candidates):
    """Prompt user to approve a Wikidata mapping"""
    print(f"\n=== {song_label} ===")
    for idx, cand in enumerate(candidates, start=1):
        qid = cand.get("item", {}).get("value")
        mbid = cand.get("mbid", {}).get("value", "")
        wiki = cand.get("wikipedia", {}).get("value", "")
        print(f"[{idx}] {qid} | MBID: {mbid} | Wikipedia: {wiki}")

    choice = input("Select [1..n], [s]kip, or type QID manually: ").strip()

    if choice.lower() == "s":
        return None
    elif choice.upper().startswith("Q"):
        return {"wikidata": f"http://www.wikidata.org/entity/{choice.upper()}"}
    elif choice.isdigit() and 1 <= int(choice) <= len(candidates):
        cand = candidates[int(choice)-1]
        return {
            "wikidata": cand.get("item", {}).get("value", ""),
            "musicbrainz": cand.get("mbid", {}).get("value", ""),
            "wikipedia": cand.get("wikipedia", {}).get("value", "")
        }
    else:
        print("sInvalid input, skipping.")
        return None

# ---- Enrich Song ----
def enrich_song(graph, song, cache_entry):
    """Add enrichment triples for a song."""
    subj = URIRef(song["uri"])

    # Normalize and self-heal
    if isinstance(cache_entry, str):
        wikidata_uri = cache_entry
        mbid, wikipedia = fetch_wikidata_metadata(wikidata_uri)
        cache_entry = {
            "wikidata": wikidata_uri,
            "musicbrainz": mbid,
            "wikipedia": wikipedia
        }
    else:
        wikidata_uri = cache_entry.get("wikidata", "")
        mbid = cache_entry.get("musicbrainz", "")
        wikipedia = cache_entry.get("wikipedia", "")

        if wikidata_uri and (not mbid or not wikipedia):
            mbid, wikipedia = fetch_wikidata_metadata(wikidata_uri)
            cache_entry["musicbrainz"] = mbid
            cache_entry["wikipedia"] = wikipedia

    # Add triples
    if cache_entry["wikidata"]:
        graph.add((subj, SCHEMA.sameAs, URIRef(cache_entry["wikidata"])))
    if cache_entry["musicbrainz"]:
        mbid_uri = cache_entry["musicbrainz"]
        if not mbid_uri.startswith("http"):
            mbid_uri = f"https://musicbrainz.org/work/{mbid_uri}"
        graph.add((subj, SCHEMA.sameAs, URIRef(mbid_uri)))
    if cache_entry["wikipedia"]:
        graph.add((subj, SCHEMA.subjectOf, URIRef(cache_entry["wikipedia"])))

    return cache_entry

# ---- Main Pipeline ----
def main():
    g = Graph()
    g.parse(INPUT_KG, format="ttl")
    songs = extract_songs(g)
    print(f"Found {len(songs)} songs to enrich.")

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["SongLabel","EntityURI","Wikidata","MusicBrainz","Wikipedia"])
        writer.writeheader()

        for song in songs:
            label = song["label"]

            # Skip if already cached
            if label in verified_cache:
                print(f"Using cached mapping for {label}")
                enriched_entry = enrich_song(g, song, verified_cache[label])
                verified_cache[label] = enriched_entry
                writer.writerow({
                    "SongLabel": label,
                    "EntityURI": song["uri"],
                    "Wikidata": enriched_entry.get("wikidata", ""),
                    "MusicBrainz": enriched_entry.get("musicbrainz", ""),
                    "Wikipedia": enriched_entry.get("wikipedia", "")
                })
                continue

            # Lookup candidates
            candidates = wikidata_lookup(label)
            if not candidates:
                print(f"No candidates found for {label}, skipping.")
                continue

            # Interactive approval
            selected = approve_candidate(label, candidates)
            if selected:
                enriched_entry = enrich_song(g, song, selected)
                verified_cache[label] = enriched_entry
                writer.writerow({
                    "SongLabel": label,
                    "EntityURI": song["uri"],
                    "Wikidata": enriched_entry.get("wikidata", ""),
                    "MusicBrainz": enriched_entry.get("musicbrainz", ""),
                    "Wikipedia": enriched_entry.get("wikipedia", "")
                })
                print(f"  Enriched {label} with {enriched_entry}")
            else:
                print(f"  Skipped {label}")

    # Save cache + KG
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(verified_cache, f, indent=2)

    g.serialize(OUTPUT_KG, format="ttl")
    print(f"\nEnriched TTL saved to {OUTPUT_KG}")
    print(f"Log saved to {LOG_FILE}")
    print(f"Cache updated at {CACHE_FILE}")

if __name__ == "__main__":
    main()
