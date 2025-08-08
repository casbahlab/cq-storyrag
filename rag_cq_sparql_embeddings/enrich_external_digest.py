import json
import requests
from urllib.parse import urlparse
import re

INPUT_FILE = "embeddings/cq_results_for_embeddings.json"
OUTPUT_FILE = "embeddings/cq_results_with_enhanced_digests.json"

# ---------------- WIKIPEDIA ----------------
def fetch_wikipedia_digest(url: str) -> str:
    try:
        title = url.split("/wiki/")[-1]
        summary_api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        parse_api = f"https://en.wikipedia.org/w/api.php?action=parse&page={title}&prop=sections&format=json"

        summary = requests.get(summary_api, timeout=10).json().get("extract", "")
        sections_resp = requests.get(parse_api, timeout=10).json()
        sections = [s["line"] for s in sections_resp.get("parse", {}).get("sections", []) if s["toclevel"] == 1]
        top_sections = ", ".join(sections[:5]) if sections else ""
        return f"Wikipedia: {summary} {'Main sections: '+top_sections if top_sections else ''}".strip()
    except Exception as e:
        print(f"[WARN] Wikipedia digest failed for {url}: {e}")
        return ""

# ---------------- WIKIDATA ----------------
from functools import lru_cache

# Cache to avoid repeated HTTP calls
@lru_cache(maxsize=1000)
def fetch_wikidata_label(qid: str) -> str:
    """Fetch English label for a QID or PID."""
    try:
        api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        data = requests.get(api_url, timeout=10).json()
        entity = data["entities"].get(qid, {})
        return entity.get("labels", {}).get("en", {}).get("value", qid)
    except:
        return qid

def fetch_wikidata_digest(url: str) -> str:
    try:
        entity_id = url.split("/")[-1]
        if not re.match(r'^Q\d+$', entity_id):
            return f"Wikidata (Label only): {entity_id.replace('_',' ')}"

        api_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        data = requests.get(api_url, timeout=10).json()
        entity = data["entities"].get(entity_id, {})

        label = entity.get("labels", {}).get("en", {}).get("value", entity_id)
        desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
        claims = entity.get("claims", {})

        readable_facts = []
        skip_props = {"P213", "P214", "P244", "P373"}  # Skip non-semantic identifiers

        for pid, claim_list in list(claims.items())[:5]:  # limit to top 5
            if pid in skip_props:
                continue

            prop_label = fetch_wikidata_label(pid)
            mainsnak = claim_list[0].get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value")

            # Handle QIDs
            if isinstance(value, dict) and "id" in value:
                value_label = fetch_wikidata_label(value["id"])
                readable_facts.append(f"{prop_label}: {value_label}")
            elif isinstance(value, (str, int, float)):
                readable_facts.append(f"{prop_label}: {value}")

        facts_text = "; ".join(readable_facts) if readable_facts else "No key facts found"
        return f"Wikidata: {label} – {desc}. Key facts: {facts_text}".strip()
    except Exception as e:
        print(f"[WARN] Wikidata digest failed for {url}: {e}")
        return f"Wikidata (Label only): {url.split('/')[-1]}"


# ---------------- MUSICBRAINZ ----------------
def fetch_musicbrainz_digest(url: str) -> str:
    try:
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) < 2:
            return ""
        entity_type, entity_id = parts[-2], parts[-1]

        if entity_type not in ["artist", "recording", "release", "work", "event"]:
            return f"MusicBrainz: Unsupported type in {url}"

        api_url = f"https://musicbrainz.org/ws/2/{entity_type}/{entity_id}?fmt=json"
        resp = requests.get(api_url, headers={"User-Agent": "WembleyRewind/1.0"}, timeout=10)
        if resp.status_code != 200:
            return f"MusicBrainz: No response for {entity_type}"

        data = resp.json()

        if entity_type == "artist":
            name = data.get("name", "")
            typ = data.get("type", "")
            area = data.get("area", {}).get("name", "")
            life = data.get("life-span", {})
            lifespan = f"{life.get('begin','')}–{life.get('end','present')}"
            tags = [t.get("name","") for t in data.get("tags", [])[:3]]
            return f"MusicBrainz Artist: {name} ({typ}) from {area}, active {lifespan}, tags: {', '.join(tags)}"

        if entity_type == "work":
            title = data.get("title", "")
            lang = data.get("language", "")
            type_ = data.get("type", "")
            return f"MusicBrainz Work: {title} ({type_}, language: {lang})"

        if entity_type == "recording":
            title = data.get("title", "")
            artist_names = [a.get("name") for a in data.get("artist-credit", [])]
            duration = data.get("length")
            dur_str = f"{int(duration/1000)}s" if duration else "Unknown length"
            return f"MusicBrainz Recording: {title} by {', '.join(artist_names)} ({dur_str})"

        if entity_type == "release":
            title = data.get("title", "")
            date = data.get("date", "")
            country = data.get("country", "")
            return f"MusicBrainz Release: {title} ({date}, {country})"

        if entity_type == "event":
            name = data.get("name", "")
            time = data.get("time", "")
            return f"MusicBrainz Event: {name} {time}".strip()

        return f"MusicBrainz {entity_type.capitalize()}: {data.get('title') or data.get('name','')}"
    except Exception as e:
        print(f"[WARN] MusicBrainz digest failed for {url}: {e}")
        return ""

# ---------------- MAIN PIPELINE ----------------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = json.load(f)

for entry in entries:
    links = [l for l in entry.get("ExternalLinks", []) if l.startswith("http") and "example.org" not in l]
    entry["ExternalLinks"] = list(dict.fromkeys(links))  # dedupe

    digests = []
    for link in entry["ExternalLinks"]:
        if "wikipedia.org" in link:
            digests.append(fetch_wikipedia_digest(link))
        elif "wikidata.org" in link:
            digests.append(fetch_wikidata_digest(link))
        elif "musicbrainz.org" in link:
            digests.append(fetch_musicbrainz_digest(link))

    external_digest = " | ".join(filter(None, digests)) if digests else "None"
    entry["ExternalDigest"] = external_digest
    entry["EmbeddingInput"] += f"\nExternal Digest:\n{external_digest}"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2, ensure_ascii=False)

print(f"[INFO] Enhanced digests generated for {len(entries)} entries → {OUTPUT_FILE}")
