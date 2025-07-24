import requests

def enrich_from_wikidata(subject_uri: str, wikidata_url: str):
    enriched = []
    try:
        qid = wikidata_url.split("/")[-1]
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        res = requests.get(url).json()
        entity = res["entities"][qid]

        name = entity.get("labels", {}).get("en", {}).get("value")
        description = entity.get("descriptions", {}).get("en", {}).get("value")

        if name:
            enriched.append((subject_uri, "schema:name", name))
        if description:
            enriched.append((subject_uri, "schema:description", description))

    except Exception as e:
        print(f"[Wikidata Enrichment Error] {e}")

    return enriched
