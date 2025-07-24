import requests

def enrich_from_wikipedia(subject_uri: str, wiki_url: str):
    enriched = []
    try:
        title = wiki_url.split("/")[-1]
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        res = requests.get(api_url).json()

        extract = res.get("extract")
        if extract:
            enriched.append((subject_uri, "schema:description", extract))

    except Exception as e:
        print(f"[Wikipedia Enrichment Error] {e}")

    return enriched
