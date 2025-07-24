import requests

def enrich_from_musicbrainz(subject_uri: str, mb_url: str):
    enriched = []
    try:
        mbid = mb_url.split("/")[-1]
        url = f"https://musicbrainz.org/ws/2/recording/{mbid}?fmt=json"
        res = requests.get(url, headers={"User-Agent": "LiveAidKG/1.0"}).json()

        title = res.get("title")
        if title:
            enriched.append((subject_uri, "schema:name", title))

        for artist_credit in res.get("artist-credit", []):
            name = artist_credit.get("name")
            if name:
                enriched.append((subject_uri, "schema:byArtist", name))

    except Exception as e:
        print(f"[MusicBrainz Enrichment Error] {e}")

    return enriched
