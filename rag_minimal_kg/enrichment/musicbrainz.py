import requests
from urllib.parse import urlparse
from rdflib import URIRef, Literal

def enrich_from_musicbrainz(subject: str, url: str):
    enriched = []

    parsed = urlparse(url)
    entity_id = parsed.path.strip("/").split("/")[-1]

    if "/recording/" in parsed.path:
        api_url = f"https://musicbrainz.org/ws/2/recording/{entity_id}?fmt=json&inc=artists+releases"
        response = requests.get(api_url, headers={"User-Agent": "WembleyRewind/1.0"})
        if response.ok:
            data = response.json()
            title = data.get("title")
            if title:
                enriched.append((subject, "schema:name", Literal(title)))
            for artist in data.get("artist-credit", []):
                name = artist.get("artist", {}).get("name")
                if name:
                    enriched.append((subject, "schema:byArtist", Literal(name)))
        else:
            print(f"recording response {response}")

    elif "/release/" in parsed.path:
        api_url = f"https://musicbrainz.org/ws/2/release/{entity_id}?fmt=json&inc=recordings+artist-credits"
        response = requests.get(api_url, headers={"User-Agent": "WembleyRewind/1.0"})

        if response.ok:
            data = response.json()
            title = data.get("title")
            if title:
                enriched.append((subject, "schema:name", Literal(title)))
            # Add each track in the release
            media = data.get("media", [])
            for disc in media:
                for track in disc.get("tracks", []):
                    track_title = track.get("title")
                    position = track.get("position")
                    if track_title:
                        enriched.append((subject, "schema:track", Literal(f"{position}. {track_title}")))
            # Add artist
            for artist in data.get("artist-credit", []):
                name = artist.get("artist", {}).get("name")
                if name:
                    enriched.append((subject, "schema:byArtist", Literal(name)))
        else:
            print(f"release response {response}")

    print(f"enrich_from_musicbrainz: {enriched}")

    return enriched
