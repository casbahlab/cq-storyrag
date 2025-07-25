# enrichment/wikipedia.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Tuple

def extract_wikipedia_title(url: str) -> str:
    """Extract the article title from a Wikipedia URL."""
    parsed = urlparse(url)
    return parsed.path.split('/')[-1]

def enrich_from_wikipedia(subject: str, url: str) -> List[Tuple[str, str, str]]:
    """
    Fetch and extract structured content from a Wikipedia page.
    Use section headings and infobox labels as predicates.
    """
    title = extract_wikipedia_title(url)
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{title}"
    response = requests.get(api_url)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    triples = []

    # Extract infobox entries (predicate = label, object = value)
    infobox = soup.find("table", class_="infobox")
    if infobox:
        for row in infobox.find_all("tr"):
            header = row.find("th")
            value = row.find("td")
            if header and value:
                predicate = header.get_text(strip=True)
                obj = value.get_text(strip=True)
                triples.append((subject, predicate, obj))

    # Extract basic section content (predicate = section heading)
    for heading in soup.find_all(["h2", "h3"]):
        section_title = heading.get_text(strip=True)
        section_content = []
        for sib in heading.find_next_siblings():
            if sib.name in ["h2", "h3"]:
                break
            if sib.name == "p":
                section_content.append(sib.get_text(strip=True))
        if section_content:
            predicate = section_title
            obj = " ".join(section_content)[:500] + "..."  # Truncate long sections
            triples.append((subject, predicate, obj))

    print(f"enrich_from_wikipedia: {triples}")

    return triples
