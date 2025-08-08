import requests

def enrich_from_wikidata(subject_uri: str, wikidata_url: str):
    enriched = []
    try:
        qid = wikidata_url.split("/")[-1]
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        entity = data["entities"][qid]

        # Optional: Label and description
        label = entity.get("labels", {}).get("en", {}).get("value")
        description = entity.get("descriptions", {}).get("en", {}).get("value")
        if label:
            enriched.append((subject_uri, "label", label))
        if description:
            enriched.append((subject_uri, "description", description))

        claims = entity.get("claims", {})
        for prop_id, claim_list in claims.items():
            # Extract property label (e.g., "genre", "country of origin")
            prop_label = get_property_label(prop_id)

            for claim in claim_list:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                value = parse_datavalue(datavalue)

                if value and prop_label:
                    enriched.append((subject_uri, prop_label, value))

    except Exception as e:
        print(f"[Wikidata Enrichment Error] {e}")

    print(f"enrich_from_wikipedia: {enriched}")

    return enriched


def get_property_label(prop_id: str) -> str:
    """Fetch English label for the given Wikidata property ID."""
    try:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{prop_id}.json"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return data["entities"][prop_id]["labels"]["en"]["value"]
    except Exception:
        return prop_id  # fallback to raw ID if label fails


def parse_datavalue(datavalue: dict) -> str:
    """Extract a readable value from a Wikidata datavalue."""
    if not datavalue:
        return ""

    dtype = datavalue.get("type")
    value = datavalue.get("value")

    if dtype == "wikibase-entityid":
        return f"https://www.wikidata.org/wiki/{value.get('id')}"
    elif dtype == "string":
        return value
    elif dtype == "monolingualtext":
        return value.get("text")
    elif dtype == "time":
        return value.get("time")
    elif dtype == "quantity":
        return str(value.get("amount"))
    elif isinstance(value, str):
        return value
    else:
        return str(value)

