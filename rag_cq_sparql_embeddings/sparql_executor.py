from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT_URL = "http://localhost:7200/repositories/liveaid"

def run_query(query: str):
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    bindings = results.get("results", {}).get("bindings", [])
    structured = [{k: v.get("value") for k,v in row.items()} for row in bindings]
    return structured
