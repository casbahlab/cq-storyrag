import json
from SPARQLWrapper import SPARQLWrapper, JSON

# ---- CONFIG ----
ENDPOINT_URL = "http://localhost:7200/repositories/liveaid-kg"
RQ_FILE = "queries/cqs_queries.rq"
OUTPUT_FILE = "embeddings/cq_results.json"


def run_query(query):
    """Executes SPARQL and returns structured list of dicts."""
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [
        {k: v.get("value", "") for k, v in binding.items()}
        for binding in results["results"]["bindings"]
    ]


def load_queries_from_rq(rq_file):
    """Splits the .rq file into blocks of (# comment + SPARQL query)."""
    with open(rq_file, encoding="utf-8") as f:
        content = f.read()

    # Split by double line breaks
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    queries = []
    current_cq = None
    for block in blocks:
        lines = block.splitlines()
        # Extract first line starting with "#"
        header_line = next((l for l in lines if l.startswith("#")), None)
        if header_line:
            cq_id = header_line.replace("#", "").split(":")[0].strip()
            cq_text = header_line.split(":", 1)[-1].strip()
            # Remove comment lines to get the actual query
            sparql_query = "\n".join(l for l in lines if not l.startswith("#"))
            queries.append({
                "CQ_ID": cq_id,
                "CQ_Text": cq_text,
                "SPARQL_Query": sparql_query
            })
    return queries


def main():
    queries = load_queries_from_rq(RQ_FILE)
    all_results = []

    for q in queries:
        cq_id = q["CQ_ID"]
        print(f"Executing {cq_id} ...")
        try:
            results = run_query(q["SPARQL_Query"])
            all_results.append({
                "CQ_ID": cq_id,
                "CQ_Text": q["CQ_Text"],
                "Results": results
            })
            print(f"  -> {len(results)} results")
        except Exception as e:
            print(f"  !! Error on {cq_id}: {e}")
            all_results.append({
                "CQ_ID": cq_id,
                "CQ_Text": q["CQ_Text"],
                "Error": str(e),
                "Results": []
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll query results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
