import json
import re
from SPARQLWrapper import SPARQLWrapper, JSON

# ---- CONFIG ----
ENDPOINT_URL = "http://localhost:7200/repositories/liveaid-kg"
RQ_FILE = "queries/cqs_queries.rq"
OUTPUT_FILE = "embeddings/cq_results.json"

def normalize_cq_id(cq_id_raw: str):
    """Normalize CQ_ID from comments like 'CQ-E3/CQ-L3 (Extended)' into a list ['CQ-E3','CQ-L3']"""
    cq_id_clean = re.sub(r"\s*\(.*?\)", "", cq_id_raw)
    cq_ids = [cid.strip().replace(" Extended", "") for cid in cq_id_clean.split("/") if cid.strip()]
    return cq_ids

def run_query(query: str):
    """Execute SPARQL query and return flat results as list of dicts"""
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    print(f"results: {results}")
    return [
        {k: v.get("value", "") for k, v in binding.items()}
        for binding in results["results"]["bindings"]
    ]

def load_queries_from_rq(rq_file: str):
    """Parse the .rq file and extract queries with CQ_ID and CQ_Text"""
    with open(rq_file, encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    queries = []
    for block in blocks:
        lines = block.splitlines()
        header_line = next((l for l in lines if l.startswith("#")), None)
        if header_line:
            cq_id = header_line.replace("#", "").split(":")[0].strip()
            cq_text = header_line.split(":", 1)[-1].strip()
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
        cq_id_raw = q["CQ_ID"]
        cq_ids = normalize_cq_id(cq_id_raw)

        print(f"Executing {cq_id_raw} ...")
        try:
            results = run_query(q["SPARQL_Query"])
            all_results.append({
                "CQ_ID": cq_id_raw,
                "CQ_List": cq_ids,
                "CQ_Text": q["CQ_Text"],
                "Results": results
            })
            print(f"  -> {len(results)} results")
        except Exception as e:
            print(f"  !! Error on {cq_id_raw}: {e}")
            all_results.append({
                "CQ_ID": cq_id_raw,
                "CQ_List": cq_ids,
                "CQ_Text": q["CQ_Text"],
                "Error": str(e),
                "Results": []
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll query results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
