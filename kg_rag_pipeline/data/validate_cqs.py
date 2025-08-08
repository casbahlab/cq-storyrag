import json
import re
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult

# ==== CONFIG ====
kg_file = "liveaid_instances.ttl"  # Your KG
#sparql_file = "cqs_queries.rq"     # SPARQL queries
sparql_file = "cqs_queries_template_filled_in.rq"
output_json = "cq_validation_report.json"

# ==== Load KG ====
g = Graph()
g.parse(kg_file, format="turtle")
print(f"Loaded KG with {len(g)} triples")

# ==== Load and split SPARQL queries ====
with open(sparql_file, "r") as f:
    content = f.read()

# Split queries by double newlines
queries = [q.strip() for q in content.split("\n\n") if q.strip()]

report = []

for idx, query in enumerate(queries, start=1):
    # Extract CQ ID from the first comment line if available
    lines = query.splitlines()
    cq_id = None
    for line in lines:
        if line.strip().startswith("#"):
            # extract something like CQ-L1 or CQ-E12
            match = re.search(r"(CQ-[A-Za-z]*\d+)", line)
            if match:
                cq_id = match.group(1)
                break
    if not cq_id:
        cq_id = f"CQ-{idx}"  # fallback

    entry = {"cq_id": cq_id, "success": False, "rows": 0, "error": None, "results": []}

    try:
        res: SPARQLResult = g.query(query)
        rows = list(res)
        entry["success"] = True
        entry["rows"] = len(rows)

        # Store each result as a dict {var: value}
        for row in rows:
            row_dict = {str(var): str(row[var]) for var in row.labels}
            entry["results"].append(row_dict)

    except Exception as e:
        entry["error"] = str(e)

    report.append(entry)

# ==== Save report ====
with open(output_json, "w") as f:
    json.dump(report, f, indent=2)

print(f"Validation complete. Report saved to {output_json}")
