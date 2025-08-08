import rdflib
import pandas as pd
import os

# ==== CONFIG ====
KG_FILE = "output/liveaid_instances.ttl"        # Path to your KG
CQ_FILE = "data/WembleyRewindCQs_categories.csv"     # Path to CQ CSV
OUTPUT_REPORT = "output/cq_coverage_report.csv"
OUTPUT_BRIDGE_TTL = "output/cq_bridge_suggestions.ttl"

# Namespaces
EX = "http://wembrewind.live/ex#"
MM = "https://w3id.org/polifonia/ontology/music-meta/"
SCHEMA = "http://schema.org/"

os.makedirs("output", exist_ok=True)

# ==== STEP 1: Load KG ====
g = rdflib.Graph()
g.parse(KG_FILE, format="turtle")
print(f"✅ Loaded KG with {len(g)} triples")

# ==== STEP 2: Load CQs ====
cqs = pd.read_csv(CQ_FILE)
required_cols = {"CQ_ID", "Question", "Category"}
if not required_cols.issubset(set(cqs.columns)):
    raise ValueError(f"CQ CSV must include: {required_cols}")

report_data = []
bridging_triples = []

# ==== STEP 3: Generate & Execute SPARQL for each CQ ====
for idx, row in cqs.iterrows():
    cq_id = row["CQ_ID"]
    question = row["Question"]
    entity = row["Expected_Entity"]
    relation = row.get("Expected_Relation", "").strip()

    # Normalize entity to URI
    entity_uri = (
        entity if entity.startswith("http") else EX + entity
    )

    # Generate SPARQL
    if relation:
        # Query for a specific relation
        sparql = f"""
        SELECT ?o WHERE {{
            <{entity_uri}> <{relation}> ?o .
        }}
        """
    else:
        # Broad query: get all outgoing triples for that entity
        sparql = f"""
        SELECT ?p ?o WHERE {{
            <{entity_uri}> ?p ?o .
        }}
        """

    results = list(g.query(sparql))
    passed = len(results) > 0

    report_data.append({
        "CQ_ID": cq_id,
        "Question": question,
        "Entity": entity_uri,
        "Relation": relation or "(any)",
        "Result_Count": len(results),
        "Pass": "✅" if passed else "❌"
    })

    # Step 4: Suggest bridging triple if failed
    if not passed and relation:
        bridging_triples.append(
            f"<{entity_uri}> <{relation}> <{EX}MissingEntityFor_{cq_id}> ."
        )

# ==== STEP 5: Save Reports ====
pd.DataFrame(report_data).to_csv(OUTPUT_REPORT, index=False)
print(f"📄 Coverage report saved to: {OUTPUT_REPORT}")

with open(OUTPUT_BRIDGE_TTL, "w") as ttl:
    ttl.write("@prefix ex: <http://wembrewind.live/ex#> .\n")
    ttl.write("@prefix mm: <https://w3id.org/polifonia/ontology/music-meta/> .\n")
    ttl.write("@prefix schema: <http://schema.org/> .\n\n")
    for t in bridging_triples:
        ttl.write(t + "\n")

print(f"🔧 Bridging TTL suggestions saved to: {OUTPUT_BRIDGE_TTL}")
print(f"Total CQs: {len(cqs)}, Passed: {sum(r['Pass']=='✅' for r in report_data)}")
