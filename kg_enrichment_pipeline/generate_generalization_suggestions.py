import rdflib
from rdflib import Graph, RDF
from collections import defaultdict
import os
import json

INPUT_FILE = "output/final_cleaned_kg.ttl"
OUTPUT_JSON = "output/generalization_suggestions.json"
OUTPUT_MD = "output/generalization_report.md"

os.makedirs("output", exist_ok=True)

# 1. Load KG
g = Graph()
g.parse(INPUT_FILE, format="turtle")
print(f"✅ Loaded KG with {len(g)} triples.")

# 2. Build pattern clusters
pattern_clusters = defaultdict(list)

for s in set(g.subjects()):
    # Ignore blank nodes
    if isinstance(s, rdflib.BNode):
        continue

    # Get types and properties
    types = sorted(str(o) for o in g.objects(s, RDF.type))
    props = sorted(set(str(p) for p in g.predicates(s, None) if p != RDF.type))

    if not types and not props:
        continue  # skip completely empty subjects

    pattern_key = (tuple(types), tuple(props))
    pattern_clusters[pattern_key].append(str(s))

# 3. Generate suggestions for clusters with ≥2 instances
suggestions = {}
report_lines = ["# Generalization Suggestions\n"]

for idx, ((types, props), instances) in enumerate(pattern_clusters.items(), start=1):
    if len(instances) < 2:
        continue  # skip singletons

    # Suggest class name based on first type or "CustomClass"
    base_name = types[0].split("/")[-1] if types else "CustomClass"
    suggested_class = f"ex:{base_name}Cluster{idx}"

    suggestions[suggested_class] = {
        "instances": instances,
        "types": types,
        "properties": props
    }

    # Markdown report
    report_lines.append(f"## {suggested_class}")
    report_lines.append(f"**Instances ({len(instances)}):** {', '.join(instances)}")
    report_lines.append(f"**Types:** {', '.join(types) or 'None'}")
    report_lines.append(f"**Properties:** {', '.join(props) or 'None'}\n")

# 4. Save outputs
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(suggestions, f, indent=2)

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"✅ Suggestions generated for {len(suggestions)} clusters.")
print(f"📄 JSON: {OUTPUT_JSON}")
print(f"📄 Markdown report: {OUTPUT_MD}")
