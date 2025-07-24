import json
from collections import defaultdict

# === Input and output paths ===
input_file = "raw_files/triples_from_gpt.json"
ttl_output_file = "schema_class_attributes_full.ttl"

# === Load JSON ===
with open(input_file, "r") as f:
    data = json.load(f)

# === Build type → properties map ===
subject_types = defaultdict(set)
subject_properties = defaultdict(set)

for item in data:
    for triple in item.get("Triples", []):
        subj = triple.get("subject")
        pred = triple.get("predicate")
        obj = triple.get("object")

        if pred == "rdf:type":
            subject_types[subj].add(obj)
        elif subj and pred:
            subject_properties[subj].add(pred)

type_to_properties = defaultdict(set)
for subj, types in subject_types.items():
    props = subject_properties.get(subj, set())
    for t in types:
        type_to_properties[t].update(props)

# === TTL Header ===
ttl_lines = [
    "@prefix schema: <http://schema.org/> .",
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
]

# === Generate TTL body ===
for t, props in sorted(type_to_properties.items()):
    ttl_lines.append(f"{t} a rdfs:Class .\n")
    for p in sorted(props):
        if p != "rdf:type":
            ttl_lines.append(f"{p} a rdf:Property ;")
            ttl_lines.append(f"  rdfs:domain {t} .\n")

# === Save TTL file ===
with open(ttl_output_file, "w") as f:
    f.write("\n".join(ttl_lines))

print(f"TTL class and property declarations written to {ttl_output_file}")
