import json
from collections import defaultdict
from pathlib import Path

# Load triples from JSON (your CQ-format input file)
INPUT_FILE = "triples_from_gpt.json"  # Replace with your actual path

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# Collect all triples across CQs
predicate_map = defaultdict(lambda: {
    "subjects": set(),
    "objects": set(),
    "example_triples": []
})

# Process each CQ
for entry in data:
    for triple in entry.get("Triples", []):
        subj = triple["subject"]
        pred = triple["predicate"]
        obj = triple["object"]

        predicate_map[pred]["subjects"].add(subj)
        predicate_map[pred]["objects"].add(obj)

        # Save a couple of example triples
        if len(predicate_map[pred]["example_triples"]) < 3:
            predicate_map[pred]["example_triples"].append(triple)

# Convert sets to lists for JSON serialization
ontology_template = {}
for predicate, info in predicate_map.items():
    ontology_template[predicate] = {
        "subjects": sorted(info["subjects"]),
        "objects": sorted(info["objects"]),
        "example_triples": info["example_triples"]
    }

# Save the template
output_file = "kg_predicate_summary.json"
with open(output_file, "w") as f:
    json.dump(ontology_template, f, indent=2)

print(f"Ontology structure saved to {output_file}")
