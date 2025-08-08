import rdflib
from rdflib import Graph
import json
import os
from collections import defaultdict

INPUT_TTL = "output/final_with_inferred.ttl"
OUTPUT_TTL = "output/final_grouped_triples_classified.ttl"
UNDEFINED_JSON = "output/undefined_entities.json"

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Load KG
g = Graph()
print("Loading KG...")
g.parse(INPUT_TTL, format="turtle")
print(f"✅ Parsed KG with {len(g)} triples")

# Extract subjects and objects
subjects = set(str(s) for s in g.subjects())
objects = set(str(o) for o in g.objects())

# Identify undefined entities (objects never used as subjects)
undefined = sorted([o for o in objects if o not in subjects and o.startswith("http://wembrewind.live/ex#")])

print(f"⚠ Found {len(undefined)} undefined entities")

# Gather context for each undefined entity
context_map = defaultdict(list)
for s, p, o in g.triples((None, None, None)):
    o_str = str(o)
    if o_str in undefined:
        context_map[o_str].append(f"{s} {p}")

# Load existing classification log if present
classified_entities = {}
if os.path.exists(UNDEFINED_JSON):
    with open(UNDEFINED_JSON, "r") as f:
        classified_entities = json.load(f)

# Interactive classification
for entity in undefined:
    if entity in classified_entities:
        continue

    print("\n============================================================")
    print(f"Undefined entity: {entity}")
    print("Used in context (sample up to 3):")
    for c in context_map[entity][:3]:
        print(f"  - {c}")

    choice = input("\nOptions: [k]eep as entity & type, [l]iteral, [s]kip: ").strip().lower()

    if choice == "l":
        classified_entities[entity] = {"action": "literal"}
    elif choice == "k":
        ent_type = input("Enter type (e.g., mm:LivePerformance, schema:Place): ").strip()
        classified_entities[entity] = {"action": "entity", "type": ent_type}
    else:
        classified_entities[entity] = {"action": "skip"}

# Save classification decisions
with open(UNDEFINED_JSON, "w") as f:
    json.dump(classified_entities, f, indent=2)
print(f"💾 Classification log saved to {UNDEFINED_JSON}")

# Build new TTL with stubs
stub_lines = []
for entity, info in classified_entities.items():
    if info["action"] == "entity":
        stub_lines.append(f"<{entity}> a {info['type']} ; schema:description \"Stub entity\" .")

with open(OUTPUT_TTL, "w", encoding="utf-8") as out:
    # Copy original KG
    with open(INPUT_TTL, "r", encoding="utf-8") as original:
        out.write(original.read())
        out.write("\n\n# === AUTO-GENERATED STUB ENTITIES ===\n")
        for line in stub_lines:
            out.write(line + "\n")

print(f"✅ New TTL with entity stubs: {OUTPUT_TTL}")
