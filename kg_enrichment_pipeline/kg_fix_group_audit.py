import rdflib
from rdflib import Graph
import re
import os
import json
from collections import defaultdict

INPUT_FILE = "output/deduplicated_triples.ttl"
OUTPUT_TTL = "output/final_grouped_triples.ttl"
AUTO_FIX_LOG = "output/auto_fixed_triples.json"

# Ensure output folder exists
os.makedirs("output", exist_ok=True)

# Define prefixes
PREFIXES = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix schema: <http://schema.org/> .
@prefix mm: <https://w3id.org/polifonia/ontology/music-meta/> .
@prefix ex: <http://wembrewind.live/ex#> .
"""

fix_log = []

# Load raw TTL lines
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    original = line.strip()
    if not original or original.startswith("@prefix"):
        continue

    # Fix accidental double dots
    line = re.sub(r'\.\s*\.', '.', original)

    # Fix semicolon immediately after dot
    line = re.sub(r'\.\s*;', ';', line)

    # Quote unquoted literals that are not URIs or numbers
    tokens = line.split()
    if len(tokens) >= 3:
        subj, pred, obj = tokens[0], tokens[1], " ".join(tokens[2:])
        # Detect literals without quotes or prefix/number
        if not (obj.startswith("<") or ":" in obj or obj.startswith('"')):
            obj = f"\"{obj}\""
        line = f"{subj} {pred} {obj} ."

    cleaned_lines.append(line.rstrip(" .;"))

# Group triples by subject
grouped = defaultdict(list)
for line in cleaned_lines:
    parts = line.split(" ", 2)
    if len(parts) >= 3:
        subj, pred, obj = parts
        grouped[subj].append(f"{pred} {obj.rstrip('.')}")

# Write grouped TTL
with open(OUTPUT_TTL, "w", encoding="utf-8") as out:
    out.write(PREFIXES)
    for subj, triples in grouped.items():
        out.write(f"{subj}\n")
        for i, triple in enumerate(triples):
            end = " ;" if i < len(triples)-1 else " ."
            out.write(f"    {triple}{end}\n")
        out.write("\n")

# Save fix log
with open(AUTO_FIX_LOG, "w", encoding="utf-8") as log_file:
    json.dump(fix_log, log_file, indent=2)

# Validate TTL with RDFLib
g = Graph()
try:
    g.parse(OUTPUT_TTL, format="turtle")
    print(f"✅ Grouped TTL is valid. Total triples: {len(g)}")
except Exception as e:
    print("❌ RDFLib parsing failed!")
    print(e)
