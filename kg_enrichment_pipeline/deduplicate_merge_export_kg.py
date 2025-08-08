import json
import re
from pathlib import Path

# ===== CONFIG =====
INPUT_FILE = "output/refined_triples_fixed.json"
OUTPUT_JSON = "output/deduplicated_triples.json"
OUTPUT_TTL = "output/deduplicated_triples.ttl"
FIX_LOG = "logs/fixed_triples.json"
DROP_LOG = "logs/dropped_triples.json"

Path("output").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

PREFIXES = """@prefix ex: <http://example.org/resource/> .
@prefix schema: <https://schema.org/> .
@prefix mm: <http://example.org/musicmeta/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix musicbrainz: <https://musicbrainz.org/> .
@prefix wikidata: <http://www.wikidata.org/entity/> .
"""

VALID_PREFIXES = ("ex:", "schema:", "mm:", "rdfs:", "rdf:", "xsd:", "musicbrainz:", "wikidata:", "<")
REVIEW_KEYWORDS = ("Fact:", "Flagged", "CQ", "Sources:", "1=✅", "Current", "Triple")

def load_triples(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_component(value: str) -> str:
    return value.strip()

def is_review_metadata(component: str) -> bool:
    return any(component.startswith(k) for k in REVIEW_KEYWORDS)

def auto_fix_object(o: str, predicate: str):
    original = o.strip()
    fixed = original

    # Handle booleans
    if fixed.lower() in ["true", "false"]:
        return f"\"{fixed.lower()}\"^^xsd:boolean", "boolean"

    # Handle integers
    if re.fullmatch(r"\d+", fixed):
        if 1900 <= int(fixed) <= 2099 and "date" in predicate.lower():
            return f"\"{fixed}\"^^xsd:gYear", "gYear"
        return f"\"{fixed}\"^^xsd:integer", "integer"

    # Add quotes if literal with spaces and not already quoted
    if " " in fixed and not fixed.startswith('"'):
        return f"\"{fixed}\"", "quoted_literal"

    # Auto-prefix bare identifiers
    if not fixed.startswith(VALID_PREFIXES):
        return f"ex:{fixed}", "auto_prefixed"

    return fixed, None  # no fix needed

def deduplicate_clean_log(triples):
    triple_map = {}
    dropped = []
    fixed = []

    for t in triples:
        s = normalize_component(t.get("subject", ""))
        p = normalize_component(t.get("predicate", ""))
        o = normalize_component(t.get("object", ""))

        if not s or not p or not o:
            dropped.append(t)
            continue
        if is_review_metadata(s) or is_review_metadata(p) or is_review_metadata(o):
            dropped.append(t)
            continue

        # Fix object
        o_fixed, fix_type = auto_fix_object(o, p)

        if fix_type:
            fixed.append({
                "original_object": o,
                "fixed_object": o_fixed,
                "fix_type": fix_type,
                "subject": s,
                "predicate": p
            })

        key = (s, p, o_fixed)
        if key not in triple_map:
            triple_map[key] = {
                "subject": s,
                "predicate": p,
                "object": o_fixed,
                "cq_ids": set(),
                "aliases": set(),
                "sources": set(),
                "status": t.get("status", "accepted")
            }

        if t.get("cq_ids"):
            triple_map[key]["cq_ids"].update(t["cq_ids"] if isinstance(t["cq_ids"], list) else [t["cq_ids"]])
        if t.get("aliases"):
            triple_map[key]["aliases"].update(t["aliases"] if isinstance(t["aliases"], list) else [t["aliases"]])
        if t.get("source"):
            triple_map[key]["sources"].update(t["source"] if isinstance(t["source"], list) else [t["source"]])

    merged_triples = []
    for data in triple_map.values():
        data["cq_ids"] = sorted(data["cq_ids"])
        data["aliases"] = sorted(data["aliases"])
        data["sources"] = sorted(data["sources"])
        merged_triples.append(data)

    merged_triples.sort(key=lambda x: (x["subject"], x["predicate"], x["object"]))
    return merged_triples, dropped, fixed

def export_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def export_ttl(triples, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(PREFIXES + "\n")
        for t in triples:
            f.write(f"{t['subject']} {t['predicate']} {t['object']} .\n")

def main():
    triples = load_triples(INPUT_FILE)
    cleaned, dropped, fixed = deduplicate_clean_log(triples)

    export_json(cleaned, OUTPUT_JSON)
    export_ttl(cleaned, OUTPUT_TTL)
    export_json(fixed, FIX_LOG)
    export_json(dropped, DROP_LOG)

    print(f"✅ Deduplication & cleaning complete.")
    print(f"→ Unique triples: {len(cleaned)}")
    print(f"→ Dropped triples: {len(dropped)}  (saved in {DROP_LOG})")
    print(f"→ Auto-fixed objects: {len(fixed)}  (saved in {FIX_LOG})")
    print(f"→ TTL: {OUTPUT_TTL}")

    if dropped:
        print(f"⚠ {len(dropped)} triples were dropped (likely review text or empty).")
    if fixed:
        print(f"🛠 {len(fixed)} objects were fixed (booleans, years, prefixes, literals).")
    if not dropped and not fixed:
        print("✅ TTL is fully clean and ready for ingestion.")

if __name__ == "__main__":
    main()
