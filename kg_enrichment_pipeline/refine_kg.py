import argparse
import pandas as pd
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

# ---- CONFIG ----
INPUT_KG = "data/enriched_kg_performances.ttl"
TYPE_CSV = "logs/kg_type_refinement_suggestions.csv"
PROP_CSV = "logs/kg_property_refinement_suggestions.csv"
OUTPUT_KG = "data/kg_refined.ttl"

# Namespaces
SCHEMA = "https://schema.org/"
MM = "https://w3id.org/polifonia/ontology/music-meta/"

# Type and property maps
TYPE_MAP = {
    SCHEMA + "Event": MM + "MusicEvent",
    SCHEMA + "BroadcastEvent": MM + "BroadcastingSituation",
    SCHEMA + "MusicGroup": MM + "MusicArtist",
    SCHEMA + "MusicRecording": MM + "MusicalWork",
    SCHEMA + "CreativeWork": MM + "MusicEntity",
}

PROPERTY_MAP = {
    SCHEMA + "performer": MM + "hasPerformer",
    SCHEMA + "location": MM + "performedAt",
    SCHEMA + "inEvent": MM + "isSubEventOf",
    SCHEMA + "subjectOf": MM + "hasSource",
    SCHEMA + "inspiredByEvent": MM + "isModelFor",
    SCHEMA + "recordedAs": MM + "records",
}

def heuristic_type(label, current_type):
    """Fallback heuristics for entity type refinement"""
    label_lower = label.lower()
    if "performance" in label_lower:
        return MM + "LivePerformance"
    if "broadcast" in label_lower or "wembley" in label_lower or "philadelphia" in label_lower:
        return MM + "BroadcastingSituation"
    if "song" in label_lower or "track" in label_lower:
        return MM + "MusicalWork"
    if "artist" in label_lower or "band" in label_lower:
        return MM + "MusicArtist"
    return TYPE_MAP.get(str(current_type))

def generate_suggestions(input_ttl, type_csv, prop_csv):
    g = Graph()
    g.parse(input_ttl, format="ttl")

    type_suggestions = []
    prop_suggestions = []

    # --- Type Suggestions ---
    for s, o in g.subject_objects(RDF.type):
        label = str(g.value(s, RDFS.label) or "")
        suggested = heuristic_type(label, o)
        if suggested and suggested != str(o):
            type_suggestions.append({
                "Entity": str(s),
                "Label": label,
                "Current_Type": str(o),
                "Suggested_mm_Type": suggested,
                "Confidence": "High" if str(o) in TYPE_MAP else "Heuristic"
            })

    # --- Property Suggestions ---
    for s, p, o in g:
        if str(p) in PROPERTY_MAP:
            suggested_prop = PROPERTY_MAP[str(p)]
            if suggested_prop != str(p):
                label = str(g.value(s, RDFS.label) or "")
                prop_suggestions.append({
                    "Entity": str(s),
                    "Label": label,
                    "Old_Property": str(p),
                    "Suggested_Property": suggested_prop,
                    "Object": str(o),
                    "Confidence": "High"
                })

    pd.DataFrame(type_suggestions).to_csv(type_csv, index=False)
    pd.DataFrame(prop_suggestions).to_csv(prop_csv, index=False)
    print(f"💡 Type suggestions saved to {type_csv}")
    print(f"💡 Property suggestions saved to {prop_csv}")

def apply_refinements(input_ttl, type_csv, prop_csv, output_ttl):
    g = Graph()
    g.parse(input_ttl, format="ttl")

    # --- Apply Type Refinements ---
    try:
        df_types = pd.read_csv(type_csv)
        for _, row in df_types.iterrows():
            entity = URIRef(row["Entity"])
            new_type = URIRef(row["Suggested_mm_Type"])
            old_type = URIRef(row["Current_Type"])
            # Remove old type and add new
            g.remove((entity, RDF.type, old_type))
            g.add((entity, RDF.type, new_type))
    except FileNotFoundError:
        print("⚠ No type refinement CSV found. Skipping type refinements.")

    # --- Apply Property Refinements ---
    try:
        df_props = pd.read_csv(prop_csv)
        for _, row in df_props.iterrows():
            entity = URIRef(row["Entity"])
            old_prop = URIRef(row["Old_Property"])
            new_prop = URIRef(row["Suggested_Property"])
            obj_str = row["Object"]
            obj = URIRef(obj_str) if obj_str.startswith("http") else obj_str

            g.remove((entity, old_prop, obj))
            g.add((entity, new_prop, obj))
    except FileNotFoundError:
        print("⚠ No property refinement CSV found. Skipping property refinements.")

    g.serialize(output_ttl, format="ttl")
    print(f"✅ Refined KG saved to {output_ttl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine KG Types and Properties")
    parser.add_argument("--mode", choices=["suggest", "apply"], required=True, help="Operation mode")
    args = parser.parse_args()

    if args.mode == "suggest":
        generate_suggestions(INPUT_KG, TYPE_CSV, PROP_CSV)
    elif args.mode == "apply":
        apply_refinements(INPUT_KG, TYPE_CSV, PROP_CSV, OUTPUT_KG)
