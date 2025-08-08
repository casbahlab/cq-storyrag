import json
from rdflib import Graph, RDF, RDFS, OWL

# ==== CONFIG ====
kg_file = "liveaid_instances.ttl"
musicmeta_file = "musicmeta.owl"
schemaorg_file = "schemaorg.ttl"
output_json = "kg_vocab_report_refined.json"

# ==== NORMALIZER ====
def normalize_schema_iri(iri: str) -> str:
    """Normalize schema.org IRIs to http:// for comparison."""
    return iri.replace("https://schema.org/", "http://schema.org/")

# ==== LOAD ONTOLOGIES ====
g_mm = Graph()
g_mm.parse(musicmeta_file, format="xml")

g_schema = Graph()
g_schema.parse(schemaorg_file, format="turtle")

def extract_valid_uris(graph, normalize_schema=False):
    classes, properties = set(), set()
    for s, _, _ in graph.triples((None, RDF.type, OWL.Class)):
        classes.add(normalize_schema_iri(str(s)) if normalize_schema else str(s))
    for s, _, _ in graph.triples((None, RDF.type, RDFS.Class)):
        classes.add(normalize_schema_iri(str(s)) if normalize_schema else str(s))
    for s, _, _ in graph.triples((None, RDF.type, RDF.Property)):
        properties.add(normalize_schema_iri(str(s)) if normalize_schema else str(s))
    for s, _, _ in graph.triples((None, RDF.type, OWL.ObjectProperty)):
        properties.add(normalize_schema_iri(str(s)) if normalize_schema else str(s))
    for s, _, _ in graph.triples((None, RDF.type, OWL.DatatypeProperty)):
        properties.add(normalize_schema_iri(str(s)) if normalize_schema else str(s))
    return classes, properties

mm_classes, mm_props = extract_valid_uris(g_mm)
schema_classes, schema_props = extract_valid_uris(g_schema, normalize_schema=True)

# ==== LOAD KG ====
g_kg = Graph()
g_kg.parse(kg_file, format="turtle")

report = {
    "valid_properties_musicmeta": [],
    "valid_properties_schemaorg": [],
    "unknown_properties": [],
    "valid_classes_musicmeta": [],
    "valid_classes_schemaorg": [],
    "unknown_classes": []
}

for s, p, o in g_kg:
    p_str = normalize_schema_iri(str(p))

    # ---- Handle properties ----
    if p_str in mm_props:
        report["valid_properties_musicmeta"].append(p_str)
    elif p_str in schema_props:
        report["valid_properties_schemaorg"].append(p_str)
    else:
        report["unknown_properties"].append(p_str)

    # ---- Handle classes ----
    if p == RDF.type:
        class_str = normalize_schema_iri(str(o))
        if class_str in mm_classes:
            report["valid_classes_musicmeta"].append(class_str)
        elif class_str in schema_classes:
            report["valid_classes_schemaorg"].append(class_str)
        else:
            report["unknown_classes"].append(class_str)

# Deduplicate
for k in report:
    report[k] = sorted(set(report[k]))

# ==== Save JSON ====
with open(output_json, "w") as f:
    json.dump(report, f, indent=2)

print(f"Refined report saved to {output_json}")
for k, v in report.items():
    print(f"{k}: {len(v)}")
