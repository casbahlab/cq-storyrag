# kg_profile_void_summary_count.py

from rdflib import Graph, RDF

# Adjust this path if needed
KG_PATH = "../liveaid_instances_master.ttl"

g = Graph()
print(f"[OK] Loading KG from {KG_PATH} ...")
g.parse(KG_PATH, format="turtle")

# 1. Total triples
total_triples = len(g)

classes = set()
predicates = set()
typed_subjects = set()
subject_type_counts = {}

for s, p, o in g:
    predicates.add(p)
    if p == RDF.type:
        classes.add(o)
        typed_subjects.add(s)
        subject_type_counts[s] = subject_type_counts.get(s, 0) + 1

# 2. Distinct classes
distinct_classes = len(classes)

# 3. Distinct predicates (excluding rdf:type)
distinct_predicates_no_type = len([p for p in predicates if p != RDF.type])

# 4. Typed subjects
typed_subjects_count = len(typed_subjects)

# 5. Subjects with dual typing (>= 2 rdf:type assertions)
dual_typed_subjects = sum(1 for n in subject_type_counts.values() if n >= 2)

print("Total Triples:", total_triples)
print("Distinct Classes:", distinct_classes)
print("Distinct Predicates (excl. rdf:type):", distinct_predicates_no_type)
print("Typed Subjects:", typed_subjects_count)
print("Subjects with Dual Typing:", dual_typed_subjects)
