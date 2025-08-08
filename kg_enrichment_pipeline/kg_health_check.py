import rdflib
from collections import Counter

KG_FILE = "output/final_with_inferred.ttl"

g = rdflib.Graph()
g.parse(KG_FILE, format="turtle")

print(f"✅ Parsed KG with {len(g)} triples")

# --- 1. Untyped Literals ---
untyped_literals = []
for s, p, o in g:
    if isinstance(o, rdflib.Literal):
        # Detect numeric or year-like strings
        if o.datatype is None:
            if o.value.isnumeric() or o.value.replace(",", "").isdigit():
                untyped_literals.append((s, p, o))

print(f"⚠ Untyped literals: {len(untyped_literals)}")
for triple in untyped_literals[:10]:
    print("  ", triple)

# --- 2. Custom ex: entities missing rdf:type ---
ex_ns = rdflib.Namespace("http://wembrewind.live/ex#")
missing_type = []
for subj in set(g.subjects()):
    if isinstance(subj, rdflib.URIRef) and str(subj).startswith(str(ex_ns)):
        types = list(g.objects(subj, rdflib.RDF.type))
        if not types:
            missing_type.append(subj)

print(f"⚠ ex: entities without rdf:type: {len(missing_type)}")
for s in missing_type[:10]:
    print("  ", s)

# --- 3. Duplicate Triples ---
triple_list = [(str(s), str(p), str(o)) for s,p,o in g]
dup_count = len(triple_list) - len(set(triple_list))
print(f"⚠ Duplicate triples: {dup_count}")

# --- 4. Blank Nodes ---
blank_nodes = [s for s in g.all_nodes() if isinstance(s, rdflib.BNode)]
print(f"⚠ Blank nodes found: {len(blank_nodes)}")

# --- 5. Summary ---
print("\n--- SUMMARY ---")
print(f"Total Triples: {len(g)}")
print(f"Unique Subjects: {len(set(g.subjects()))}")
print(f"Unique Predicates: {len(set(g.predicates()))}")
print(f"Unique Objects: {len(set(g.objects()))}")
print(f"Issues: {len(untyped_literals)} untyped literals, "
      f"{len(missing_type)} missing types, "
      f"{dup_count} duplicates, "
      f"{len(blank_nodes)} blank nodes")
