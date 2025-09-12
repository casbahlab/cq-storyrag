from rdflib import Graph
from pyvis.network import Network

INPUT_FILE = "output/liveaid_schema.ttl"
OUTPUT_HTML = "output/liveaid_schema_graph.html"

g = Graph()
g.parse(INPUT_FILE, format="turtle")
print(f"✅ Loaded {len(g)} triples from {INPUT_FILE}")

nodes = set()
edges = []

for subj, pred, obj in g:
    subj_str = str(subj)
    pred_str = str(pred)
    obj_str = str(obj)

    nodes.add(subj_str)
    # Only link URI objects, skip literals
    if obj_str.startswith("http"):
        nodes.add(obj_str)
        edges.append((subj_str, obj_str, pred_str))

print(f"Total nodes: {len(nodes)}, edges: {len(edges)}")

net = Network(notebook=False, height="900px", width="100%", directed=True)
net.toggle_physics(True)  # Make nodes move dynamically

# Add nodes (clean label)
for n in nodes:
    label = n.split("#")[-1] if "#" in n else n.split("/")[-1]
    net.add_node(n, label=label, title=n, color="lightblue")

# Add edges with predicate as label
for subj, obj, pred in edges:
    pred_label = pred.split("#")[-1] if "#" in pred else pred.split("/")[-1]
    net.add_edge(subj, obj, label=pred_label)

net.write_html(OUTPUT_HTML)
print(f"✅ Interactive graph saved to {OUTPUT_HTML}")

