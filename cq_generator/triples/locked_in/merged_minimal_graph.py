from rdflib import Graph

# File paths
schema_path = "minimal_schema_template_manual_updated.ttl"
instance_path = "minimal_instance.ttl"
output_path = "merged_graph.ttl"

# Load schema TTL
schema_graph = Graph()
schema_graph.parse(schema_path, format="turtle")
print(f"Loaded Schema Graph with {len(schema_graph)} triples")

# Load instance TTL
instance_graph = Graph()
instance_graph.parse(instance_path, format="turtle")
print(f"Loaded Instance Graph with {len(instance_graph)} triples")

# Merge the graphs
merged_graph = schema_graph + instance_graph
print(f"Merged Graph contains {len(merged_graph)} triples")

# Save merged TTL
merged_graph.serialize(destination=output_path, format="turtle")
print(f"Merged TTL saved to: {output_path}")
