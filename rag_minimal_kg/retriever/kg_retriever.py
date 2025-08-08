# retriever/kg_retriever.py

from rdflib import Graph
from typing import List, Tuple

class KGTripleRetriever:
    def __init__(self, ttl_path: str):
        self.graph = Graph()
        self.graph.parse(ttl_path, format="turtle")

    def query_triples(self, subject_filter: str = None) -> List[Tuple[str, str, str]]:
        query = """
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
            %s
        }
        LIMIT 50
        """ % (
            f'FILTER(CONTAINS(STR(?s), "{subject_filter}"))' if subject_filter else ""
        )

        results = self.graph.query(query)
        return [(str(s), str(p), str(o)) for s, p, o in results]

# Path to your TTL graph
KG_PATH = "retriever/data/merged_graph.ttl"

# Convenience method for main.py
def retrieve_triples(subject_filter: str = None) -> List[Tuple[str, str, str]]:
    retriever = KGTripleRetriever(KG_PATH)
    return retriever.query_triples(subject_filter=subject_filter)

