from typing import Dict, Any, List
from rdflib import Graph, URIRef, Literal
import re

def strip_known_bases(url: str) -> str:
    if url.startswith("http://wembrewind.live/ex#"):
        return url.replace("http://wembrewind.live/ex#", "")
    if url.startswith("http://schema.org/"):
        return url.replace("http://schema.org/", "")
    return url

def _value_to_str(val):
    if isinstance(val, URIRef):
        return str(val)
    if isinstance(val, Literal):
        return str(val)
    return str(val)

def _enrich_labels(g: Graph, iris: List[str]) -> Dict[str, str]:
    labels = {}
    # Simpler per-IRI query (fine for current size). Batch later if needed.
    for iri in iris:
        try:
            q = f"""
            SELECT ?l WHERE {{
              <{iri}> <http://www.w3.org/2000/01/rdf-schema#label> ?l .
              FILTER (lang(?l) = "" || langMatches(lang(?l), "en"))
            }} LIMIT 1
            """
            res = g.query(q)
            for row in res:
                labels[iri] = str(row[0])
                break
        except Exception:
            continue
    return labels

def retrieve(plan: Dict[str, Any], kg_ttl: str, *, enrich_labels: bool = True) -> Dict[str, Any]:
    g = Graph()
    g.parse(kg_ttl, format="turtle")
    out = {"Entry": [], "Core": [], "Exit": []}

    for cat in ["Entry", "Core", "Exit"]:
        for item in plan["execution"].get(cat, []):
            q = item.get("sparql", "").strip()
            question_text = item.get("question", "")
            cqid = item.get("cq_id", "")

            if not q:
                continue
            try:
                res = g.query(q)
                rows = []
                iris = set()
                for row in res:
                    obj = {}
                    for v in res.vars:
                        s = _value_to_str(row[v])
                        obj[str(v)] = strip_known_bases(s)
                        if s.startswith("http://") or s.startswith("https://"):
                            iris.add(s)
                    rows.append(obj)

                labels = _enrich_labels(g, list(iris)) if enrich_labels and iris else {}

                out[cat].append({
                    "cq_id": cqid,
                    "question": question_text,
                    "rows": rows,
                    "labels": labels,
                    "provenance": {"kg": kg_ttl}
                })
            except Exception as e:
                print(f"[WARN] SPARQL failed for {cqid}: {e}\nQuery: {q[:200]}...")

    return out