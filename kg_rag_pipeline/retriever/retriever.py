from typing import Dict, Any
from rdflib import Graph

def retrieve(plan: Dict[str, Any], kg_path: str):
    """
    plan: Dict -> Plan JSON loaded from file
    kg_path: str -> Path to KG .ttl file
    """
    g = Graph()
    g.parse(kg_path, format="turtle")

    out = {"Entry": [], "Core": [], "Exit": []}

    for cat, items in plan["execution"].items():
        for item in items:
            cqid = item["cq_id"]
            q = item["sparql"]
            question_text = item["question"]
            if not q:
                continue
            try:
                res = g.query(q)
                rows = [{str(v): str(row[v]) for v in res.vars} for row in res]
                out[cat].append({
                    "cq_id": cqid,
                    "rows": rows,
                    "question": question_text
                })
            except:
                print(f"q : {q} , question_text : {question_text} , cqid : {cqid} ")

    return out
