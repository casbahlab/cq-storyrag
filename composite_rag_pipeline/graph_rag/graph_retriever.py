from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import itertools, hashlib
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities

class IRetriever(Protocol):
    def retrieve(self, plan: Dict[str, Any], cq: Dict[str, Any], persona: Dict[str, Any]) -> "RetrievalBundle": ...

@dataclass
class RetrievalBundle:
    facts: List[Dict[str, Any]]
    texts: List[str]
    triples: List[Tuple[str, str, str]]
    subgraph: Any
    provenance: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphRetrieverConfig:
    seed_strategy: List[str] = field(default_factory=lambda: ["cq_entities", "sparql_seed"])
    k_hops: int = 2
    max_nodes: int = 300
    community: str = "label_propagation"  # or "none"
    edge_types_include: Optional[List[str]] = None
    edge_types_exclude: Optional[List[str]] = None
    summarise_subgraph: bool = True

def _qname_or_str(g: Graph, term: rdflib.term.Identifier) -> str:
    try:
        if isinstance(term, URIRef):
            return g.namespace_manager.normalizeUri(term)
        return str(term)
    except Exception:
        return str(term)

def _pred_ok(pred_uri: str, cfg: GraphRetrieverConfig) -> bool:
    if cfg.edge_types_include and not any(pred_uri.endswith(x) or pred_uri == x for x in cfg.edge_types_include):
        return False
    if cfg.edge_types_exclude and any(pred_uri.endswith(x) or pred_uri == x for x in cfg.edge_types_exclude):
        return False
    return True

def seed_from_cq_entities(cq: Dict[str, Any], plan: Dict[str, Any], kg: Graph) -> List[URIRef]:
    seeds: List[URIRef] = []
    for k in ("entities", "seed_entities", "uris"):
        for val in (cq.get(k) or []):
            try: seeds.append(URIRef(str(val)))
            except Exception: pass
    if seeds: return _dedupe_uri_list(seeds)

    labels = []
    for k in ("labels", "entity_labels"): labels.extend(cq.get(k) or [])
    if labels:
        for label in labels:
            q = """
            SELECT ?s WHERE {
              { ?s rdfs:label ?l } UNION { ?s <http://schema.org/name> ?l }
              FILTER (lcase(str(?l)) = lcase(str(?lab)))
            } LIMIT 3
            """
            for row in kg.query(q, initBindings={"lab": Literal(label)}):
                seeds.append(URIRef(row["s"]))
    return _dedupe_uri_list(seeds)

def seed_from_sparql(cq: Dict[str, Any], plan: Dict[str, Any], kg: Graph) -> List[URIRef]:
    sql = cq.get("seed_sparql") or ""
    if not sql.strip(): return []
    out = []
    for row in kg.query(sql):
        s = row.get("s")
        if isinstance(s, URIRef): out.append(s)
    return out

def k_hop_expand(kg: Graph, seeds: List[URIRef], k: int, cfg: GraphRetrieverConfig):
    nodes, edges, frontier, hop = set(seeds), [], set(seeds), 0
    while frontier and hop < k and len(nodes) < cfg.max_nodes:
        nxt = set()
        for s in list(frontier):
            for p, o in kg.predicate_objects(s):
                if not _pred_ok(str(p), cfg): continue
                edges.append((s, p, o))
                if isinstance(o, URIRef) and o not in nodes and len(nodes) < cfg.max_nodes:
                    nxt.add(o); nodes.add(o)
            for ss, p in kg.subject_predicates(s):
                if not _pred_ok(str(p), cfg): continue
                edges.append((ss, p, s))
                if isinstance(ss, URIRef) and ss not in nodes and len(nodes) < cfg.max_nodes:
                    nxt.add(ss); nodes.add(ss)
        frontier, hop = nxt, hop + 1
    return nodes, edges

def _first_label(kg: Graph, node: rdflib.term.Identifier) -> str:
    RDFS = rdflib.namespace.RDFS; SCHEMA = rdflib.Namespace("http://schema.org/")
    for _,_,l in kg.triples((node, RDFS.label, None)): return str(l)
    for _,_,l in kg.triples((node, SCHEMA.name, None)): return str(l)
    return kg.namespace_manager.normalizeUri(node) if isinstance(node, URIRef) else str(node)

def to_networkx(nodes, edges, kg: Graph) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n in nodes: G.add_node(str(n), label=_first_label(kg, n))
    for s, p, o in edges:
        s_id = str(s)
        if isinstance(o, (URIRef, BNode)):
            o_id = str(o)
            if not G.has_node(o_id): G.add_node(o_id, label=_first_label(kg, o) if isinstance(o, URIRef) else str(o))
            G.add_edge(s_id, o_id, predicate=str(p), pred_qname=_qname_or_str(kg, p))
        else:
            lit = f"lit::{hashlib.md5(str(o).encode()).hexdigest()[:8]}"
            G.add_node(lit, label=str(o)); G.add_edge(s_id, lit, predicate=str(p), pred_qname=_qname_or_str(kg, p))
    return G

def prune_by_community(G: nx.MultiDiGraph, seeds: List[URIRef]) -> nx.MultiDiGraph:
    if G.number_of_nodes() == 0: return G
    comms = list(asyn_lpa_communities(G.to_undirected()))
    if not comms: return G
    seed_ids = {str(s) for s in seeds}
    best = max(comms, key=lambda c: len(seed_ids.intersection(set(c))))
    return G.subgraph(best).copy()

def summarise_neighborhood(G: nx.MultiDiGraph, kg: Graph, max_lines: int = 18) -> str:
    if G.number_of_edges() == 0: return ""
    deg = dict(G.degree())
    ranked = sorted(((deg.get(u,0)+deg.get(v,0), u, v, d) for u,v,d in G.edges(data=True)), reverse=True)
    lines = []
    for _, u, v, d in itertools.islice(ranked, 0, max_lines):
        su, sv = G.nodes[u].get("label") or u, G.nodes[v].get("label") or v
        pp = d.get("pred_qname") or d.get("predicate")
        lines.append(f"- {su} → {pp} → {sv}")
    return "Graph neighborhood:\n" + "\n".join(lines)

def export_triples(G: nx.MultiDiGraph):
    tris, prov = [], []
    for u, v, d in G.edges(data=True):
        p = d.get("predicate") or d.get("pred_qname") or "predicate"
        tris.append((u, p, v))
        prov.append({"s": u, "p": p, "o": v, "source": "kg", "confidence": 1.0})
    return tris, prov

def extract_atomic_facts(triples, G: nx.MultiDiGraph):
    return [{"subject": s, "predicate": p, "object": o} for s,p,o in triples[:200]]

def _dedupe_uri_list(xs: List[URIRef]) -> List[URIRef]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

class GraphRetriever(IRetriever):
    def __init__(self, cfg: GraphRetrieverConfig, kg: Graph):
        self.cfg, self.kg = cfg, kg

    def retrieve(self, plan: Dict[str, Any], cq: Dict[str, Any], persona: Dict[str, Any]) -> RetrievalBundle:
        seeds: List[URIRef] = []
        if "cq_entities" in self.cfg.seed_strategy: seeds += seed_from_cq_entities(cq, plan, self.kg)
        if "sparql_seed" in self.cfg.seed_strategy: seeds += seed_from_sparql(cq, plan, self.kg)
        seeds = _dedupe_uri_list(seeds) or ([URIRef(plan["default_event_uri"])] if plan.get("default_event_uri") else [])

        node_set, edge_list = k_hop_expand(self.kg, seeds, self.cfg.k_hops, self.cfg)
        G = to_networkx(node_set, edge_list, self.kg)
        if self.cfg.community != "none": G = prune_by_community(G, seeds)

        triples, prov = export_triples(G)
        summary = summarise_neighborhood(G, self.kg) if self.cfg.summarise_subgraph else ""
        facts = extract_atomic_facts(triples, G)
        meta = {"seeds":[str(s) for s in seeds], "k_hops":self.cfg.k_hops, "nodes":G.number_of_nodes(), "edges":G.number_of_edges(), "community":self.cfg.community}

        return RetrievalBundle(facts=facts, texts=[summary] if summary else [], triples=triples, subgraph=G, provenance=prov, meta=meta)
