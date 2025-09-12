# composite_rag_pipeline/graph_rag/outline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import itertools
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities

@dataclass
class OutlineItem:
    id: str
    beat: str
    question: str | None   # optional; generator_dual accepts 'question' text field
    node_ids: List[str]    # nodes belonging to this segment

def _top_labels(G: nx.MultiDiGraph, nodes: List[str], k=3) -> List[str]:
    deg = G.degree(nodes)
    ranked = sorted(nodes, key=lambda n: deg[n], reverse=True)
    labels = []
    for n in ranked[:k]:
        lbl = G.nodes[n].get("label") or n
        labels.append(lbl)
    return labels

def outline_single_shot(G: nx.MultiDiGraph) -> List[OutlineItem]:
    return [OutlineItem(id="seg_001", beat="Story", question=None, node_ids=list(G.nodes()))]

def outline_by_communities(G: nx.MultiDiGraph, num_sections=4) -> List[OutlineItem]:
    comms = list(asyn_lpa_communities(G.to_undirected()))
    # sort by size desc
    comms = sorted(comms, key=lambda c: len(c), reverse=True)[:max(1, num_sections)]
    items: List[OutlineItem] = []
    for i, c in enumerate(comms, start=1):
        nodes = [str(n) for n in c]
        title_bits = _top_labels(G, nodes, k=3)
        beat = " / ".join(title_bits) if title_bits else f"Cluster {i}"
        items.append(OutlineItem(id=f"seg_{i:03d}", beat=beat, question=None, node_ids=nodes))
    return items

def outline_by_timeline(G: nx.MultiDiGraph, date_predicates: List[str]) -> List[OutlineItem]:
    # crude heuristic: group edges that touch date-like literals by year
    import re
    buckets: Dict[str, List[str]] = {}  # year -> node ids
    for u, v, d in G.edges(data=True):
        p = (d.get("predicate") or "").split("/")[-1]
        if any(p.endswith(dp) for dp in date_predicates):
            # literal is usually v or a child; try to parse YYYY
            for node in (u, v):
                lbl = (G.nodes[node].get("label") or "")
                m = re.search(r"\b(19|20)\d{2}\b", lbl)
                if m:
                    year = m.group(0)
                    buckets.setdefault(year, [])
                    buckets[year].extend([u, v])
    if not buckets:
        return outline_single_shot(G)
    items = []
    for i, (year, nodes) in enumerate(sorted(buckets.items(), key=lambda kv: kv[0])):
        uniq = list(dict.fromkeys(nodes))
        items.append(OutlineItem(id=f"year_{year}", beat=f"{year}", question=None, node_ids=uniq))
    return items

def outline_by_paths(G: nx.MultiDiGraph, seeds: List[str], top_paths=3) -> List[OutlineItem]:
    # find high-centrality targets and compute simple paths from seeds
    if not seeds:
        return outline_single_shot(G)
    UG = G.to_undirected()
    bc = nx.betweenness_centrality(UG)
    # top targets excluding seeds
    candidates = [n for n, _ in sorted(bc.items(), key=lambda kv: kv[1], reverse=True) if n not in seeds][:top_paths]
    items: List[OutlineItem] = []
    idx = 1
    for t in candidates:
        # pick the seed with a path
        for s in seeds:
            if nx.has_path(UG, s, t):
                try:
                    path = nx.shortest_path(UG, s, t)
                    items.append(OutlineItem(id=f"path_{idx:03d}", beat=f"Path {idx}", question=None, node_ids=[str(n) for n in path]))
                    idx += 1
                    break
                except Exception:
                    continue
    return items or outline_single_shot(G)

def make_outline(G: nx.MultiDiGraph, strategy: Dict[str, Any], seeds: List[str]) -> List[OutlineItem]:
    mode = (strategy or {}).get("mode", "communities")
    if mode == "single_shot":
        return outline_single_shot(G)
    if mode == "timeline":
        return outline_by_timeline(G, strategy.get("date_predicates", ["hostedOn","startDate","releaseDate"]))
    if mode == "paths":
        return outline_by_paths(G, seeds, strategy.get("top_paths", 3))
    # default
    return outline_by_communities(G, strategy.get("num_sections", 4))
