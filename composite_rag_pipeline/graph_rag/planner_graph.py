#!/usr/bin/env python3
"""
Graph-led planner for Graph-RAG (no CQs).

- Loads RDF (rdflib)
- Expands a k-hop neighborhood from seed entities/labels/SPARQL
- Derives an outline (sections) from the graph shape
- Emits a plan_with_evidence_Graph.json that your dedicated graph generator can consume

Usage (CLI):
  python3 composite_rag_pipeline/graph_rag/planner_graph.py \
    --topic "Queen at Live Aid — Wembley" \
    --rdf data/liveaid_instances_master.ttl \
    --out retriever/plan_with_evidence_Graph.json \
    --config graph_rag/graph_config.yaml \
    --seed-uri http://wembrewind.live/ex#Queen \
    --seed-uri http://wembrewind.live/ex#LiveAid1985_Wembley

You can also pass --seed-label "Queen" etc., or --seeds-json file.json with
{"entities":[...], "labels":[...], "seed_sparql":"SELECT ?s {...}"}
"""

from __future__ import annotations
import argparse, json, hashlib, itertools, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

# deps: rdflib, networkx
import rdflib
from rdflib import Graph as RDFGraph, URIRef, BNode, Literal, Namespace, RDFS
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities

# ---------------- Config loading (optional YAML) ----------------

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML optional for CLI; pipeline can inject dict config

DEFAULT_CFG: Dict[str, Any] = {
    "retrieval": {
        "seed_strategy": ["cq_entities", "sparql_seed"],
        "k_hops": 2,
        "max_nodes": 300,
        "community": "label_propagation",  # "none" to disable
        # sensible defaults for the Live Aid KG (tune as needed)
        "edge_types_include": [
            "involvesMusicEnsemble","involvesMemberOfMusicEnsemble","member",
            "isOriginalMember","recordingOf","startDate","hasPart","inAlbum","location"
        ],
        "edge_types_exclude": [],
        "summarise_subgraph": True,
    },
    "planning": {
        "mode": "communities",           # "single_shot" | "communities" | "timeline" | "paths"
        "num_sections": 5,
        "date_predicates": ["hostedOn","startDate","releaseDate"],
        "top_paths": 3,
    },
    "generation": {
        "max_context_chars": 6000,
        "max_triples": 350,
        "max_facts": 200,
        "citation_style": "cqid",
        "enforce_citation_each_sentence": True,
        "beat_sentences": 4,
    }
}

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_graph_cfg(path: Optional[Path]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CFG)
    if path and path.exists() and yaml:
        file_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = _deep_merge(cfg, file_cfg)
    return cfg

# ---------------- Small utils ----------------

SCHEMA = Namespace("http://schema.org/")

def _slug(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()[:8]

def first_label(g: RDFGraph, node) -> str:
    for _,_,l in g.triples((node, RDFS.label, None)): return str(l)
    for _,_,l in g.triples((node, SCHEMA.name, None)): return str(l)
    return str(node)

def qname_or_str(g: RDFGraph, term) -> str:
    try:
        if isinstance(term, URIRef):
            return g.namespace_manager.normalizeUri(term)
        return str(term)
    except Exception:
        return str(term)

# ---------------- Seeding ----------------

def seeds_from_inputs(g: RDFGraph, seeds_from: Dict[str, Any]) -> List[URIRef]:
    seeds: List[URIRef] = []
    # URIs / entities first (preferred)
    for key in ("entities","uris"):
        for val in (seeds_from.get(key) or []):
            try:
                seeds.append(URIRef(str(val)))
            except Exception:
                pass

    if seeds:
        return _dedupe_uris(seeds)

    # labels fallback
    labels: List[str] = []
    for k in ("labels","entity_labels"):
        labels.extend(seeds_from.get(k) or [])
    if labels:
        for lab in labels:
            q = """
            SELECT ?s WHERE {
              { ?s rdfs:label ?l } UNION { ?s <http://schema.org/name> ?l }
              FILTER (lcase(str(?l)) = lcase(str(?lab)))
            } LIMIT 5
            """
            for row in g.query(q, initBindings={"lab": Literal(lab)}):
                s = row.get("s")
                if isinstance(s, URIRef): seeds.append(s)

    # seed_sparql (expects ?s)
    sql = (seeds_from.get("seed_sparql") or "").strip()
    if sql:
        for row in g.query(sql):
            s = row.get("s")
            if isinstance(s, URIRef): seeds.append(s)

    return _dedupe_uris(seeds)

def _dedupe_uris(xs: Iterable[URIRef]) -> List[URIRef]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ---------------- Expansion & graph build ----------------

def _pred_ok(pred_uri: str, include: List[str] | None, exclude: List[str] | None) -> bool:
    local = pred_uri.split("/")[-1]
    if include and local not in include and pred_uri not in include:
        return False
    if exclude and (local in exclude or pred_uri in exclude):
        return False
    return True

def k_hop_expand(
    g: RDFGraph, seeds: List[URIRef], *, k: int, max_nodes: int,
    include_preds: List[str] | None, exclude_preds: List[str] | None
) -> Tuple[set, List[Tuple[URIRef, URIRef, Any]]]:
    nodes, edges, frontier, hop = set(seeds), [], set(seeds), 0
    while frontier and hop < k and len(nodes) < max_nodes:
        nxt = set()
        for s in list(frontier):
            # outgoing
            for p, o in g.predicate_objects(s):
                if not _pred_ok(str(p), include_preds, exclude_preds): continue
                edges.append((s,p,o))
                if isinstance(o, URIRef) and o not in nodes and len(nodes) < max_nodes:
                    nxt.add(o); nodes.add(o)
            # incoming
            for ss, p in g.subject_predicates(s):
                if not _pred_ok(str(p), include_preds, exclude_preds): continue
                edges.append((ss,p,s))
                if isinstance(ss, URIRef) and ss not in nodes and len(nodes) < max_nodes:
                    nxt.add(ss); nodes.add(ss)
        frontier, hop = nxt, hop + 1
    return nodes, edges

def to_networkx(g: RDFGraph, nodes, edges) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for n in nodes:
        G.add_node(str(n), label=first_label(g, n))
    for s,p,o in edges:
        su = str(s)
        if isinstance(o, (URIRef, BNode)):
            ov = str(o)
            if not G.has_node(ov):
                G.add_node(ov, label=first_label(g, o) if isinstance(o, URIRef) else str(o))
            G.add_edge(su, ov, predicate=str(p), pred_qname=qname_or_str(g, p), pred_local=str(p).split("/")[-1])
        else:
            lit = f"lit::{_slug(str(o))}"
            G.add_node(lit, label=str(o))
            G.add_edge(su, lit, predicate=str(p), pred_qname=qname_or_str(g, p), pred_local=str(p).split("/")[-1])
    return G

def prune_by_community(G: nx.MultiDiGraph, seeds: List[URIRef]) -> nx.MultiDiGraph:
    if G.number_of_nodes() == 0: return G
    comms = list(asyn_lpa_communities(G.to_undirected()))
    if not comms: return G
    seed_ids = {str(s) for s in seeds}
    best = max(comms, key=lambda c: len(seed_ids.intersection(set(c))))
    return G.subgraph(best).copy()

# ---------------- Outline ----------------

@dataclass
class OutlineItem:
    id: str
    beat: str
    node_ids: List[str]

def outline_single_shot(G: nx.MultiDiGraph) -> List[OutlineItem]:
    return [OutlineItem(id="seg_001", beat="Story", node_ids=list(G.nodes()))]

def outline_by_communities(G: nx.MultiDiGraph, num_sections=5) -> List[OutlineItem]:
    comms = list(asyn_lpa_communities(G.to_undirected()))
    comms = sorted(comms, key=lambda c: len(c), reverse=True)[:max(1, num_sections)]
    outs: List[OutlineItem] = []
    for i, c in enumerate(comms, start=1):
        nodes = [str(n) for n in c]
        deg = G.degree(nodes)
        ranked = sorted(nodes, key=lambda n: deg[n], reverse=True)
        title_bits = []
        for nid in ranked:
            pretty = _pretty_name(G, nid)
            if pretty:
                title_bits.append(pretty)
            if len(title_bits) == 3:
                break
        beat = " / ".join(title_bits) if title_bits else f"Cluster {i}"

        outs.append(OutlineItem(id=f"seg_{i:03d}", beat=beat, node_ids=nodes))
    return outs

def outline_by_timeline(G: nx.MultiDiGraph, date_preds: List[str]) -> List[OutlineItem]:
    import re
    buckets: Dict[str, List[str]] = {}
    for u,v,d in G.edges(data=True):
        p_local = d.get("pred_local") or ""
        if any(p_local.endswith(dp) for dp in date_preds):
            for node in (u,v):
                lbl = (G.nodes[node].get("label") or "")
                m = re.search(r"\b(19|20)\d{2}\b", lbl)
                if m:
                    yr = m.group(0)
                    buckets.setdefault(yr, []).extend([u,v])
    if not buckets:
        return outline_single_shot(G)
    outs: List[OutlineItem] = []
    for i, (yr, nodes) in enumerate(sorted(buckets.items(), key=lambda kv: kv[0])):
        uniq = list(dict.fromkeys(nodes))
        outs.append(OutlineItem(id=f"year_{yr}", beat=yr, node_ids=uniq))
    return outs

def outline_by_paths(G: nx.MultiDiGraph, seed_ids: List[str], top_paths=3) -> List[OutlineItem]:
    if not seed_ids:
        return outline_single_shot(G)
    UG = G.to_undirected()
    bc = nx.betweenness_centrality(UG)
    targets = [n for n,_ in sorted(bc.items(), key=lambda kv: kv[1], reverse=True) if n not in seed_ids][:top_paths]
    outs: List[OutlineItem] = []
    idx = 1
    for t in targets:
        for s in seed_ids:
            if nx.has_path(UG, s, t):
                try:
                    path = nx.shortest_path(UG, s, t)
                    outs.append(OutlineItem(id=f"path_{idx:03d}", beat=f"Path {idx}", node_ids=[str(n) for n in path]))
                    idx += 1
                    break
                except Exception:
                    continue
    return outs or outline_single_shot(G)

def make_outline(G: nx.MultiDiGraph, planning_cfg: Dict[str, Any], seeds: List[str]) -> List[OutlineItem]:
    mode = (planning_cfg or {}).get("mode", "communities")
    if mode == "single_shot":
        return outline_single_shot(G)
    if mode == "timeline":
        return outline_by_timeline(G, planning_cfg.get("date_predicates", ["hostedOn","startDate","releaseDate"]))
    if mode == "paths":
        return outline_by_paths(G, seeds, planning_cfg.get("top_paths", 3))
    return outline_by_communities(G, planning_cfg.get("num_sections", 5))

# ---------------- Summarisation & evidence ----------------

# in composite_rag_pipeline/graph_rag/planner_graph.py

def _pretty_name(G, nid: str) -> str | None:
    """Prefer labels; fall back to local name; drop opaque membership IDs."""
    lbl = G.nodes.get(nid, {}).get("label")
    if lbl and not lbl.startswith("http"):
        return lbl
    local = nid.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    # hide reified membership nodes entirely
    if local.startswith("Membership_"):
        return None
    return local.replace("_", " ")

def _compress_membership_triples(G, nodes: list[str]) -> list[tuple[str,str,str]]:
    """
    Collapse reified membership nodes into readable 'Queen | member (since YYYY-MM) | Barry Mitchell'.
    Looks for a node that links to both ensemble and member, plus optional startDate literal.
    """
    from collections import defaultdict
    sub = G.subgraph(nodes)
    # gather membership-style hubs: they have edges to ensemble & member
    hubs = defaultdict(dict)  # hub -> {"band": str, "person": str, "since": "YYYY-MM"}
    for u, v, d in sub.edges(data=True):
        p = d.get("pred_local") or d.get("pred_qname") or ""
        if "involvesMusicEnsemble" in p or p == "involvesMusicEnsemble":
            hubs[u]["band"] = v
        if "involvesMemberOfMusicEnsemble" in p or p == "involvesMemberOfMusicEnsemble":
            hubs[u]["person"] = v
        if "startDate" in p or p.endswith("startDate"):
            hubs[u]["since"] = sub.nodes[v].get("label") or v  # literal node label holds date
    out = []
    for hub, info in hubs.items():
        band = _pretty_name(G, info.get("band", ""))
        person = _pretty_name(G, info.get("person", ""))
        if band and person:
            since = info.get("since")
            pred = "member" + (f" (since {since})" if since else "")
            out.append((band, pred, person))
    return out

def section_summary(G: nx.MultiDiGraph, nodes: List[str], max_lines: int) -> str:
    """Readable 1-hop neighborhood; **no raw IDs**."""
    sub = G.subgraph(nodes)
    if sub.number_of_edges() == 0:
        return "Graph neighborhood:"
    deg = dict(sub.degree())
    scored = sorted(sub.edges(data=True), key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
    lines = []
    for u, v, d in scored:
        su = _pretty_name(G, u); sv = _pretty_name(G, v)
        if not su or not sv:         # skip membership hubs & opaque nodes
            continue
        p = d.get("pred_local") or d.get("pred_qname") or "relatedTo"
        lines.append(f"- {su} → {p} → {sv}")
        if len(lines) >= max_lines:
            break
    # also inject compressed membership statements (readable)
    for band, pred, person in _compress_membership_triples(G, nodes):
        if len(lines) >= max_lines:
            break
        lines.append(f"- {band} → {pred} → {person}")
    return "Graph neighborhood:\n" + "\n".join(lines)

def section_triples(G: nx.MultiDiGraph, nodes: List[str], limit: int) -> List[tuple[str,str,str]]:
    """Facts for the LLM: **labels only**, membership hubs flattened, no URIs/lit::."""
    sub = G.subgraph(nodes)
    out, seen = [], set()

    # 1) compressed membership relations
    for band, pred, person in _compress_membership_triples(G, nodes):
        tup = (band, pred, person)
        if tup not in seen:
            out.append(tup); seen.add(tup)
            if len(out) >= limit: return out

    # 2) regular edges with human names
    for u, v, d in sub.edges(data=True):
        su = _pretty_name(G, u); sv = _pretty_name(G, v)
        if not su or not sv:
            continue
        p = d.get("pred_local") or d.get("pred_qname") or "relatedTo"
        tup = (su, p, sv)
        if tup in seen:
            continue
        out.append(tup); seen.add(tup)
        if len(out) >= limit:
            break
    return out

def section_triples(G: nx.MultiDiGraph, nodes: List[str], limit: int) -> List[Tuple[str,str,str]]:
    sub = G.subgraph(nodes)
    out: List[Tuple[str,str,str]] = []
    seen = set()
    for u,v,d in sub.edges(data=True):
        p = d.get("predicate") or d.get("pred_qname") or d.get("pred_local") or "predicate"
        tup = (u, p, v)
        if tup in seen: continue
        out.append(tup); seen.add(tup)
        if len(out) >= limit: break
    return out

# --- Neighbour crawl / expansion ---------------------------------------------

PRED_WEIGHTS_DEFAULT = {
    "involvesMusicEnsemble": 1.6,
    "involvesMemberOfMusicEnsemble": 1.6,
    "member": 1.3,
    "isOriginalMember": 1.4,
    "recordingOf": 1.2,
    "hasPart": 1.2,
    "inAlbum": 1.0,
    "location": 1.1,
    "startDate": 0.8,
}

def _rank_frontier_neighbours(G: nx.MultiDiGraph, section_nodes: set[str],
                              pred_weights: dict[str, float], top_k: int):
    neighbours: dict[str, dict] = {}
    for u in section_nodes:
        if not G.has_node(u):
            continue
        # outgoing
        for _, v, d in G.out_edges(u, data=True):
            if v in section_nodes:
                continue
            pl = d.get("pred_local","")
            w = pred_weights.get(pl, 1.0)
            r = neighbours.setdefault(v, {"score": 0.0, "via": []})
            r["score"] += w
            r["via"].append((u, pl, v))
        # incoming
        for v, _, d in G.in_edges(u, data=True):
            if v in section_nodes:
                continue
            pl = d.get("pred_local","")
            w = pred_weights.get(pl, 1.0)
            r = neighbours.setdefault(v, {"score": 0.0, "via": []})
            r["score"] += w
            r["via"].append((v, pl, u))
    # degree tie-breaker
    for v, r in neighbours.items():
        r["score"] += 0.1 * G.degree(v)
    ranked = sorted(neighbours.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]
    return ranked  # list[(node_id, {"score":..., "via":[(a,p,b), ...]})]

def _expand_section_evidence(G: nx.MultiDiGraph, item: dict, ranked_frontier, *,
                             extra_facts_per_section: int,
                             summary_append_lines: int):
    ev = item.get("evidence", [])
    # find the summary block
    sidx = next((i for i,e in enumerate(ev) if e.get("type") == "text"), None)
    summary_lines = []
    extra_edges = []
    for node_id, rec in ranked_frontier:
        # include the via edges
        extra_edges.extend(rec["via"])
        # also include 1-hop local edges around frontier node (enrich)
        for _, v, d in G.out_edges(node_id, data=True):
            extra_edges.append((node_id, d.get("pred_local",""), v))
        for v, _, d in G.in_edges(node_id, data=True):
            extra_edges.append((v, d.get("pred_local",""), node_id))

    # summarise a few human-readable lines
    if summary_append_lines > 0 and sidx is not None:
        count = 0
        for a, p, b in extra_edges:
            if str(a).startswith("lit::") or str(b).startswith("lit::"):
                continue
            la = G.nodes[a].get("label") or a
            lb = G.nodes[b].get("label") or b
            summary_lines.append(f"- {la} → {p} → {lb}")
            count += 1
            if count >= summary_append_lines:
                break
        if summary_lines:
            ev[sidx]["value"] = (ev[sidx]["value"].rstrip() + "\n" + "\n".join(summary_lines)).strip()

    # append extra facts (dedup)
    seen = set()
    for block in ev:
        if block.get("type") in {"fact","triple"}:
            seen.add(block.get("value",""))
    added = 0
    for a, p, b in extra_edges:
        fact = f"{a} | {p} | {b}"
        if fact in seen:
            continue
        ev.append({"type":"fact","value":fact,"source":"kg"})
        seen.add(fact)
        added += 1
        if added >= extra_facts_per_section:
            break
    item["evidence"] = ev

def _spawn_section_from_frontier(G: nx.MultiDiGraph, node_id: str, idx: int) -> dict:
    # lightweight new section seeded by the top frontier node neighbourhood
    lbl = G.nodes[node_id].get("label") or node_id
    seg_nodes = set([node_id])
    # pull a small ego network
    for _, v, _ in G.out_edges(node_id, data=True):
        seg_nodes.add(v)
    for v, _, _ in G.in_edges(node_id, data=True):
        seg_nodes.add(v)
    # build summary and facts
    def _summary(nodes: list[str], limit=20):
        lines = []
        sub = G.subgraph(nodes)
        deg = dict(sub.degree())
        edges = sorted(sub.edges(data=True), key=lambda e: deg[e[0]]+deg[e[1]], reverse=True)[:limit]
        for u,v,d in edges:
            lu = sub.nodes[u].get("label") or u
            lv = sub.nodes[v].get("label") or v
            lines.append(f"- {lu} → {d.get('pred_local','')} → {lv}")
        return "Graph neighborhood:\n" + "\n".join(lines)

    def _facts(nodes: list[str], limit=150):
        sub = G.subgraph(nodes)
        out, seen = [], set()
        for u,v,d in sub.edges(data=True):
            tup = (u, d.get("pred_local",""), v)
            if tup in seen:
                continue
            out.append({"type":"fact","value":f"{u} | {d.get('pred_local','')} | {v}", "source":"kg"})
            seen.add(tup)
            if len(out) >= limit:
                break
        return out

    nodes_list = list(seg_nodes)
    ev = [{"type":"text","value":_summary(nodes_list),"source":"graph"}] + _facts(nodes_list)
    return {
        "id": f"seg_spawn_{idx:02d}",
        "beat": f"{lbl}",
        "question": None,
        "evidence": ev,
        "urls": [],
        "meta": {"segment_nodes": nodes_list, "spawned_from": node_id},
    }


# ---------------- Planner (public) ----------------
def load_persona_pack(name: str, path="config/personas.yaml") -> dict:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    p = (data.get("personas") or {}).get(name) or {}
    # safe fallbacks
    return {
        "name": name,
        "description": p.get("description", "General reader; prefers clarity over flourish."),
        "tone": p.get("tone", ["clear", "neutral"]),
        "reading_level": p.get("reading_level", "~10th grade"),
        "length_words": p.get("length_words", [160, 220]),
        "coverage": p.get("coverage", {"min_factlets": 4, "min_pct": 0.7, "require_breadth_buckets": True}),
        "buckets": p.get("buckets", ["people", "place", "time", "action", "impact"]),
        "citations": p.get("citations", {"per_sentence": True, "style": "Bracket IDs like [CQ-...; CQ-...]"}),
        "dos": p.get("dos", ["Use concrete evidence.", "Prefer short sentences."]),
        "donts": p.get("donts", ["Do not roleplay.", "Do not add facts."]),
    }

def build_graph_plan(
    *,
    topic: str,
    rdf_files: List[Path],
    graph_cfg: Dict[str, Any],
    persona_name: str,
    seeds_from: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a graph-led plan_with_evidence dict (does not write)."""
    persona = load_persona_pack(persona_name, path="config/personas.yaml")
    # 1) Load KG
    kg = RDFGraph()
    for f in rdf_files:
        kg.parse(str(f))

    # 2) Resolve seeds
    if seeds_from is None:
        seeds_from = {"labels": [topic]}  # minimal fallback
    seed_uris = seeds_from_inputs(kg, seeds_from)
    if not seed_uris:
        # final fallback to topic label search
        seed_uris = seeds_from_inputs(kg, {"labels": [topic]})

    # 3) Expand
    rcfg = graph_cfg.get("retrieval", {})
    nodes, edges = k_hop_expand(
        kg, seed_uris,
        k=int(rcfg.get("k_hops", 2)),
        max_nodes=int(rcfg.get("max_nodes", 300)),
        include_preds=rcfg.get("edge_types_include"),
        exclude_preds=rcfg.get("edge_types_exclude"),
    )
    G = to_networkx(kg, nodes, edges)
    if rcfg.get("community", "label_propagation") != "none":
        G = prune_by_community(G, seed_uris)

    # 4) Outline
    seeds_str = [str(s) for s in seed_uris]
    outline = make_outline(G, graph_cfg.get("planning", {}), seeds_str)

    # 5) Evidence per section
    genc = graph_cfg.get("generation", {})
    max_context_chars = int(genc.get("max_context_chars", 6000))
    max_triples = int(genc.get("max_triples", 350))
    max_facts   = int(genc.get("max_facts", 200))
    beat_sentences = int(genc.get("beat_sentences", 4))

    items: List[Dict[str, Any]] = []
    for seg in outline:
        summ = section_summary(G, seg.node_ids, max_lines=20)[:max_context_chars]
        tris = section_triples(G, seg.node_ids, limit=max_triples)
        ev = [{"type": "text", "value": summ, "source": "graph"}]
        for (s,p,o) in tris:
            ev.append({"type": "fact", "value": f"{s} | {p} | {o}", "source": "kg"})
            if len(ev) >= 1 + max_facts:
                break
        items.append({
            "id": seg.id,
            "beat": seg.beat,
            "question": f"{topic} — {seg.beat}",  # optional; generator ignores CQ framing
            "evidence": ev,
            "urls": [],
            "meta": {"segment_nodes": seg.node_ids},
        })

    # --- Neighbour expansion & optional spawning ---
    expand_cfg = (graph_cfg.get("planning") or {}).get("expand") or {}
    if expand_cfg.get("enable", False):
        pred_w = dict(PRED_WEIGHTS_DEFAULT)
        topk   = int(expand_cfg.get("top_neighbours_per_section", 8))
        extra  = int(expand_cfg.get("extra_facts_per_section", 80))
        s_lines= int(expand_cfg.get("summary_append_lines", 25))
        spawn  = bool(expand_cfg.get("spawn_new_sections", False))
        spawn_max = int(expand_cfg.get("spawn_sections_max", 2))

        spawned = []
        for idx, it in enumerate(items):
            seg_nodes = set(it.get("meta", {}).get("segment_nodes", []))
            ranked = _rank_frontier_neighbours(G, seg_nodes, pred_w, topk)
            _expand_section_evidence(G, it, ranked,
                                     extra_facts_per_section=extra,
                                     summary_append_lines=s_lines)
            if spawn and len(spawned) < spawn_max and ranked:
                top_node = ranked[0][0]
                spawned.append(_spawn_section_from_frontier(G, top_node, len(spawned)+1))

        if spawned:
            items.extend(spawned)


    plan = {
        "plan_id": f"graph_auto_{_slug(topic)}",
        "topic": topic,
        "persona": persona or {},
        "length": "auto",
        "items": items,
        "graph_generation": {
            "max_context_chars": max_context_chars,
            "max_triples": max_triples,
            "max_facts": max_facts,
            "citation_style": genc.get("citation_style", "cqid"),
            "enforce_citation_each_sentence": bool(genc.get("enforce_citation_each_sentence", True)),
            "beat_sentences": beat_sentences,
        }
    }
    return plan

def write_plan(plan: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- CLI ----------------

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Graph-led planner for Graph-RAG")
    ap.add_argument("--topic", required=True, help="Display topic/title for the story")
    ap.add_argument("--rdf", required=True, nargs="+", help="RDF files (ttl/nt/rdf)")
    ap.add_argument("--out", required=True, help="Output JSON path (plan_with_evidence_Graph.json)")
    ap.add_argument("--config", default=None, help="graph_rag/graph_config.yaml (optional)")
    ap.add_argument("--persona", default=None, help="Path to persona JSON (optional)")
    # seeds
    ap.add_argument("--seed-uri", action="append", default=[], help="Seed entity URI (repeatable)")
    ap.add_argument("--seed-label", action="append", default=[], help="Seed label (repeatable)")
    ap.add_argument("--seed-sparql", default=None, help="SPARQL SELECT returning ?s seeds")
    ap.add_argument("--seeds-json", default=None, help="Path to a seeds_from JSON")
    return ap.parse_args()

def _load_persona(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def main():
    args = _parse_cli()
    cfg = load_graph_cfg(Path(args.config)) if args.config else dict(DEFAULT_CFG)
    persona = _load_persona(args.persona)

    seeds_from: Dict[str, Any] = {}
    if args.seeds_json and Path(args.seeds_json).exists():
        seeds_from = json.loads(Path(args.seeds_json).read_text(encoding="utf-8"))
    else:
        if args.seed_uri:   seeds_from["entities"] = args.seed_uri
        if args.seed_label: seeds_from.setdefault("labels", []).extend(args.seed_label)
        if args.seed_sparql: seeds_from["seed_sparql"] = args.seed_sparql
        if not seeds_from:  seeds_from = {"labels": [args.topic]}

    plan = build_graph_plan(
        topic=args.topic,
        rdf_files=[Path(p) for p in args.rdf],
        graph_cfg=cfg,
        persona=persona,
        seeds_from=seeds_from,
    )
    write_plan(plan, Path(args.out))
    print(f"✓ Wrote plan with {len(plan.get('items', []))} sections to {args.out}")

if __name__ == "__main__":
    main()
