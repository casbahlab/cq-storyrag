# composite_rag_pipeline/graph_rag/adapter_build_plan_with_evidence.py
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Any, Dict, List
from rdflib import Graph
from graph_rag.graph_retriever import GraphRetriever, GraphRetrieverConfig
from graph_rag.outline import make_outline
import yaml

def _slug(x: str) -> str:
    return hashlib.md5(x.encode("utf-8")).hexdigest()[:8]


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

def build_graph_auto_plan_with_evidence(
    *,
    persona_name: str,
    topic: str,                         # seed topic or display title
    rdf_files: List[str],
    out_path: Path,
    graph_cfg: Dict[str, Any],
    seeds_from: Dict[str, Any] | None = None,  # optional seed URIs/labels
) -> Dict[str, Any]:
    persona = load_persona_pack(persona_name, path="config/personas.yaml")
    # 1) Load KG
    kg = Graph()
    for f in rdf_files:
        kg.parse(str(f))

    # 2) Retriever config
    rcfg = graph_cfg.get("retrieval", {})
    retriever_cfg = GraphRetrieverConfig(
        seed_strategy=rcfg.get("seed_strategy", ["cq_entities","sparql_seed"]),
        k_hops=int(rcfg.get("k_hops", 2)),
        max_nodes=int(rcfg.get("max_nodes", 300)),
        community=rcfg.get("community", "label_propagation"),
        edge_types_include=rcfg.get("edge_types_include") or ["performedAt","performedSong","influencedBy"],
        edge_types_exclude=rcfg.get("edge_types_exclude") or [],
        summarise_subgraph=bool(rcfg.get("summarise_subgraph", True)),
    )
    retriever = GraphRetriever(retriever_cfg, kg)

    # 3) Minimal pseudo-CQ to drive seeding
    cq_stub = {"id": _slug(topic), "text": topic}
    if seeds_from:
        cq_stub.update(seeds_from)

    # 4) Retrieve graph neighborhood
    plan_stub = {"persona": persona, "default_event_uri": None}
    bundle = retriever.retrieve(plan=plan_stub, cq=cq_stub, persona=persona)

    # 5) Derive outline (beats) from the subgraph
    strategy = graph_cfg.get("planning", {}) or {"mode": "communities"}
    seeds_str = [str(s) for s in bundle.meta.get("seeds", [])]
    outline = make_outline(bundle.subgraph, strategy, seeds_str)

    # 6) Prepare evidence per section
    genc = graph_cfg.get("generation", {})
    max_context_chars = int(genc.get("max_context_chars", 6000))
    max_triples = int(genc.get("max_triples", 350))
    max_facts   = int(genc.get("max_facts", 200))

    def _summary_for_nodes(node_ids: List[str]) -> str:
        # re-use the already computed summary; keep it simple for now
        txt = "\n".join(bundle.texts)
        if len(txt) > max_context_chars:
            txt = txt[:max_context_chars]
        return txt

    # Trim triples to per-section nodes for a tighter context
    triples = [(s,p,o) for (s,p,o) in bundle.triples if (s in bundle.subgraph) and (o in bundle.subgraph)]
    items = []
    for seg in outline:
        seg_nodes = set(seg.node_ids)
        seg_triples = [(s,p,o) for (s,p,o) in triples if (s in seg_nodes) or (o in seg_nodes)]
        ev = []
        summary = _summary_for_nodes(seg.node_ids)
        if summary:
            ev.append({"type": "text", "value": summary, "source": "graph"})
        for (s,p,o) in seg_triples[:max_triples]:
            ev.append({"type": "fact", "value": f"{s} | {p} | {o}", "source": "kg"})
            if len(ev) >= 1 + max_facts:
                break
        items.append({
            "id": seg.id,
            "beat": seg.beat,
            "question": seg.question or f"{topic} — {seg.beat}",
            "evidence": ev,
            "urls": [],
            "meta": {"graph": bundle.meta, "segment_nodes": list(seg_nodes)},
        })

    out = {
        "plan_id": f"graph_auto_{_slug(topic)}",
        "persona": persona,
        "length": "auto",
        "items": items,
        "graph_generation": genc,
        "topic": topic,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
