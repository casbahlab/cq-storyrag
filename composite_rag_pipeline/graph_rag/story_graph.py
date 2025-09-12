from typing import Dict, Any
from graph_retriever import GraphRetriever, GraphRetrieverConfig
from .prompting import render_story_from_graph

def run_graph_story(plan: Dict[str, Any], cq: Dict[str, Any], persona: Dict[str, Any], kg):
    cfg = plan["retrieval"]["graph"]
    retriever = GraphRetriever(GraphRetrieverConfig(**cfg), kg)
    bundle = retriever.retrieve(plan, cq, persona)

    gen_in = {
        "cq_text": cq.get("text") or cq.get("question") or "",
        "persona": persona,
        "graph_summary": "\n".join(bundle.texts)[: plan["generation"]["graph"]["max_context_chars"]],
        "triples": bundle.triples[:350],
        "facts": bundle.facts[:200],
        "provenance": bundle.provenance,
        "meta": bundle.meta,
    }
    story = render_story_from_graph(gen_in)
    return story, bundle
