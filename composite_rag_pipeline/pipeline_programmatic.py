#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re, sys, types, warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Silence the LibreSSL/urllib3 warning spam
try:
    from urllib3.exceptions import NotOpenSSLWarning  # type: ignore
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

# ----------------------------- utils -----------------------------

def p(*parts: str) -> Path:
    return Path(*parts).resolve()

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def append_jsonl(path: Path, rec: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _slug_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")

def _to_text(x: Any) -> str:
    if isinstance(x, str): return x
    if isinstance(x, list):
        for y in x:
            if isinstance(y, str) and y.strip(): return y
        return " / ".join(str(y) for y in x if y is not None)
    if isinstance(x, dict):
        for k in ("title","label","name","value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip(): return v
        vals = [str(v) for v in x.values() if isinstance(v, (str,int,float))]
        return " / ".join(vals)
    return "" if x is None else str(x)

# ---------------------- loaders (self-contained) -----------------

def load_meta_from_path(meta_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read cq_metadata.json and normalize keys we need."""
    data = read_json(meta_path)
    meta = data.get("metadata") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for cid, rec in meta.items():
        r = dict(rec)
        bt = _to_text(r.get("beat_title") or r.get("Beat") or r.get("beat") or
                      r.get("beatLabel") or r.get("beat_name"))
        r["beat_title"] = bt
        r["beat_slug"] = _slug_text(bt)
        r["sparql"] = (r.get("sparql") or "").strip()
        r["question"] = r.get("question") or r.get("Question") or ""
        out[cid] = r
    return out

def extract_beats(narrative: Any, persona: str, length: str) -> List[Dict[str, Any]]:
    """Supports shapes:
       A) { "<persona>": { "<length>": [ {title|beat|name}, ... ] } }
       B) {"plans":[{"persona":...,"length":...,"beats":[...]}]}
    """
    beats: List[Dict[str, Any]] = []
    def _title_from(obj, i):
        if isinstance(obj, dict):
            return obj.get("title") or obj.get("beat") or obj.get("name") or f"Beat {i+1}"
        return str(obj) if obj is not None else f"Beat {i+1}"

    if isinstance(narrative, dict):
        node = (narrative.get(persona) or {}).get(length)
        if isinstance(node, list) and node:
            for i, b in enumerate(node):
                beats.append({"index": i, "title": _title_from(b, i)})
        if not beats and isinstance(narrative.get("plans"), list):
            for plan in narrative["plans"]:
                if plan.get("persona")==persona and plan.get("length")==length:
                    for i, b in enumerate(plan.get("beats") or []):
                        beats.append({"index": i, "title": _title_from(b, i)})
                    break
    return beats

# ---------------------------- planner ----------------------------

def _pool_for_beat(meta: Dict[str,Dict[str,Any]], beat_slug: str, require_sparql: bool) -> List[Tuple[str,Dict[str,Any]]]:
    pool = []
    for cid, rec in meta.items():
        if (rec.get("beat_slug") or "") != beat_slug:
            continue
        if require_sparql and not (rec.get("sparql") or "").strip():
            continue
        pool.append((cid, rec))
    return pool

def _any_pool(meta: Dict[str,Dict[str,Any]], require_sparql: bool) -> List[Tuple[str,Dict[str,Any]]]:
    pool = []
    for cid, rec in meta.items():
        if require_sparql and not (rec.get("sparql") or "").strip(): continue
        pool.append((cid, rec))
    return pool

def _intersect_ids(kg_pool: List[Tuple[str,Dict[str,Any]]], hy_pool: List[Tuple[str,Dict[str,Any]]]) -> List[str]:
    return sorted({cid for cid,_ in kg_pool} & {cid for cid,_ in hy_pool})

def _mk_item(cid: str, rec: Dict[str,Any], b_idx: int, b_title: str, mode: str) -> Dict[str,Any]:
    return {
        "id": cid,
        "mode": mode,
        "beat": {"index": b_idx, "title": b_title},
        "question": rec.get("question",""),
        "sparql": rec.get("sparql",""),
        "sparql_source": "meta",
    }

def build_plans_local(
    kg_meta: Dict[str,Dict[str,Any]],
    hy_meta: Dict[str,Dict[str,Any]],
    beats: List[Dict[str,Any]],
    items_per_beat: int,
    rng: random.Random,
    match_strategy: str = "intersect",
    require_sparql: bool = True,
    allow_backfill: bool = True,
    anybeat_fallback: bool = True,
) -> Tuple[Dict[str,Any], Dict[str,Any], List[str]]:
    warnings: List[str] = []
    items_kg: List[Dict[str,Any]] = []
    items_hy: List[Dict[str,Any]] = []

    for b in beats:
        i = b["index"]; title = b["title"]; bslug = _slug_text(title)

        kg_pool = _pool_for_beat(kg_meta, bslug, require_sparql)
        hy_pool = _pool_for_beat(hy_meta, bslug, require_sparql)

        if not kg_pool and anybeat_fallback:
            kg_pool = _any_pool(kg_meta, require_sparql)
            warnings.append(f"[beat {i}] KG pool empty for '{title}' → ANY-beat ({len(kg_pool)})")
        if not hy_pool and anybeat_fallback:
            hy_pool = _any_pool(hy_meta, require_sparql)
            warnings.append(f"[beat {i}] Hybrid pool empty for '{title}' → ANY-beat ({len(hy_pool)})")

        if match_strategy == "intersect":
            ids = _intersect_ids(kg_pool, hy_pool)
            if not ids:
                warnings.append(f"[beat {i}] intersect empty for '{title}' → sampling independently.")
                picked_kg = rng.sample(kg_pool, min(items_per_beat, len(kg_pool))) if kg_pool else []
                picked_hy = rng.sample(hy_pool, min(items_per_beat, len(hy_pool))) if hy_pool else []
            else:
                rng.shuffle(ids)
                ids_pick = ids[:items_per_beat]
                kg_by = {cid: rec for cid,rec in kg_pool}
                hy_by = {cid: rec for cid,rec in hy_pool}
                picked_kg = [(cid, kg_by[cid]) for cid in ids_pick if cid in kg_by]
                picked_hy = [(cid, hy_by[cid]) for cid in ids_pick if cid in hy_by]
        else:  # union
            picked_kg = rng.sample(kg_pool, min(items_per_beat, len(kg_pool))) if kg_pool else []
            picked_hy = rng.sample(hy_pool, min(items_per_beat, len(hy_pool))) if hy_pool else []

        if allow_backfill:
            if len(picked_kg) < items_per_beat:
                rest = [p for p in kg_pool if p not in picked_kg]
                more = rng.sample(rest, min(items_per_beat - len(picked_kg), len(rest)))
                picked_kg += more
            if len(picked_hy) < items_per_beat:
                rest = [p for p in hy_pool if p not in picked_hy]
                more = rng.sample(rest, min(items_per_beat - len(picked_hy), len(rest)))
                picked_hy += more

        for cid, rec in picked_kg:
            items_kg.append(_mk_item(cid, rec, i, title, "KG"))
        for cid, rec in picked_hy:
            items_hy.append(_mk_item(cid, rec, i, title, "Hybrid"))

    plan_kg = {
        "persona": None, "length": None, "mode": "KG",
        "beats": [{"index": b["index"], "title": b["title"], "items": items_per_beat} for b in beats],
        "items": items_kg,
    }
    plan_hy = {
        "persona": None, "length": None, "mode": "Hybrid",
        "beats": [{"index": b["index"], "title": b["title"], "items": items_per_beat} for b in beats],
        "items": items_hy,
    }
    return plan_kg, plan_hy, warnings

# --------------------------- pipeline core ----------------------------

def run_pipeline(
    persona: str = "Emma",
    length: str = "Medium",
    items_per_beat: int = 2,
    seed: int = 42,
    match_strategy: str = "intersect",
    llm_provider: str = "ollama",
    llm_model: str = "llama3.1-128k",
    # paths
    planner_dir: Path = p(ROOT, "planner"),
    retriever_dir: Path = p(ROOT, "retriever"),
    generator_dir: Path = p(ROOT, "generator"),
    kg_meta_path: Path = p(ROOT, "index/KG/cq_metadata.json"),
    hy_meta_path: Path = p(ROOT, "index/Hybrid/cq_metadata.json"),
    narrative_path: Path = p(ROOT, "data/narrative_plans.json"),
    rdf_path: Path = p(ROOT, "data/liveaid_instances_master.ttl"),
    params_path: Path = p(ROOT, "retriever_params.json"),
    # knobs
    per_item_sample: int = 5,
    timeout_s: int = 10,
    url_timeout_s: int = 5,
    max_urls_per_item: int = 5,
    content_max_chars: int = 240,
    max_url_snippets: int = 2,
    snippet_chars: int = 500,
) -> Tuple[Path, Path]:

    # import retriever & generator
    sys.path[:0] = [str(retriever_dir), str(generator_dir)]
    import retriever_local_rdflib as retriever  # type: ignore
    import generator_dual as gen  # type: ignore

    # Load inputs
    kg_meta = load_meta_from_path(kg_meta_path)
    hy_meta = load_meta_from_path(hy_meta_path)
    narrative = read_json(narrative_path)
    beats = extract_beats(narrative, persona, length)
    if not beats:
        # fallback to top beats from KG meta
        by_beat: Dict[str,int] = {}
        for r in kg_meta.values() or hy_meta.values():
            s = r.get("beat_slug") or ""
            if s: by_beat[s] = by_beat.get(s,0) + 1
        tops = [k for k,_ in sorted(by_beat.items(), key=lambda x:(-x[1], x[0]))[:6]]
        beats = [{"index": i, "title": next((r.get("beat_title","") for r in kg_meta.values() if r.get("beat_slug")==s),
                                            s.replace("-"," ").title())} for i,s in enumerate(tops)]

    # Build plans locally (no planner internals)
    rng = random.Random(seed)
    plan_kg, plan_hy, warns = build_plans_local(
        kg_meta=kg_meta, hy_meta=hy_meta, beats=beats,
        items_per_beat=items_per_beat, rng=rng,
        match_strategy=match_strategy, require_sparql=True,
        allow_backfill=True, anybeat_fallback=True,
    )
    plan_kg["persona"] = plan_hy["persona"] = persona
    plan_kg["length"] = plan_hy["length"] = length

    # Save plans
    out_plan_kg = p(planner_dir, "plan_KG.json"); write_json(out_plan_kg, plan_kg)
    out_plan_hy = p(planner_dir, "plan_Hybrid.json"); write_json(out_plan_hy, plan_hy)
    if warns:
        print("Planner warnings:")
        for w in warns: print(" -", w)

    # Retriever (KG)
    print(f"params_path :{str(params_path)}")
    params = read_json(params_path) if params_path.exists() else {}
    print(f"\nRunning Retriever (KG) with params: {params}")
    kg_enriched = retriever.run(
        plan=plan_kg,
        rdf_files=[str(rdf_path)],
        bindings=params,
        per_item_sample=per_item_sample,
        require_sparql=True,
        timeout_s=timeout_s,
        log_dir=str(p(retriever_dir, "run_trace", "logs")),
        errors_jsonl=str(p(retriever_dir, "run_trace", "retriever.jsonl")),
        include_stack=True,
        include_executed_query=True,
        strict_bindings=False,
        execute_on_unbound=False,
        fetch_url_content=False,
        url_timeout_s=url_timeout_s,
        max_urls_per_item=max_urls_per_item,
        content_max_bytes=250_000,
        content_max_chars=content_max_chars,
    )
    out_plan_evid_kg = p(retriever_dir, "plan_with_evidence_KG.json"); write_json(out_plan_evid_kg, kg_enriched)
    evid_kg = p(retriever_dir, "evidence_KG.jsonl")
    for it in kg_enriched.get("items", []):
        append_jsonl(evid_kg, {
            "mode": "KG",
            "cq_id": it.get("id"),
            "beat": it.get("beat", {}),
            "row_count": it.get("row_count", 0),
            "rows": it.get("rows", []),
            "executed_query": it.get("executed_query", ""),
            "url_candidates": it.get("url_candidates", []),
        })

    # Retriever (Hybrid) with URL content
    hy_enriched = retriever.run(
        plan=plan_hy,
        rdf_files=[str(rdf_path)],
        bindings=params,
        per_item_sample=per_item_sample,
        require_sparql=True,
        timeout_s=timeout_s,
        log_dir=str(p(retriever_dir, "run_trace", "logs")),
        errors_jsonl=str(p(retriever_dir, "run_trace", "retriever.jsonl")),
        include_stack=True,
        include_executed_query=True,
        strict_bindings=False,
        execute_on_unbound=False,
        fetch_url_content=True,
        url_timeout_s=url_timeout_s,
        max_urls_per_item=max_urls_per_item,
        content_max_bytes=250_000,
        content_max_chars=content_max_chars,
    )
    out_plan_evid_hy = p(retriever_dir, "plan_with_evidence_Hybrid.json"); write_json(out_plan_evid_hy, hy_enriched)
    evid_hy = p(retriever_dir, "evidence_Hybrid.jsonl")
    for it in hy_enriched.get("items", []):
        append_jsonl(evid_hy, {
            "mode": "Hybrid",
            "cq_id": it.get("id"),
            "beat": it.get("beat", {}),
            "row_count": it.get("row_count", 0),
            "rows": it.get("rows", []),
            "executed_query": it.get("executed_query", ""),
            "url_candidates": it.get("url_candidates", []),
            "url_info": it.get("url_info", []),
        })

    # Generator
    sys.path.insert(0, str(generator_dir))
    gen_module: types.ModuleType = sys.modules["generator_dual"]

    def call_generator(mode: str, plan_path: Path, plan_evid_path: Path,
                       meta_path: Path, story_out: Path, answers_out: Path):
        # Try a programmatic API
        if hasattr(gen_module, "generate"):
            plan = read_json(plan_path)
            plan_ev = read_json(plan_evid_path)
            story_md, answers = gen_module.generate(
                mode=mode,
                plan=plan,
                plan_with_evidence=plan_ev,
                meta_path=str(meta_path),
                params=params,
                llm_provider="ollama",
                llm_model="llama3.1-128k",
                use_url_content=(mode=="Hybrid"),
                max_url_snippets=max_url_snippets,
                snippet_chars=snippet_chars,
                compose="chunked",
                max_facts_per_beat=5,
                beat_sentences=3,
                max_rows=4,
                context_budget_chars=1000,
                include_citations=True,
            )
            story_out.write_text(story_md, encoding="utf-8")
            with answers_out.open("w", encoding="utf-8") as f:
                for rec in answers:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        # Fallback to main() with mocked argv
        if hasattr(gen_module, "main"):
            old_argv = sys.argv[:]
            argv = [
                "generator_dual.py",
                "--plan", str(plan_path),
                "--plan_with_evidence", str(plan_evid_path),
                ("--kg_meta" if mode == "KG" else "--hy_meta"), str(meta_path),
                "--params", str(p(ROOT, "generator_params.json")),
                "--llm_provider", "ollama",
                "--llm_model", "llama3.1-128k",
                "--out", str(answers_out.name),
                "--story_out", str(story_out.name),
                "--include_citations",
            ]
            if mode == "Hybrid":
                argv += ["--use_url_content", "--max_url_snippets", str(max_url_snippets), "--snippet_chars", str(snippet_chars)]
            try:
                sys.argv = argv
                gen_module.main()
            finally:
                sys.argv = old_argv
            return

        raise RuntimeError("generator_dual has no callable API (expected generate() or main()).")

    story_kg = p(generator_dir, "story_KG.md")
    story_hy = p(generator_dir, "story_Hybrid.md")
    ans_kg = p(generator_dir, "answers_KG.jsonl")
    ans_hy = p(generator_dir, "answers_Hybrid.jsonl")

    call_generator("KG", out_plan_kg, out_plan_evid_kg, kg_meta_path, story_kg, ans_kg)
    call_generator("Hybrid", out_plan_hy, out_plan_evid_hy, hy_meta_path, story_hy, ans_hy)

    print("\n✅ Stories written to:")
    print("  -", story_kg)
    print("  -", story_hy)
    return story_kg, story_hy

# ------------------------------ CLI ------------------------------

def cli():
    ap = argparse.ArgumentParser(description="Run KG + Hybrid pipeline in-process (no subprocess).")
    ap.add_argument("--persona", default="Emma")
    ap.add_argument("--length", default="Medium")
    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match_strategy", choices=["union","intersect"], default="intersect")
    ap.add_argument("--kg_meta", default=str(p(ROOT, "index/KG/cq_metadata.json")))
    ap.add_argument("--hy_meta", default=str(p(ROOT, "index/Hybrid/cq_metadata.json")))
    ap.add_argument("--narrative_plans", default=str(p(ROOT, "data/narrative_plans.json")))
    ap.add_argument("--rdf", default=str(p(ROOT, "data/liveaid_instances_master.ttl")))
    ap.add_argument("--params", default=str(p(ROOT, "retriever_params.json")))
    ap.add_argument("--per_item_sample", type=int, default=5)
    ap.add_argument("--timeout_s", type=int, default=10)
    ap.add_argument("--url_timeout_s", type=int, default=5)
    ap.add_argument("--max_urls_per_item", type=int, default=5)
    ap.add_argument("--content_max_chars", type=int, default=240)
    ap.add_argument("--max_url_snippets", type=int, default=2)
    ap.add_argument("--snippet_chars", type=int, default=500)
    args = ap.parse_args()

    run_pipeline(
        persona=args.persona,
        length=args.length,
        items_per_beat=args.items_per_beat,
        seed=args.seed,
        match_strategy=args.match_strategy,
        kg_meta_path=Path(args.kg_meta),
        hy_meta_path=Path(args.hy_meta),
        narrative_path=Path(args.narrative_plans),
        rdf_path=Path(args.rdf),
        params_path=Path(args.params),
        per_item_sample=args.per_item_sample,
        timeout_s=args.timeout_s,
        url_timeout_s=args.url_timeout_s,
        max_urls_per_item=args.max_urls_per_item,
        content_max_chars=args.content_max_chars,
        max_url_snippets=args.max_url_snippets,
        snippet_chars=args.snippet_chars,
    )

if __name__ == "__main__":
    cli()
