#!/usr/bin/env python3
# pipeline_programmatic.py
from __future__ import annotations
import argparse, copy, json, random, re, warnings, subprocess, sys, inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from typing import Optional


# Silence LibreSSL warning on some macOS setups
try:
    from urllib3.exceptions import NotOpenSSLWarning  # type: ignore
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# Local modules — ensure PYTHONPATH includes your repo root
from retriever.retriever_local_rdflib import run as retriever_run
from generator.generator_dual import generate as generator_generate

# =========================
# Defaults (configs & params)
# =========================

DEFAULT_RETRIEVER_CFG: Dict[str, Any] = {
    "shared": {
        "per_item_sample": 5,
        "timeout_s": 10.0,
        "require_sparql": True,

        # URL / content
        "max_urls_per_item": 5,
        "content_max_bytes": 250_000,
        "content_max_chars": 50000,
        "url_timeout_s": 5.0,

        # Chunking for large pages → small, LLM-friendly pieces
        "chunk_chars": 30000,
        "chunk_overlap": 50,
        "max_chunks_per_url": 15,
        "max_url_chunks_total_per_item": 20,
    },
    "kg": {
        "enrich_urls": False,
        "fetch_url_content": False,
        "chunk_url_content": False,
    },
    "hybrid": {
        "enrich_urls": True,
        "fetch_url_content": True,
        "chunk_url_content": True,
    },
}

DEFAULT_GENERATOR_CFG: Dict[str, Any] = {
    "llm_provider": "ollama",
    "llm_model": "llama3.1-128k",
    "ollama_num_ctx": None,

    # Citations / structure
    "include_citations": True,
    "citation_style": "cqid",
    "enforce_citation_each_sentence": True,

    # Context shaping
    "max_rows": 6,
    "max_facts_per_beat": 12,
    "beat_sentences": 4,
    "context_budget_chars": 1600,

    # Hybrid snippets
    "use_url_content_hybrid": True,
    "max_url_snippets": 3,
    "snippet_chars": 400,

    # Evaluation outputs
    "make_claims": True,
}

# Params for templating (NL vs SPARQL)
DEFAULT_GENERATOR_PARAMS = {
    "Event": "Live Aid 1985",
    "MusicGroup": "Queen",
    "SingleArtist": "Madonna",
    "BandMember": "Brian May",
    "Venue": "Wembley Stadium",
    "Venue2": "JFK Stadium",
}
DEFAULT_RETRIEVER_PARAMS = {
    "event": "ex:LiveAid1985",
    "musicgroup": "ex:Queen",
    "singleartist": "ex:Madonna",
    "bandmember": "ex:BrianMay",
    "venue": "ex:WembleyStadium",
    "venue2": "ex:JFKStadium",
}

# ==========
# Tiny utils
# ==========

def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def slug(s: Any) -> str:
    if isinstance(s, (list, dict)):
        s = json.dumps(s, ensure_ascii=False)
    s = (s or "").strip().lower()
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"[^a-z0-9 &]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "-")

def _normalize_beat_title(x: Any) -> str:
    """Return a *string* beat title with a safe fallback."""
    if isinstance(x, str) and x.strip():
        return x.strip()
    if isinstance(x, list):
        for v in x:
            if isinstance(v, str) and v.strip():
                return v.strip()
        return " / ".join(str(v) for v in x if v is not None) or "Unspecified"
    if isinstance(x, dict):
        for k in ("title", "label", "name", "value"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        vals = [str(v) for v in x.values() if isinstance(v, (str, int, float))]
        return " / ".join(vals) or "Unspecified"
    return "Unspecified"

# ======================
# Robust meta normalization (internal fallback planner)
# ======================

def _rows_from_meta(meta: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - {"cqs":[...]}, {"cqs":{id:rec}}, {id:rec}, or [rec,...]
    Returns list[dict] with ensured 'beat_title' and 'mode' (maps RetrievalMode → mode if needed).
    """
    def _as_rows(obj: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if isinstance(obj, list):
            for v in obj:
                if isinstance(v, dict):
                    rows.append(v)
                elif isinstance(v, str):
                    rows.append({"id": v})
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    rec = {"id": k}; rec.update(v); rows.append(rec)
                elif isinstance(v, str):
                    rows.append({"id": k, "question": v})
        return rows

    base = meta["cqs"] if isinstance(meta, dict) and "cqs" in meta else meta
    rows = _as_rows(base)

    for r in rows:
        # Map RetrievalMode -> mode if needed
        if "mode" not in r and "RetrievalMode" in r:
            r["mode"] = r["RetrievalMode"]

        # Ensure beat_title
        r["beat_title"] = _normalize_beat_title(r.get("beat_title") or r.get("beat") or r.get("beat_slug") or r.get("beatTitle"))

    return rows

def _filter_rows_by_mode(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Keep rows with missing mode, or mode==X, or list contains X (case-insensitive)."""
    m = (mode or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for r in rows:
        mv = r.get("mode")
        if mv is None:
            out.append(r); continue
        if isinstance(mv, str):
            if mv.strip().lower() == m:
                out.append(r)
        elif isinstance(mv, list):
            vals = [str(x).strip().lower() for x in mv if isinstance(x, str)]
            if m in vals:
                out.append(r)
    return out

def _group_meta_by_beat(meta_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in meta_rows:
        bt = _normalize_beat_title(r.get("beat_title", ""))
        by.setdefault(slug(bt), []).append(r)
    return by

def _random_pick(rows: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows = [r for r in rows if (r.get("sparql") or "").strip()]
    if not rows: return []
    if k >= len(rows): return rows[:]
    return rng.sample(rows, k)

def _synthesize_beats_from_meta(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    titles = [ _normalize_beat_title(r.get("beat_title","")) for r in rows ]
    seen, uniq = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return [{"index": i, "title": t} for i, t in enumerate(uniq or ["Unspecified"])]

def _internal_build_plan(
    mode: str, meta_path: Path, narrative_plans: Path,
    persona: str, length: str, items_per_beat: int, seed: int
) -> Dict[str, Any]:
    def _load_beats(narr: Path, persona: str, length: str) -> List[Dict[str, Any]]:
        spec = read_json(narr)
        if isinstance(spec, dict) and persona in spec:
            o = spec[persona]
            if isinstance(o, dict) and length in o and isinstance(o[length], list):
                return [{"index": i, "title": _normalize_beat_title(b.get("title") if isinstance(b, dict) else b)}
                        for i, b in enumerate(o[length])]
        if isinstance(spec, dict) and "beats" in spec and isinstance(spec["beats"], list):
            return [{"index": i, "title": _normalize_beat_title(b.get("title") if isinstance(b, dict) else b)}
                    for i, b in enumerate(spec["beats"])]
        if isinstance(spec, list):
            return [{"index": i, "title": _normalize_beat_title(b.get("title") if isinstance(b, dict) else b)}
                    for i, b in enumerate(spec)]
        return [{"index": 0, "title": "Unspecified"}]

    meta = read_json(meta_path)
    rows_all = _rows_from_meta(meta)
    rows = _filter_rows_by_mode(rows_all, mode)

    try:
        beats = _load_beats(narrative_plans, persona, length)
    except Exception:
        beats = _synthesize_beats_from_meta(rows if rows else rows_all)

    by_beat = _group_meta_by_beat(rows if rows else rows_all)
    rng = random.Random(seed)
    global_pool = [r for r in (rows if rows else rows_all) if (r.get("sparql") or "").strip()]

    items: List[Dict[str, Any]] = []
    for b in beats:
        title = b["title"]; bslug = slug(title)
        pool = by_beat.get(bslug, []) or global_pool
        picks = _random_pick(pool, items_per_beat, rng)
        for r in picks:
            items.append({
                "id": r.get("id") or r.get("cq_id") or "CQ-UNK",
                "question": r.get("question") or "",
                "beat": {"index": b["index"], "title": title},
                "sparql": r.get("sparql") or "",
            })

    if not items and global_pool:
        picks = _random_pick(global_pool, max(1, items_per_beat), rng)
        beats = [{"index": 0, "title": "Unspecified"}]
        for r in picks:
            items.append({
                "id": r.get("id") or r.get("cq_id") or "CQ-UNK",
                "question": r.get("question") or "",
                "beat": {"index": 0, "title": "Unspecified"},
                "sparql": r.get("sparql") or "",
            })

    return {"mode": mode, "persona": persona, "length": length, "beats": beats, "items": items}

# =========================
# External planner integration
# =========================

def _resolve_beats_from_narrative(narrative_plans: Path, persona: str, length: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    spec = read_json(narrative_plans)
    def _make_objs(seq):
        return [{"index": i, "title": _normalize_beat_title(b.get("title") if isinstance(b, dict) else b)}
                for i, b in enumerate(seq)]
    beat_objs: List[Dict[str, Any]] = []
    if isinstance(spec, dict) and persona in spec:
        o = spec[persona]
        if isinstance(o, dict) and length in o and isinstance(o[length], list):
            beat_objs = _make_objs(o[length])
    elif isinstance(spec, dict) and "beats" in spec and isinstance(spec["beats"], list):
        beat_objs = _make_objs(spec["beats"])
    elif isinstance(spec, list):
        beat_objs = _make_objs(spec)

    if not beat_objs:
        beat_objs = [{"index": 0, "title": "Unspecified"}]
    beat_titles = [bo["title"] or "Unspecified" for bo in beat_objs]
    beat_titles = ["Unspecified" if (t is None or str(t).strip() == "" or str(t).lower() == "none") else str(t) for t in beat_titles]
    return beat_objs, beat_titles

def _call_planner_programmatically_if_possible(
    planner_module, *, kg_meta: Path, hy_meta: Path, narrative_plans: Path,
    persona: str, length: str, items_per_beat: int, seed: int,
    match_strategy: str, out_kg: Path, out_hybrid: Path
) -> bool:
    """
    Try calling the planner module without spawning a new process.
    Pass `beats` as list[str] (titles), and `beat_plan` as list[dict] if requested.
    Return True iff both plan files were written.
    """
    candidates = [
        "build_plans", "build_plans_from_meta", "make_plans",
        "plan_dual_random", "create_plans"
    ]
    beat_plan_objs, beat_titles = _resolve_beats_from_narrative(narrative_plans, persona, length)
    rng = random.Random(seed)

    for fn in candidates:
        if not hasattr(planner_module, fn):
            continue
        func = getattr(planner_module, fn)
        if not callable(func):
            continue

        try:
            sig = inspect.signature(func)
            kwargs: Dict[str, Any] = {}
            for p in sig.parameters.values():
                name = p.name
                if name in {"kg_meta", "kg_meta_path"}:
                    kwargs[name] = str(kg_meta)
                elif name in {"hy_meta", "hy_meta_path", "hybrid_meta"}:
                    kwargs[name] = str(hy_meta)
                elif name in {"narrative_plans", "narrative_path"}:
                    kwargs[name] = str(narrative_plans)
                elif name in {"persona"}:
                    kwargs[name] = persona
                elif name in {"length"}:
                    kwargs[name] = length
                elif name in {"items_per_beat", "k_per_beat"}:
                    kwargs[name] = items_per_beat
                elif name in {"seed", "random_seed"}:
                    kwargs[name] = seed
                elif name in {"match_strategy", "strategy"}:
                    kwargs[name] = match_strategy
                elif name in {"out_kg", "out_kg_path"}:
                    kwargs[name] = str(out_kg)
                elif name in {"out_hybrid", "out_hy", "out_hy_path"}:
                    kwargs[name] = str(out_hybrid)
                elif name in {"beats"}:
                    # <-- Many planners expect beats as list[str]
                    kwargs[name] = beat_titles
                elif name in {"beat_plan", "beat_list"}:
                    # <-- If they want rich objects, give the dicts
                    kwargs[name] = beat_plan_objs
                elif name in {"rng", "random", "rand", "random_state"}:
                    kwargs[name] = rng

            func(**kwargs)

            if out_kg.exists() and out_hybrid.exists():
                print(f"[planner] used programmatic call '{fn}' successfully")
                return True

        except Exception as e:
            print(f"[planner] programmatic call '{fn}' failed: {e}")

    return False

def _call_planner_subprocess(
    planner_path: Path, *, kg_meta: Path, hy_meta: Path, narrative_plans: Path,
    persona: str, length: str, items_per_beat: int, seed: int,
    match_strategy: str, out_kg: Path, out_hybrid: Path
) -> None:
    cmd = [
        sys.executable, str(planner_path),
        "--kg_meta", str(kg_meta),
        "--hy_meta", str(hy_meta),
        "--narrative_plans", str(narrative_plans),
        "--persona", persona, "--length", length,
        "--items_per_beat", str(items_per_beat),
        "--seed", str(seed),
        "--match_strategy", match_strategy,
        "--out_kg", str(out_kg),
        "--out_hybrid", str(out_hybrid),  # <-- fixed
    ]
    print(f"[planner] invoking subprocess: {' '.join(cmd)}")
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"planner_dual_random failed with code {cp.returncode}")

def _make_plans_via_external_or_internal(
    *, use_external: bool, planner_path: Path, match_strategy: str,
    kg_meta: Path, hy_meta: Path, narrative_plans: Path,
    persona: str, length: str, items_per_beat: int, seed: int,
    out_kg_path: Path, out_hy_path: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if use_external:
        try:
            sys.path.insert(0, str(planner_path.parent))
            mod_name = planner_path.stem
            planner_module = __import__(mod_name)
            used_prog = _call_planner_programmatically_if_possible(
                planner_module,
                kg_meta=kg_meta, hy_meta=hy_meta, narrative_plans=narrative_plans,
                persona=persona, length=length, items_per_beat=items_per_beat,
                seed=seed, match_strategy=match_strategy,
                out_kg=out_kg_path, out_hybrid=out_hy_path,
            )
            if not used_prog:
                _call_planner_subprocess(
                    planner_path,
                    kg_meta=kg_meta, hy_meta=hy_meta, narrative_plans=narrative_plans,
                    persona=persona, length=length, items_per_beat=items_per_beat,
                    seed=seed, match_strategy=match_strategy,
                    out_kg=out_kg_path, out_hybrid=out_hy_path,
                )
        except Exception as e:
            print(f"[planner] external planner failed ({e}); using internal fallback")

    if not out_kg_path.exists() or not out_hy_path.exists():
        print("[planner] building plans internally as fallback")
        plan_kg = _internal_build_plan("KG", kg_meta, narrative_plans, persona, length, items_per_beat, seed)
        plan_hy = _internal_build_plan("Hybrid", hy_meta, narrative_plans, persona, length, items_per_beat, seed+7)
        write_json(out_kg_path, plan_kg)
        write_json(out_hy_path, plan_hy)

    return read_json(out_kg_path), read_json(out_hy_path)

# ==========
# Pipeline
# ==========

def run_pipeline(
    *, kg_meta: Path, hy_meta: Path, narrative_plans: Path, rdf_files: List[Path],
    persona: str, length: str, items_per_beat: int, seed: int,
    generator_params: Dict[str, Any], retriever_params: Dict[str, Any],
    out_root: Path, retriever_cfg: Dict[str, Any], generator_cfg: Dict[str, Any],
    use_external_planner: bool, planner_path: Path, planner_match_strategy: str,
    run_root: Path = Path("runs"),
    run_tag: Optional[str] = None,
    persist_params: bool = False,
):
    planner_dir   = out_root / "planner"
    retriever_dir = out_root / "retriever"
    generator_dir = out_root / "generator"
    for d in (planner_dir, retriever_dir, generator_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Plans (external → internal fallback)
    plan_kg_path = planner_dir / "plan_KG.json"
    plan_hy_path = planner_dir / "plan_Hybrid.json"
    plan_kg, plan_hy = _make_plans_via_external_or_internal(
        use_external=use_external_planner,
        planner_path=planner_path,
        match_strategy=planner_match_strategy,
        kg_meta=kg_meta, hy_meta=hy_meta, narrative_plans=narrative_plans,
        persona=persona, length=length, items_per_beat=items_per_beat, seed=seed,
        out_kg_path=plan_kg_path, out_hy_path=plan_hy_path,
    )
    run_dirs = _make_run_dirs(run_root, persona, length, seed, run_tag)
    planner_dir = run_dirs["planner"]
    retriever_dir = run_dirs["retriever"]
    generator_dir = run_dirs["generator"]
    logs_dir = run_dirs["logs"]
    params_dir = run_dirs["params"]
    print(f"✓ planner → {plan_kg_path} , {plan_hy_path}")

    if persist_params:
        try:
            write_json(params_dir / "retriever_params_used.json", retriever_params)
            write_json(params_dir / "generator_params_used.json", generator_params)
        except Exception:
            pass




    # 2) Retriever
    rc, shared, kgc, hyc = retriever_cfg, retriever_cfg.get("shared", {}), retriever_cfg.get("kg", {}), retriever_cfg.get("hybrid", {})

    # KG
    retr_kg_out_path = retriever_dir / "plan_with_evidence_KG.json"
    out_kg = retriever_run(
        plan=plan_kg,
        rdf_files=[str(p) for p in rdf_files],
        bindings=retriever_params,
        per_item_sample=int(shared.get("per_item_sample", 5)),
        require_sparql=bool(shared.get("require_sparql", True)),
        timeout_s=float(shared.get("timeout_s", 10.0)),
        log_dir=str(retriever_dir / "logs"),
        errors_jsonl=str(retriever_dir / "retriever.jsonl"),
        include_stack=True,
        include_executed_query=True,
        strict_bindings=False,
        execute_on_unbound=False,

        # enrichment (KG)
        enrich_urls=bool(kgc.get("enrich_urls", False)),
        fetch_url_content=bool(kgc.get("fetch_url_content", False)),
        url_timeout_s=float(shared.get("url_timeout_s", 5.0)),
        max_urls_per_item=int(shared.get("max_urls_per_item", 5)),
        content_max_bytes=int(shared.get("content_max_bytes", 250_000)),
        content_max_chars=int(shared.get("content_max_chars", 5000)),

        # chunking (KG)
        chunk_url_content=bool(kgc.get("chunk_url_content", False)),
        chunk_chars=int(shared.get("chunk_chars", 500)),
        chunk_overlap=int(shared.get("chunk_overlap", 50)),
        max_chunks_per_url=int(shared.get("max_chunks_per_url", 8)),
        max_url_chunks_total_per_item=int(shared.get("max_url_chunks_total_per_item", 20)),
    )
    write_json(retr_kg_out_path, out_kg)
    print(f"✓ retriever KG → {retr_kg_out_path}")

    # Hybrid
    retr_hy_out_path = retriever_dir / "plan_with_evidence_Hybrid.json"
    content_max_chars_hy = max(
        int(shared.get("content_max_chars", 5000)),
        int(shared.get("chunk_chars", 500)) * max(1, int(shared.get("max_chunks_per_url", 8))),
    )
    out_hy = retriever_run(
        plan=plan_hy,
        rdf_files=[str(p) for p in rdf_files],
        bindings=retriever_params,
        per_item_sample=int(shared.get("per_item_sample", 5)),
        require_sparql=bool(shared.get("require_sparql", True)),
        timeout_s=float(shared.get("timeout_s", 10.0)),
        log_dir=str(retriever_dir / "logs"),
        errors_jsonl=str(retriever_dir / "retriever.jsonl"),
        include_stack=True,
        include_executed_query=True,
        strict_bindings=False,
        execute_on_unbound=False,

        # enrichment (Hybrid)
        enrich_urls=bool(hyc.get("enrich_urls", True)),
        fetch_url_content=bool(hyc.get("fetch_url_content", True)),
        url_timeout_s=float(shared.get("url_timeout_s", 5.0)),
        max_urls_per_item=int(shared.get("max_urls_per_item", 5)),
        content_max_bytes=int(shared.get("content_max_bytes", 250_000)),
        content_max_chars=content_max_chars_hy,

        # chunking (Hybrid)
        chunk_url_content=bool(hyc.get("chunk_url_content", True)),
        chunk_chars=int(shared.get("chunk_chars", 500)),
        chunk_overlap=int(shared.get("chunk_overlap", 50)),
        max_chunks_per_url=int(shared.get("max_chunks_per_url", 8)),
        max_url_chunks_total_per_item=int(shared.get("max_url_chunks_total_per_item", 20)),
    )
    write_json(retr_hy_out_path, out_hy)
    print(f"✓ retriever Hybrid → {retr_hy_out_path}")

    # 3) Generator
    gc = generator_cfg

    # KG
    story_kg         = generator_dir / "story_KG.md"
    story_kg_clean   = generator_dir / "story_KG_clean.md"
    answers_kg_path  = generator_dir / "answers_KG.jsonl"
    claims_kg_path   = generator_dir / "claims_KG.jsonl" if gc.get("make_claims", True) else None
    print(f"out_kg :{out_kg}")
    story_md_kg, answers_kg = generator_generate(
        mode="KG",
        plan=plan_kg,
        plan_with_evidence=out_kg,
        meta_path=str(kg_meta),
        params=DEFAULT_GENERATOR_PARAMS,  # (can be overridden by your generator)
        llm_provider=gc.get("llm_provider", "ollama"),
        llm_model=gc.get("llm_model", "llama3.1-128k"),
        ollama_num_ctx=gc.get("ollama_num_ctx"),
        use_url_content=False,
        max_url_snippets=int(gc.get("max_url_snippets", 3)),
        snippet_chars=int(gc.get("snippet_chars", 40000)),
        include_citations=bool(gc.get("include_citations", True)),
        max_rows=int(gc.get("max_rows", 6)),
        max_facts_per_beat=int(gc.get("max_facts_per_beat", 12)),
        beat_sentences=int(gc.get("beat_sentences", 4)),
        context_budget_chars=int(gc.get("context_budget_chars", 50000)),
        enforce_citation_each_sentence=bool(gc.get("enforce_citation_each_sentence", True)),
        citation_style=gc.get("citation_style", "cqid"),
        claims_out=str(claims_kg_path) if claims_kg_path else None,
        story_clean_out=str(story_kg_clean),
    )
    story_kg.write_text(story_md_kg, encoding="utf-8")
    write_jsonl(answers_kg_path, answers_kg)
    if claims_kg_path:
        print(f"✓ claims KG → {claims_kg_path}")
    print(f"✓ generator KG → {story_kg} (+ clean {story_kg_clean})")

    # Hybrid
    story_hy         = generator_dir / "story_Hybrid.md"
    story_hy_clean   = generator_dir / "story_Hybrid_clean.md"
    answers_hy_path  = generator_dir / "answers_Hybrid.jsonl"
    claims_hy_path   = generator_dir / "claims_Hybrid.jsonl" if gc.get("make_claims", True) else None

    print(f"out_hy :{out_hy}")
    story_md_hy, answers_hy = generator_generate(
        mode="Hybrid",
        plan=plan_hy,
        plan_with_evidence=out_hy,
        meta_path=str(hy_meta),
        params=DEFAULT_GENERATOR_PARAMS,
        llm_provider=gc.get("llm_provider", "ollama"),
        llm_model=gc.get("llm_model", "llama3.1-128k"),
        ollama_num_ctx=gc.get("ollama_num_ctx"),
        use_url_content=bool(gc.get("use_url_content_hybrid", True)),
        max_url_snippets=int(gc.get("max_url_snippets", 3)),
        snippet_chars=int(gc.get("snippet_chars", 400)),
        include_citations=bool(gc.get("include_citations", True)),
        max_rows=int(gc.get("max_rows", 6)),
        max_facts_per_beat=int(gc.get("max_facts_per_beat", 12)),
        beat_sentences=int(gc.get("beat_sentences", 4)),
        context_budget_chars=int(gc.get("context_budget_chars", 1600)),
        enforce_citation_each_sentence=bool(gc.get("enforce_citation_each_sentence", True)),
        citation_style=gc.get("citation_style", "cqid"),
        claims_out=str(claims_hy_path) if claims_hy_path else None,
        story_clean_out=str(story_hy_clean),
    )
    story_hy.write_text(story_md_hy, encoding="utf-8")
    write_jsonl(answers_hy_path, answers_hy)
    if claims_hy_path:
        print(f"✓ claims Hybrid → {claims_hy_path}")
    print(f"✓ generator Hybrid → {story_hy} (+ clean {story_hy_clean})")
    print("\nAll done ✅")

# === CLI ===

def _load_params(defaults: Dict[str, Any], file_path: Optional[Path], json_str: Optional[str]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if file_path:
        merged = deep_update(merged, read_json(file_path))
    if json_str:
        merged = deep_update(merged, json.loads(json_str))
    return merged

# --- NEW: run-folder helper ---
def _make_run_dirs(run_root: Path, persona: str, length: str, seed: int, tag: Optional[str]) -> Dict[str, Path]:
    """
    Creates a timestamped run folder:
      <run_root>/<YYYYmmdd_HHMMSS>__<persona>-<length>__seed<seed>[__tag]/
        planner/
        retriever/logs/
        generator/
        params/
    Returns a dict of Paths.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = lambda s: re.sub(r"[^a-z0-9\-]+", "-", (s or "").strip().lower())
    base = f"{ts}__{safe(persona)}-{safe(length)}__seed{seed}"
    if tag:
        base = f"{base}__{safe(tag)}"
    run_dir = (run_root / base).resolve()
    dirs = {
        "run": run_dir,
        "planner": run_dir / "planner",
        "retriever": run_dir / "retriever",
        "generator": run_dir / "generator",
        "logs": run_dir / "retriever" / "logs",
        "params": run_dir / "params",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def cli():
    Boolean = argparse.BooleanOptionalAction
    ap = argparse.ArgumentParser(description="Composite pipeline calling planner_dual_random (with internal fallback)")

    ap.add_argument("--kg_meta", type=Path, required=True)
    ap.add_argument("--hy_meta", type=Path, required=True)
    ap.add_argument("--narrative_plans", type=Path, required=True)
    ap.add_argument("--rdf", type=Path, nargs="+", required=True)

    ap.add_argument("--persona", default="Emma")
    ap.add_argument("--length", default="Medium")
    ap.add_argument("--items_per_beat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", type=Path, default=Path("."))

    # Params (separate)
    ap.add_argument("--generator_params", type=Path)
    ap.add_argument("--generator_params_json", default=None)
    ap.add_argument("--retriever_params", type=Path)
    ap.add_argument("--retriever_params_json", default=None)

    # Config files to merge into defaults
    ap.add_argument("--retriever_cfg", type=Path)
    ap.add_argument("--generator_cfg", type=Path)

    # Optional overrides (retriever shared)
    ap.add_argument("--require_sparql", action=Boolean, default=None)
    ap.add_argument("--per_item_sample", type=int, default=None)
    ap.add_argument("--timeout_s", type=float, default=None)
    ap.add_argument("--max_urls_per_item", type=int, default=None)
    ap.add_argument("--content_max_bytes", type=int, default=None)
    ap.add_argument("--content_max_chars", type=int, default=None)
    ap.add_argument("--url_timeout_s", type=float, default=None)
    ap.add_argument("--chunk_chars", type=int, default=None)
    ap.add_argument("--chunk_overlap", type=int, default=None)
    ap.add_argument("--max_chunks_per_url", type=int, default=None)
    ap.add_argument("--max_url_chunks_total_per_item", type=int, default=None)

    # Optional overrides (per-mode)
    ap.add_argument("--kg_enrich", action=Boolean, default=None)
    ap.add_argument("--kg_fetch", action=Boolean, default=None)
    ap.add_argument("--kg_chunk", action=Boolean, default=None)
    ap.add_argument("--hy_enrich", action=Boolean, default=None)
    ap.add_argument("--hy_fetch", action=Boolean, default=None)
    ap.add_argument("--hy_chunk", action=Boolean, default=None)

    # Generator overrides
    ap.add_argument("--llm_provider", default=None, choices=["ollama","gemini"])
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--ollama_num_ctx", type=int, default=None)
    ap.add_argument("--include_citations", action=Boolean, default=None)
    ap.add_argument("--citation_style", choices=["numeric","cqid"], default=None)
    ap.add_argument("--enforce_citation_each_sentence", action=Boolean, default=None)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--max_facts_per_beat", type=int, default=None)
    ap.add_argument("--beat_sentences", type=int, default=None)
    ap.add_argument("--context_budget_chars", type=int, default=None)
    ap.add_argument("--use_url_content_hybrid", action=Boolean, default=None)
    ap.add_argument("--max_url_snippets", type=int, default=None)
    ap.add_argument("--snippet_chars", type=int, default=None)
    ap.add_argument("--claims", action=Boolean, default=None)

    # Planner integration
    ap.add_argument("--use_external_planner", action=Boolean, default=True)
    ap.add_argument("--planner_path", type=Path, default=Path("planner/planner_dual_random.py"))
    ap.add_argument("--planner_match_strategy", default="intersect")

    ap.add_argument("--run_root", default="runs", help="Folder where timestamped run directories are created.")
    ap.add_argument("--run_tag", default=None, help="Optional label appended to the run folder name.")
    ap.add_argument("--persist_params", action="store_true", help="Write the resolved params into run/params/.")

    args = ap.parse_args()

    # Merge retriever config
    rcfg = deep_update(copy.deepcopy(DEFAULT_RETRIEVER_CFG),
                       read_json(args.retriever_cfg) if args.retriever_cfg else {})
    if args.require_sparql is not None: rcfg["shared"]["require_sparql"] = bool(args.require_sparql)
    if args.kg_enrich   is not None:    rcfg["kg"]["enrich_urls"] = bool(args.kg_enrich)
    if args.kg_fetch    is not None:    rcfg["kg"]["fetch_url_content"] = bool(args.kg_fetch)
    if args.kg_chunk    is not None:    rcfg["kg"]["chunk_url_content"] = bool(args.kg_chunk)
    if args.hy_enrich   is not None:    rcfg["hybrid"]["enrich_urls"] = bool(args.hy_enrich)
    if args.hy_fetch    is not None:    rcfg["hybrid"]["fetch_url_content"] = bool(args.hy_fetch)
    if args.hy_chunk    is not None:    rcfg["hybrid"]["chunk_url_content"] = bool(args.hy_chunk)
    for k in ("per_item_sample","timeout_s","max_urls_per_item","content_max_bytes",
              "content_max_chars","url_timeout_s","chunk_chars","chunk_overlap",
              "max_chunks_per_url","max_url_chunks_total_per_item"):
        v = getattr(args, k)
        if v is not None:
            rcfg["shared"][k] = v

    # Merge generator config
    gcfg = deep_update(copy.deepcopy(DEFAULT_GENERATOR_CFG),
                       read_json(args.generator_cfg) if args.generator_cfg else {})
    if args.llm_provider is not None: gcfg["llm_provider"] = args.llm_provider
    if args.llm_model    is not None: gcfg["llm_model"] = args.llm_model
    if args.ollama_num_ctx is not None: gcfg["ollama_num_ctx"] = args.ollama_num_ctx
    if args.include_citations is not None: gcfg["include_citations"] = bool(args.include_citations)
    if args.citation_style is not None:    gcfg["citation_style"] = args.citation_style
    if args.enforce_citation_each_sentence is not None:
        gcfg["enforce_citation_each_sentence"] = bool(args.enforce_citation_each_sentence)
    if args.max_rows is not None:             gcfg["max_rows"] = args.max_rows
    if args.max_facts_per_beat is not None:   gcfg["max_facts_per_beat"] = args.max_facts_per_beat
    if args.beat_sentences is not None:       gcfg["beat_sentences"] = args.beat_sentences
    if args.context_budget_chars is not None: gcfg["context_budget_chars"] = args.context_budget_chars
    if args.use_url_content_hybrid is not None:
        gcfg["use_url_content_hybrid"] = bool(args.use_url_content_hybrid)
    if args.max_url_snippets is not None:     gcfg["max_url_snippets"] = args.max_url_snippets
    if args.snippet_chars is not None:        gcfg["snippet_chars"] = args.snippet_chars
    if args.claims is not None:               gcfg["make_claims"] = bool(args.claims)

    # Load params (defaults → file → inline JSON)
    def _load_params(defaults: Dict[str, Any], file_path: Optional[Path], json_str: Optional[str]) -> Dict[str, Any]:
        merged = copy.deepcopy(defaults)
        if file_path: merged = deep_update(merged, read_json(file_path))
        if json_str:  merged = deep_update(merged, json.loads(json_str))
        return merged
    generator_params = _load_params(DEFAULT_GENERATOR_PARAMS, args.generator_params, args.generator_params_json)
    retriever_params = _load_params(DEFAULT_RETRIEVER_PARAMS, args.retriever_params, args.retriever_params_json)

    run_pipeline(
        kg_meta=args.kg_meta, hy_meta=args.hy_meta, narrative_plans=args.narrative_plans,
        rdf_files=args.rdf, persona=args.persona, length=args.length,
        items_per_beat=args.items_per_beat, seed=args.seed,
        generator_params=generator_params, retriever_params=retriever_params,
        out_root=args.out_root, retriever_cfg=rcfg, generator_cfg=gcfg,
        use_external_planner=bool(args.use_external_planner),
        planner_path=args.planner_path,
        planner_match_strategy=args.planner_match_strategy,
        run_root=Path(args.run_root),
        run_tag=args.run_tag,
        persist_params=args.persist_params,
    )

if __name__ == "__main__":
    cli()
