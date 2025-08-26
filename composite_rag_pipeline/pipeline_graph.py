#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-only RAG pipeline (length-aware + dedupe + strict enforcement)

Writes per-run artifacts:
  retriever/plan_with_evidence_Graph.json
  generator/answers_Graph.jsonl
  generator/story_Graph.md
  generator/story_Graph_clean.md
  params/graph_config_resolved.yaml
  params/length_report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional YAML (for persisting resolved graph config)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# Try your real Graph helpers. If missing, we fallback to tiny local stubs.
try:
    from graph_rag.planner_graph import build_graph_plan as ext_build_graph_plan, write_plan as ext_write_plan  # type: ignore
except Exception:
    ext_build_graph_plan = None
    ext_write_plan = None

try:
    from graph_rag.generator_graph import generate_graph_story as ext_generate_graph_story  # type: ignore
except Exception:
    ext_generate_graph_story = None

try:
    from graph_rag.config import load_graph_config as ext_load_graph_config  # type: ignore
except Exception:
    ext_load_graph_config = None


# ---------------------- fs helpers ----------------------

def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _ensure_run_dirs(run_root: Path) -> Tuple[Path, Path, Path]:
    retriever_dir = _mkdir(run_root / "retriever")
    generator_dir = _mkdir(run_root / "generator")
    params_dir    = _mkdir(run_root / "params")
    return retriever_dir, generator_dir, params_dir

def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def _json_dump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------- normalization ----------------------

_EX_BASE = "http://wembrewind.live/ex#"
_WS = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r'(?<=[.?!])\s+')

def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return _WS.sub(" ", s.strip().lower())

def _expand_ex(curie: str) -> str:
    return (_EX_BASE + curie.split(":", 1)[1]) if isinstance(curie, str) and curie.startswith("ex:") else curie

def _norm_length(s: Optional[str]) -> str:
    if not s:
        return "Medium"
    m = s.strip().lower()
    return {"small": "Short", "short": "Short", "medium": "Medium", "long": "Long"}.get(m, "Medium")


# ---------------------- length profile ----------------------

def _graph_length_profile(length: str) -> Dict[str, int]:
    L = _norm_length(length)
    profiles = {
        "Short":  {"beats_limit": 4, "beat_sentences": 3},
        "Medium": {"beats_limit": 6, "beat_sentences": 4},
        "Long":   {"beats_limit": 8, "beat_sentences": 5},
    }
    return profiles[L]

def _apply_length_limit(plan_obj: Dict[str, Any], beats_limit: int) -> Dict[str, Any]:
    plan = copy.deepcopy(plan_obj) if isinstance(plan_obj, dict) else {}
    for key in ("beats", "sections", "outline"):
        if isinstance(plan.get(key), list):
            plan[key] = plan[key][:max(1, int(beats_limit))]
            break
    return plan


# ---------------------- plan/evidence de-dup ----------------------

def _dedupe_plan_evidence(plan_obj: Dict[str, Any], per_section_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Remove duplicate evidence items across beats (and within each beat).
    Works whether you store evidence under 'evidence' (list[dict]) or 'rows' (list[dict]).
    Dedup key = normalized 'value' or 'text'.
    """
    plan = copy.deepcopy(plan_obj) if isinstance(plan_obj, dict) else {}
    seen_global: set[str] = set()

    def uniq(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen_local: set[str] = set()
        for it in items:
            key = _norm_text(it.get("value") or it.get("text") or "")
            if not key:
                continue
            if key in seen_local or key in seen_global:
                continue
            seen_local.add(key)
            seen_global.add(key)
            out.append(it)
            if per_section_limit and len(out) >= per_section_limit:
                break
        return out

    for key in ("beats", "sections", "outline"):
        seq = plan.get(key)
        if isinstance(seq, list):
            for beat in seq:
                if isinstance(beat.get("evidence"), list):
                    beat["evidence"] = uniq(beat["evidence"])
                if isinstance(beat.get("rows"), list):
                    beat["rows"] = uniq(beat["rows"])
            break
    return plan


# ---------------------- answers JSONL dedupe ----------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

def _write_jsonl_overwrite(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dedupe_answers_jsonl_inplace(path: Path) -> None:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = _norm_text(r.get("text", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    _write_jsonl_overwrite(path, out)



# ---- paste near top imports ----
import re
from collections import Counter

# ---- Title normalization & beat de-dup ----
import re, copy
EX_BASE = "http://wembrewind.live/ex#"

def _norm_title_token(s: str) -> str:
    if not isinstance(s, str): return ""
    if s.startswith(EX_BASE): s = s.split("#", 1)[1]
    s = s.replace("_", " ")
    s = re.sub(r"\b[0-9a-f]{6,}\b", "", s)  # drop hash-like tails
    return re.sub(r"\s{2,}", " ", s).strip()

def _make_title(entities):
    seen=set(); out=[]
    for e in entities or []:
        t=_norm_title_token(str(e))
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return " / ".join(out[:3]) or "Section"

def _beat_key(entities, relations):
    ekey = tuple(sorted(str(e) for e in set(entities or [])))
    rkey = tuple(sorted(str(r) for r in set(relations or [])))
    return (ekey, rkey)

def postprocess_plan(plan: dict) -> dict:
    beats = plan.get("beats") or plan.get("sections") or []
    kept, seen = [], set()
    for b in beats:
        ents = b.get("entities") or b.get("seed_entities") or []
        rels = b.get("relations") or []
        key = _beat_key(ents, rels)
        if key in seen:
            continue
        seen.add(key)
        b["title"] = _norm_title_token(b.get("title") or _make_title(ents))
        kept.append(b)

    if "beats" in plan:   plan["beats"] = kept
    elif "sections" in plan: plan["sections"] = kept
    else: plan["beats"] = kept
    return plan




# ---------------------- post-gen strict length ----------------------

def _split_sents(t: str) -> List[str]:
    if not t:
        return []
    return [x.strip() for x in _SENT_SPLIT_RE.split(t.strip()) if x.strip()]

def _truncate_sents(t: str, n: int) -> str:
    s = _split_sents(t)
    return " ".join(s[:max(1, int(n))]).strip()

def _simple_clean(text: str) -> str:
    lines_out: List[str] = []
    for raw in (text or "").splitlines():
        ln = raw.rstrip()
        if not ln:
            lines_out.append("")
            continue
        if ln.lstrip().startswith(("#", "- ", "* ", "+ ", "```", "~~~")):
            continue
        ln = re.sub(r"<https?://[^>]+>", "", ln)
        ln = re.sub(r"https?://\S+", "", ln)
        ln = re.sub(r"\b[a-z]{3,}:[A-Za-z0-9/_#\.\-]+", "", ln)
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if ln:
            lines_out.append(ln)
    out, buf = [], []
    for ln in lines_out:
        if ln == "":
            if buf:
                out.append(" ".join(buf))
                buf = []
        else:
            buf.append(ln)
    if buf:
        out.append(" ".join(buf))
    return ("\n\n".join(out)).strip() + ("\n" if out else "")

def enforce_length_targets(
    answers_jsonl: Path,
    story_md: Path,
    story_clean_md: Path,
    beats_limit: int,
    beat_sentences: int,
    report_path: Optional[Path] = None,
) -> None:
    rows = _read_jsonl(answers_jsonl)
    # Cap sections
    rows = rows[:max(1, int(beats_limit))]
    # Cap sentences per section
    for r in rows:
        r["text"] = _truncate_sents(r.get("text", ""), beat_sentences)
    # Rewrite answers
    _write_jsonl_overwrite(answers_jsonl, rows)
    # Rebuild story files
    story_text = "\n\n".join([r.get("text", "").strip() for r in rows if r.get("text")]).strip()
    _write_text(story_md, story_text + ("\n" if story_text else ""))
    _write_text(story_clean_md, _simple_clean(story_text))
    # Report
    if report_path:
        per = [len(_split_sents(r.get("text", ""))) for r in rows]
        rep = {
            "target": {
                "beats_limit": int(beats_limit),
                "beat_sentences": int(beat_sentences),
                "target_total_sentences": int(beats_limit) * int(beat_sentences),
            },
            "actual": {
                "sections": len(rows),
                "total_sentences": sum(per),
                "per_section_sentences": per,
            },
            "note": "Strict post-generation enforcement applied.",
        }
        _json_dump(report_path, rep)


# ---------------------- seeds/topic derivation ----------------------

def _parse_json_arg(maybe: Optional[str]) -> Dict[str, Any]:
    if not maybe:
        return {}
    try:
        return json.loads(maybe)
    except Exception:
        return {}

def _derive_topic(args) -> str:
    gp = _parse_json_arg(getattr(args, "generator_params_json", None)) or \
         _parse_json_arg(getattr(args, "generator_params", None))
    event = gp.get("Event") or gp.get("event") or "Live Aid 1985"
    group = gp.get("MusicGroup") or gp.get("musicgroup") or "Queen"
    venue = gp.get("Venue") or gp.get("venue") or "Wembley Stadium"
    return f"{group} at {event} — {venue}"

def _derive_seeds(args) -> Dict[str, List[str]]:
    rp = _parse_json_arg(getattr(args, "retriever_params_json", None)) or \
         _parse_json_arg(getattr(args, "retriever_params", None))
    ents: List[str] = []
    for k in ("event", "musicgroup", "singleartist", "bandmember", "venue", "venue2"):
        v = rp.get(k)
        if isinstance(v, str):
            ents.append(_expand_ex(v))
    if not ents:
        ents = [_EX_BASE + "LiveAid1985"]
    gp = _parse_json_arg(getattr(args, "generator_params_json", None)) or \
         _parse_json_arg(getattr(args, "generator_params", None))
    labels = [x for x in [gp.get("Event"), gp.get("MusicGroup"), gp.get("Venue")] if isinstance(x, str)]
    if not labels:
        labels = ["Live Aid 1985"]
    return {"entities": ents, "labels": labels}


# ---------------------- fallback planner/generator ----------------------

def _fallback_build_plan(topic: str, rdf_files: List[str], persona: str, seeds_from: Dict[str, List[str]]) -> Dict[str, Any]:
    titles = [
        "Introduction", "Context Setup", "Performance Detail",
        "Audience & Broadcast", "Cultural Impact", "Legacy & Reflection",
        "Aftermath", "Closing"
    ]
    beats = []
    for i, ttl in enumerate(titles):
        beats.append({
            "index": i,
            "title": ttl,
            "seed_entities": seeds_from.get("entities", []),
            "seed_labels": seeds_from.get("labels", []),
            "evidence": [{"type": "text", "value": f"{ttl}: {topic}", "source": "graph"}],
            "rows": []
        })
    return {"topic": topic, "persona": persona, "beats": beats}

# --- replace _fallback_generate_story(...) with this version ---
def _fallback_generate_story(plan: Dict[str, Any], out_jsonl: Path, out_story_md: Path,
                             out_story_clean_md: Path, beat_sentences: int, persona_name: str):
    beats = plan.get("beats") or plan.get("sections") or []
    answers_lines: List[str] = []
    paras: List[str] = []

    for i, b in enumerate(beats):
        title = b.get("title", "Section")
        # naive prose (placeholder)
        sents = []
        for _ in range(max(1, int(beat_sentences))):
            sents.append(f"{title}: The narrative explores how people, place, and purpose converged at the event, linking prior moments to subsequent impact.")
        para = " ".join(sents)
        paras.append(para)

        # NEW: collect evidence lines from the plan
        ctx_lines = _collect_context_lines_from_beat(b)

        # also emit beat_* keys for compatibility
        rec = {
            "section_index": b.get("index", i),
            "section_title": title,
            "beat_index": b.get("index", i),
            "beat_title": title,
            "context_lines": ctx_lines,
            "text": para,
        }
        answers_lines.append(json.dumps(rec, ensure_ascii=False))

    _write_text(out_jsonl, "\n".join(answers_lines) + ("\n" if answers_lines else ""))
    story_text = "\n\n".join(paras) + ("\n" if paras else "")
    _write_text(out_story_md, story_text)
    _write_text(out_story_clean_md, _simple_clean(story_text))

def unique_sentences(sentences, seen_ngrams, n=5, thresh=1):
    out = []
    for s in sentences:
        words = s.split()
        grams = {tuple(words[i:i+n]) for i in range(len(words)-n+1)} if len(words) >= n else {tuple(words)}
        if len(grams & seen_ngrams) <= thresh:
            out.append(s)
            seen_ngrams |= grams
    return out


# --- add near the other helpers in pipeline_graph.py ---
def _collect_context_lines_from_beat(b: Dict[str, Any]) -> List[str]:
    """
    Build per-beat context_lines from the plan's evidence/rows.
    Keeps short, atomic strings; de-duplicates case-insensitively.
    """
    lines: List[str] = []

    # evidence: usually [{ "value": "...", ...}, ...]
    for ev in b.get("evidence", []) or []:
        v = (ev.get("value") or ev.get("text") or ev.get("content") or "").strip()
        if v:
            lines.append(v)

    # rows: sometimes have text/value or triples
    for r in b.get("rows", []) or []:
        if isinstance(r, str):
            lines.append(r.strip())
            continue
        for key in ("text", "value", "label", "object", "o"):
            v = r.get(key)
            if isinstance(v, str) and v.strip():
                lines.append(v.strip())
        tri = r.get("triple") or r.get("spo")
        if isinstance(tri, (list, tuple)):
            parts = [str(x).strip() for x in tri if x]
            if parts:
                lines.append(" — ".join(parts))

    # tidy + de-dup (light)
    seen = set()
    out = []
    for ln in lines:
        s = _WS.sub(" ", ln).strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ---------------------- main graph runner ----------------------

def run_graph(args) -> None:
    # Per-run dirs
    run_root = Path(getattr(args, "run_root", "runs"))
    # timestamped run dir (if not already)
    if not run_root.exists() or not any((run_root / x).exists() for x in ("retriever", "generator", "params")):
        # If a raw folder was passed, create substructure
        (run_root / "retriever").mkdir(parents=True, exist_ok=True)
        (run_root / "generator").mkdir(parents=True, exist_ok=True)
        (run_root / "params").mkdir(parents=True, exist_ok=True)

    retriever_dir, generator_dir, params_dir = _ensure_run_dirs(run_root)

    plan_graph_json    = retriever_dir / "plan_with_evidence_Graph.json"
    answers_graph      = generator_dir / "answers_Graph.jsonl"
    story_graph        = generator_dir / "story_Graph.md"
    story_graph_clean  = generator_dir / "story_Graph_clean.md"

    # Config (load file if present; else create minimal from args)
    graph_cfg_path = Path(getattr(args, "graph_config", "graph_rag/graph_config.yaml"))
    if ext_load_graph_config and graph_cfg_path.exists():
        graph_cfg = ext_load_graph_config(graph_cfg_path)
    else:
        graph_cfg = {
            "llm_provider": getattr(args, "llm_provider", "gemini"),
            "llm_model": getattr(args, "llm_model", "gemini-2.5-flash"),
            "beam_width": 2,
            "max_hops": 2,
            "generation": {
                "beat_sentences": getattr(args, "beat_sentences", None)
            }
        }
    if yaml is not None:
        _write_text(params_dir / "graph_config_resolved.yaml", yaml.safe_dump(graph_cfg, sort_keys=False, allow_unicode=True))

    # Length profile (with argparse override allowed)
    prof = _graph_length_profile(getattr(args, "length", "Medium"))
    beat_sentences = int(graph_cfg.get("generation", {}).get("beat_sentences") or prof["beat_sentences"])

    # Inputs
    rdf_files = [str(p) for p in getattr(args, "rdf", [])]
    seeds_from = _derive_seeds(args)
    topic = _derive_topic(args)
    persona = getattr(args, "persona", "Emma")

    try:
        # 1) Build plan from RDF (real helper → fallback)
        if ext_build_graph_plan:
            plan_obj = ext_build_graph_plan(
                topic=topic,
                rdf_files=rdf_files,
                graph_cfg=graph_cfg,
                persona_name=persona,
                seeds_from=seeds_from,
            )
        else:
            plan_obj = _fallback_build_plan(topic, rdf_files, persona, seeds_from)

        plan_obj = postprocess_plan(plan_obj)
        # 2) Trim to requested length & de-duplicate evidence
        plan_obj = _apply_length_limit(plan_obj, prof["beats_limit"])
        plan_obj = _dedupe_plan_evidence(plan_obj, per_section_limit=6)

        # Persist plan
        if ext_write_plan:
            ext_write_plan(plan_obj, plan_graph_json)
        else:
            _json_dump(plan_graph_json, plan_obj)

        # 3) Generate story
        if ext_generate_graph_story:
            ext_generate_graph_story(
                graph_plan_path=plan_graph_json,
                out_jsonl=answers_graph,
                out_story_md=story_graph,
                out_story_clean_md=story_graph_clean,
                llm_provider=graph_cfg.get("llm_provider", "gemini"),
                llm_model=graph_cfg.get("llm_model", "gemini-2.5-flash"),
                beat_sentences=beat_sentences,
            )
        else:
            _fallback_generate_story(plan_obj, answers_graph, story_graph, story_graph_clean,
                                     beat_sentences=beat_sentences, persona_name=persona)

        # 4) Post-gen: dedupe + strict enforce length, then rebuild story files
        dedupe_answers_jsonl_inplace(answers_graph)
        enforce_length_targets(
            answers_jsonl=answers_graph,
            story_md=story_graph,
            story_clean_md=story_graph_clean,
            beats_limit=prof["beats_limit"],
            beat_sentences=beat_sentences,
            report_path=(params_dir / "length_report.json"),
        )

        print(f"✓ Graph plan → {plan_graph_json}")
        print(f"✓ Graph story → {story_graph} (+ clean {story_graph_clean})")
        print(f"✓ Length enforced → {params_dir / 'length_report.json'}")
    except Exception:
        traceback.print_exc()
        raise


# ---------------------- CLI ----------------------

def _timestamped_run_root(base: Path, persona: str, length: str, seed: int, tag: Optional[str]) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = lambda s: re.sub(r"[^a-z0-9\-]+", "-", (s or "").strip().lower())
    name = f"{ts}__{safe(persona)}-{safe(length)}__seed{seed}"
    if tag:
        name += f"__{safe(tag)}"
    return (base / name).resolve()

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Graph-only RAG pipeline (length-aware)")
    ap.add_argument("--rdf", nargs="+", required=True, help="RDF/TTL files for the knowledge graph")
    ap.add_argument("--persona", default="Emma")
    ap.add_argument("--length", default="Medium")   # we normalize small→Short etc.
    ap.add_argument("--items_per_beat", type=int, default=2)   # accepted for compat; not used here
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_root", default="runs", help="Root folder where timestamped run dirs are created.")
    ap.add_argument("--run_tag", default=None, help="Optional label appended to the run folder name.")

    # Params (compatible with your runner)
    ap.add_argument("--generator_params", default=None)        # JSON string or path? (runner passes string)
    ap.add_argument("--generator_params_json", default=None)
    ap.add_argument("--retriever_params", default=None)
    ap.add_argument("--retriever_params_json", default=None)

    # LLM knobs
    ap.add_argument("--llm_provider", choices=["ollama", "gemini"], default="gemini")
    ap.add_argument("--llm_model", default="gemini-2.5-flash")
    ap.add_argument("--ollama_num_ctx", type=int, default=4096)

    # Story shaping
    ap.add_argument("--beat_sentences", type=int, default=None, help="Override sentences per section")

    # Graph config file (optional)
    ap.add_argument("--graph_config", default="graph_rag/graph_config.yaml")

    # Compatibility flags (ignored, so runner can pass the same CLI as KG/Hybrid)
    ap.add_argument("--kg_meta", default=None)
    ap.add_argument("--hy_meta", default=None)
    ap.add_argument("--narrative_plans", default=None)
    ap.add_argument("--use_external_planner", action="store_true")
    ap.add_argument("--planner_path", default=None)
    ap.add_argument("--planner_match_strategy", default=None)
    ap.add_argument("--persist_params", action="store_true")
    return ap

def main():
    ap = build_argparser()
    args, _unknown = ap.parse_known_args()

    # Create a timestamped run folder under --run_root for this invocation
    base = Path(args.run_root)
    #run_dir = _timestamped_run_root(base, args.persona, _norm_length(args.length), args.seed, args.run_tag)
    run_dir = base
    run_dir.mkdir(parents=True, exist_ok=True)

    # Re-point args.run_root to the timestamped run folder so artifacts land there
    args.run_root = str(run_dir)

    random.seed(args.seed)
    run_graph(args)

if __name__ == "__main__":
    main()
