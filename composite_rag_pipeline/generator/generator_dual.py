#!/usr/bin/env python3
"""
generator_dual.py — Dual-mode generator (KG / Hybrid) with Story output

Inputs:
- evidence.jsonl (from retriever) OR plan_with_evidence.json (whole-data JSON)
- planner plan.json (for beat alignment and titles)
- optional cq_metadata.json (for per-CQ templates, not required)

LLMs:
- none (deterministic template fallback)
- ollama (local Llama)
- openai
- gemini (with throttling, retries, and disk cache)

Key features:
- Question placeholder substitution with --params (supports [Key] and {Key})
- Optional inclusion of URL titles/snippets with strict clipping
- Context pruning with a total character budget
- Chunked composition: one call per beat with tiny inputs, then stitch into a story
- Gemini rate-limit handling: min interval, exponential backoff, disk cache

Example (Hybrid; whole-data JSON; chunked per-beat compose with Llama):
-----------------------------------------------------------------------
python3 generator_dual.py \
  --plan ../planner/plan_Hybrid.json \
  --plan_with_evidence ../retriever/plan_with_evidence_Hybrid.json \
  --hy_meta ../index/Hybrid/cq_metadata.json \
  --params ../planner/params.json \
  --llm_provider ollama --llm_model llama3.1-128k --ollama_num_ctx 8192 \
  --use_url_content --max_url_snippets 1 --snippet_chars 140 \
  --max_rows 4 --context_budget_chars 1000 \
  --compose chunked --max_facts_per_beat 5 --beat_sentences 3 \
  --out answers_Hybrid.jsonl \
  --story_out story_Hybrid.md --include_citations
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Optional HTTP for LLMs
try:
    import requests
except Exception:
    requests = None

# ====================== Simple literal replacement ======================

def simple_replace(template: Optional[str], bindings: Dict[str, Any]) -> str:
    """
    Literal text replacement:
      - Replaces "[Key]" and "{Key}" with bindings[Key] (verbatim)
      - Case-sensitive keys; include quotes/IRIs yourself if needed
      - Replaces longer keys first to avoid 'Venue' vs 'Venue1' collisions
    """
    if not template:
        return ""
    if not bindings:
        return template
    out = template
    for k in sorted(bindings.keys(), key=lambda x: len(str(x)), reverse=True):
        v = str(bindings[k])
        key = str(k)
        out = out.replace(f"[{key}]", v).replace(f"{{{key}}}", v)
    return out

# ============================== IO helpers ==============================

def _load_json(path: Optional[Path]) -> Any:
    if not path:
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # tolerate bad lines
                pass
    return rows

def _load_meta(meta_path: Optional[Path]) -> Dict[str, Any]:
    if not meta_path:
        return {}
    meta = _load_json(meta_path) or {}
    return meta.get("metadata") or {}

# =========================== Evidence loaders ===========================

def _ev_from_jsonl(ev_path: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(ev_path)

def _ev_from_whole(plan_with_ev: Dict[str, Any], mode_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """Transform plan_with_evidence.json items into evidence-like records."""
    out: List[Dict[str, Any]] = []
    mode_default = (
        mode_override
        or plan_with_ev.get("retriever_stats", {}).get("mode")
        or plan_with_ev.get("mode")
        or "KG"
    )
    for it in plan_with_ev.get("items", []):
        beat_obj = it.get("beat") if isinstance(it.get("beat"), dict) else {
            "index": it.get("beat_index"),
            "title": it.get("beat_title") or "Beat"
        }
        rec = {
            "id": it.get("id"),
            "mode": mode_override or it.get("mode") or mode_default,
            "beat": beat_obj,
            "question": it.get("question", ""),
            "sparql": it.get("executed_query") or it.get("sparql") or "",
            "sparql_source": it.get("sparql_source") or "plan",
            "bindings": it.get("rows") or [],
        }
        if it.get("enrichment"):
            rec["enrichment"] = it["enrichment"]
        if it.get("url_info"):
            rec["url_info"] = it["url_info"]
        if it.get("url_candidates"):
            rec["url_candidates"] = it["url_candidates"]
        if not it.get("kg_ok"):
            rec["error"] = it.get("kg_reason") or "error"
        out.append(rec)
    return out

# ======================== Citations & context bits ======================

def _gather_kg_citations(bindings: List[Dict[str, str]], enrichment: Dict[str, Any]) -> List[str]:
    uris = set()
    for row in bindings or []:
        for v in row.values():
            if isinstance(v, str) and v.startswith("http"):
                uris.add(v)
    if enrichment:
        for ent in enrichment.get("entities", []) or []:
            u = ent.get("uri") or ent.get("id")
            if u and isinstance(u, str) and u.startswith("http"):
                uris.add(u)
        for card in enrichment.get("neighbors", []) or []:
            u = card.get("uri")
            if u and isinstance(u, str) and u.startswith("http"):
                uris.add(u)
            for nb in card.get("neighbors", []) or []:
                nuri = (nb.get("node") or {}).get("uri")
                puri = (nb.get("predicate") or {}).get("uri")
                if nuri and nuri.startswith("http"):
                    uris.add(nuri)
                if puri and puri.startswith("http"):
                    uris.add(puri)
    return sorted(uris)[:50]

def _pick_url_snippets(url_info: List[Dict[str, Any]], max_snips: int = 2, chars: int = 400) -> List[str]:
    snips = []
    for info in url_info or []:
        title = (info.get("title") or "").strip()
        url = info.get("url") or ""
        domain = info.get("domain") or ""
        text = (info.get("content_text") or "").strip()
        if title or text:
            head = f"{title} ({domain or url})".strip()
            if text:
                body = text[:chars].strip()
                if head:
                    snips.append(f"{head}: {body}")
                else:
                    snips.append(body)
            else:
                snips.append(head)
        if len(snips) >= max_snips:
            break
    return snips

# ============================== Context build ===========================

def _make_context(
    item: Dict[str, Any],
    *,
    style: str = "concise",
    max_rows: int = 6,
    use_url_content: bool = False,
    max_url_snippets: int = 2,
    snippet_chars: int = 400,
    budget: int = 1200,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build context text with hard character budget; prune in this order:
    Web snippets -> Neighbors -> Entities -> (Bindings last)
    """
    mode = item.get("mode") or "KG"
    bindings = item.get("bindings") or []
    enrichment = item.get("enrichment") or {}
    url_info = item.get("url_info") or []

    lines: List[str] = []
    used = {"bindings_rows": 0, "entities": 0, "neighbors_cards": 0, "url_snips": 0}

    # Bindings summary (first priority to keep)
    for r in bindings[:max_rows]:
        pairs = [f"{k}={v}" for k, v in r.items()]
        lines.append(" • " + "; ".join(pairs))
        used["bindings_rows"] += 1
    if not bindings:
        lines.append(" • (no rows)")

    # Hybrid KG enrichment
    if mode == "Hybrid" and enrichment:
        ents = enrichment.get("entities", [])
        if ents:
            lines.append("Entities:")
            for e in ents[:12]:
                lab = e.get("label") or ""
                uri = e.get("uri") or ""
                desc = e.get("desc") or ""
                v = f" - {lab or uri}"
                if lab and uri:
                    v += f" ({uri})"
                if desc:
                    v += f" — {desc[:140]}"
                lines.append(v)
                used["entities"] += 1
        neigh = enrichment.get("neighbors", [])
        if neigh:
            lines.append("Neighbors:")
            for card in neigh[:4]:
                lines.append(f" - From {card.get('uri','')}")
                for nb in card.get("neighbors", [])[:6]:
                    p = nb.get("predicate", {})
                    n = nb.get("node", {})
                    pl = p.get("label") or p.get("uri", "")
                    nl = n.get("label") or n.get("uri", "")
                    lines.append(f"    · {pl} -> {nl}")
                used["neighbors_cards"] += 1

    # Optional web-content snippets (lowest priority; prune first if needed)
    if use_url_content and url_info:
        snips = _pick_url_snippets(url_info, max_snips=max_url_snippets, chars=snippet_chars)
        if snips:
            lines.append("Web snippets:")
            for s in snips:
                lines.append(f" - {s}")
            used["url_snips"] = len(snips)

    # Build + budget
    ctx = "CONTEXT\n" + "\n".join(lines)
    if len(ctx) <= budget:
        return ctx, used

    # prune helpers
    def _drop_section(prefix_lc: str):
        nonlocal lines
        i = next((i for i, l in enumerate(lines) if l.strip().lower().startswith(prefix_lc)), None)
        if i is None:
            return
        j = i + 1
        while j < len(lines) and (lines[j].lstrip().startswith("-") or lines[j].lstrip().startswith("·")):
            j += 1
        lines = lines[:i] + lines[j:]

    # Prune in order: web → neighbors → entities
    for pref in ("web snippets:", "neighbors:", "entities:"):
        ctx = "CONTEXT\n" + "\n".join(lines)
        if len(ctx) <= budget:
            break
        _drop_section(pref)

    # Final hard cap
    ctx = "CONTEXT\n" + "\n".join(lines)
    if len(ctx) > budget:
        ctx = ctx[:budget]
    return ctx, used

SYS_KG = "You are a precise assistant. Answer strictly from the given context. Keep the answer concise and factual."
SYS_HY = "You are a precise assistant. Answer from the given context and you may add one brief sentence of relevant context from entities/links if available. Avoid speculation."

def _prompt_for(
    item: Dict[str, Any],
    question: str,
    *,
    style: str = "concise",
    use_url_content: bool = False,
    max_url_snippets: int = 2,
    snippet_chars: int = 400,
    budget: int = 1200,
) -> Tuple[str, str]:
    mode = item.get("mode") or "KG"
    system = SYS_KG if mode == "KG" else SYS_HY
    ctx, _ = _make_context(
        item,
        style=style,
        use_url_content=use_url_content,
        max_url_snippets=max_url_snippets,
        snippet_chars=snippet_chars,
        budget=budget,
    )
    user = f"""{ctx}

QUESTION
{question}

INSTRUCTIONS
- Cite entities/URIs inline in parentheses when helpful (short).
- If the context has no rows, say "No data found".
- Keep it to 1–2 sentences; no bullet points.
"""
    return system, user

# ================================ LLM calls ==============================

def call_ollama(model: str, prompt: str, base_url: str = None, system: str = None, options: dict | None = None) -> str:
    if requests is None:
        raise RuntimeError("requests not installed; cannot call Ollama.")
    url = (base_url or "http://localhost:11434").rstrip("/") + "/api/generate"
    full_prompt = f"<<SYS>>{system or ''}\n<</SYS>>\n\n{prompt}" if system else prompt
    #print(f"full_prompt : {full_prompt}")
    body = {"model": model, "prompt": full_prompt, "stream": False}
    if options:
        body["options"] = options
    r = requests.post(url, json=body, timeout=180)
    r.raise_for_status()
    response = r.json().get("response", "").strip()
    #print(f"response : {response}")
    return response

def call_openai(model: str, system: str, user: str, base_url: Optional[str] = None) -> str:
    if requests is None:
        raise RuntimeError("requests not installed; cannot call OpenAI.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    url = (base_url or "https://api.openai.com/v1/chat/completions").rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "n": 1,
    }
    r = requests.post(url, headers=headers, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def call_gemini(model: str, system: str, user: str, max_output_tokens: int = 120) -> str:
    if requests is None:
        raise RuntimeError("requests not installed; cannot call Gemini.")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "system_instruction": {"parts": [{"text": system or ""}]},
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": max_output_tokens,
            "candidateCount": 1
        }
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    for cand in (data.get("candidates") or []):
        parts = ((cand.get("content") or {}).get("parts") or [])
        for p in parts:
            if "text" in p:
                return p["text"].strip()
    return ""

# ===================== Throttle / Retry / Cache (Gemini) =====================

_last_llm_ts = 0.0
_CACHE: Dict[str, str] = {}

def _cache_path() -> Path:
    p = Path(".cache")
    p.mkdir(exist_ok=True)
    return p / "gemini_cache.json"

def _cache_load():
    global _CACHE
    fp = _cache_path()
    if fp.exists():
        try:
            _CACHE = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            _CACHE = {}

def _cache_save():
    try:
        _cache_path().write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _cache_key(provider: str, model: str, system: str, user: str) -> str:
    return hashlib.sha256((provider + "|" + model + "|" + system + "|" + user).encode("utf-8")).hexdigest()

def _sleep_if_needed(min_interval_ms: int):
    global _last_llm_ts
    now = time.time()
    gap = (min_interval_ms / 1000.0) - (now - _last_llm_ts)
    if gap > 0:
        time.sleep(gap)
    _last_llm_ts = time.time()

def smart_llm_call(provider: str, model: str, system: str, user: str, base_url: Optional[str], args) -> str:
    """
    Unified caller with min-interval throttle, retries on rate/quota errors, and disk cache for Gemini.
    """
    _sleep_if_needed(args.llm_min_interval_ms)

    # simple cache (only Gemini by default)
    key = _cache_key(provider, model, system, user)
    if provider == "gemini" and key in _CACHE:
        return _CACHE[key]

    backoff = args.llm_initial_backoff_ms / 1000.0
    for attempt in range(args.llm_retries + 1):
        try:
            if provider == "ollama":
                opts = {}
                if args.ollama_num_ctx:
                    opts["num_ctx"] = args.ollama_num_ctx
                resp = call_ollama(model, user, base_url=base_url, system=system, options=opts)
            elif provider == "openai":
                resp = call_openai(model, system, user, base_url=base_url)
            else:  # gemini
                resp = call_gemini(model, system, user, max_output_tokens=args.gemini_max_output_tokens)
                _CACHE[key] = resp
                _cache_save()
            return resp
        except Exception as e:
            msg = str(e).lower()
            is_rate = any(x in msg for x in ["rate", "quota", "429", "resource_exhausted", "unavailable", "503"])
            if attempt < args.llm_retries and is_rate:
                time.sleep(min(backoff, args.llm_max_backoff_ms / 1000.0))
                backoff *= args.llm_backoff_multiplier
                continue
            raise

# =========================== Deterministic fallback ======================

def _template_answer(item: Dict[str, Any], question: str) -> str:
    rows = item.get("bindings") or []
    if not rows:
        return "No data found."
    row = rows[0]
    pairs = [f"{k}={v}" for k, v in list(row.items())[:3]]
    base = "; ".join(pairs)
    if (item.get("mode") == "Hybrid") and (item.get("enrichment") or {}).get("entities"):
        ent = item["enrichment"]["entities"][0]
        lab = ent.get("label") or ""
        if lab:
            return f"{base}. Related: {lab}."
    return base + "."

# ============================== Story weaving ===========================

def _story_header(title: str, mode: str, length: str) -> str:
    return f"# {title} — {mode} Story ({length})\n"

def _beat_header(title: str, fmt_md: bool = True) -> str:
    return f"## {title}\n" if fmt_md else f"{title}\n" + "-" * len(title) + "\n"

def _story_from_answers(
    plan: Dict[str, Any],
    answers: List[Dict[str, Any]],
    include_citations: bool,
    fmt_md: bool = True,
) -> str:
    by_beat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    titles: Dict[int, str] = {}
    for idx, b in enumerate(plan.get("beats", [])):
        if isinstance(b, dict):
            titles[idx] = b.get("title") or f"Beat {idx}"
    for a in answers:
        b = (a.get("beat") or {}).get("index")
        if isinstance(b, int):
            by_beat[b].append(a)

    mode = answers[0].get("mode") if answers else (plan.get("mode") or "KG")
    title = plan.get("persona") or "WembleyRewind"
    length = plan.get("length") or "Medium"

    parts = [_story_header(title, mode, length)]
    footnotes: List[str] = []
    fn_counter = 1

    for b in sorted(by_beat.keys()):
        parts.append(_beat_header(titles.get(b, f"Beat {b}"), fmt_md))
        paras = []
        for a in by_beat[b]:
            ans = (a.get("answer") or "").strip()
            if not ans:
                continue
            line = ans
            if include_citations:
                for uri in (a.get("citations") or [])[:2]:
                    footnotes.append(uri)
                    line += f" [{fn_counter}]"
                    fn_counter += 1
            paras.append(line)
        parts.append((" ".join(paras) + "\n") if paras else ("_No data found for this beat._\n" if fmt_md else "No data found for this beat.\n"))

    if include_citations and footnotes:
        parts.append("\n### References\n" if fmt_md else "\nReferences\n----------\n")
        for i, uri in enumerate(footnotes, start=1):
            parts.append(f"[{i}] {uri}\n")

    return "".join(parts)

# --------------- Chunked per-beat composition (small LLM calls) ----------

def compose_story_chunked(plan: Dict[str, Any], answers: List[Dict[str, Any]], args) -> str:
    # group answers by beat
    print(f"answers : {answers}")
    by_beat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    beat_titles: Dict[int, str] = {}
    for i, b in enumerate(plan.get("beats", [])):
        if isinstance(b, dict):
            beat_titles[i] = b.get("title") or f"Beat {i}"
    for a in answers:
        idx = (a.get("beat") or {}).get("index")
        if isinstance(idx, int):
            by_beat[idx].append(a)

    paragraphs: List[str] = []
    print(f"bybeat : {by_beat.keys()}")
    for idx in sorted(by_beat.keys()):
        title = beat_titles.get(idx, f"Beat {idx}")
        print(f"Processing beat {idx}: {title}")
        facts: List[str] = []
        # take the answer text as a "fact", clipped
        for rec in by_beat[idx]:
            t = (rec.get("answer") or "").strip()
            if not t:
                continue
            facts.append(t[:120])
            if len(facts) >= args.max_facts_per_beat:
                break

        if not facts:
            paragraphs.append(f"## {title}\n_No data found for this beat._\n")
            continue

        system = f"Rewrite the facts into a cohesive paragraph. Use at most {args.beat_sentences} sentences. Be faithful; no new facts."
        user = f"Beat: {title}\nFacts:\n- " + "\n- ".join(facts)

        if args.llm_provider == "none":
            para = " ".join(facts)[:400]
        else:
            para = smart_llm_call(args.llm_provider, args.llm_model, system, user, args.llm_base_url, args)
        paragraphs.append(f"## {title}\n{para.strip()}\n")

    # Optional intro (tiny call)
    intro = ""
    try:
        beat_list = ", ".join([beat_titles[i] for i in sorted(beat_titles.keys())])[:300]
        sys2 = "Write a 1–2 sentence intro for this story. No new facts."
        usr2 = f"Persona: {plan.get('persona','WembleyRewind')}\nBeats: {beat_list}"
        if args.llm_provider == "none":
            intro = f"{plan.get('persona','WembleyRewind')} explores {beat_list}."
        else:
            intro = smart_llm_call(args.llm_provider, args.llm_model, sys2, usr2, args.llm_base_url, args)
    except Exception:
        pass

    header = f"# {plan.get('persona','WembleyRewind')} — {plan.get('mode','Story')} Story ({plan.get('length','Medium')})\n"
    body = ("\n".join(paragraphs)).strip() + "\n"

    # References from answers (if requested)
    refs = []
    if args.include_citations:
        for a in answers:
            for uri in (a.get("citations") or [])[:2]:
                refs.append(uri)
        refs = list(dict.fromkeys(refs))  # dedupe preserving order
        if refs:
            body += "\n### References\n" + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(refs)) + "\n"

    return header + (intro.strip() + "\n\n" if intro else "") + body

# ================================ Helpers ===============================

def _index_questions_from_plan(plan: Dict[str, Any], bindings: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a map id->question from plan, robust to beats[i].items being an INT.
    Applies simple_replace with provided bindings.
    """
    q: Dict[str, str] = {}
    # flat items variant
    for it in (plan.get("items") or []):
        if isinstance(it, dict) and it.get("id"):
            qt = it.get("question") or ""
            q[it["id"]] = simple_replace(qt, bindings)
    # beats variant (items may be an int or a list)
    for b in (plan.get("beats") or []):
        if not isinstance(b, dict):
            continue
        b_items = b.get("items")
        if isinstance(b_items, list):
            for it in b_items:
                if isinstance(it, dict) and it.get("id"):
                    qt = it.get("question") or ""
                    q[it["id"]] = simple_replace(qt, bindings)
        # if int: nothing to index
    return q

# ================================= Main =================================

def main():
    ap = argparse.ArgumentParser(description="Dual-mode generator for KG/Hybrid evidence, with optional story output and chunked composition.")
    # Inputs
    ap.add_argument("--plan", required=True, help="plan.json from planner (for question/beat alignment)")
    ap.add_argument("--evidence", default=None, help="evidence.jsonl from retriever")
    ap.add_argument("--plan_with_evidence", default=None, help="retriever's combined JSON (plan_with_evidence.json)")
    ap.add_argument("--kg_meta", default=None, help="KG cq_metadata.json (optional)")
    ap.add_argument("--hy_meta", default=None, help="Hybrid cq_metadata.json (optional)")
    ap.add_argument("--mode_override", default=None, choices=["KG", "Hybrid"], help="Force mode if missing")
    ap.add_argument("--params", default=None, help="JSON with question placeholders (replaces [Key] and {Key})")

    # LLM
    ap.add_argument("--llm_provider", default="none", choices=["none", "ollama", "openai", "gemini"])
    ap.add_argument("--llm_model", default="llama3.1")
    ap.add_argument("--llm_base_url", default=None)

    # LLM rate & cache controls (primarily for Gemini)
    ap.add_argument("--llm_min_interval_ms", type=int, default=1200, help="Minimum spacing between LLM calls")
    ap.add_argument("--llm_retries", type=int, default=5)
    ap.add_argument("--llm_initial_backoff_ms", type=int, default=800)
    ap.add_argument("--llm_backoff_multiplier", type=float, default=2.0)
    ap.add_argument("--llm_max_backoff_ms", type=int, default=8000)
    ap.add_argument("--gemini_max_output_tokens", type=int, default=120)
    ap.add_argument("--ollama_num_ctx", type=int, default=None, help="Ollama context window (e.g., 8192)")

    # Context controls
    ap.add_argument("--style", default="concise", choices=["concise", "rich"])
    ap.add_argument("--max_rows", type=int, default=6)
    ap.add_argument("--use_url_content", action="store_true", help="Include URL titles/snippets from retriever in context")
    ap.add_argument("--max_url_snippets", type=int, default=2)
    ap.add_argument("--snippet_chars", type=int, default=400)
    ap.add_argument("--context_budget_chars", type=int, default=1200, help="Hard cap on context size (characters) per CQ prompt")

    # Compose mode
    ap.add_argument("--compose", default="none", choices=["none", "chunked"], help="If 'chunked', write per-beat paragraphs with small LLM calls")
    ap.add_argument("--max_facts_per_beat", type=int, default=6)
    ap.add_argument("--beat_sentences", type=int, default=3)

    # Outputs
    ap.add_argument("--out", required=True, help="Write answers JSONL here")
    ap.add_argument("--story_out", default=None, help="Write composed story here (e.g., story.md)")
    ap.add_argument("--story_format", default="md", choices=["md", "txt"])
    ap.add_argument("--include_citations", action="store_true", help="Append compact numbered references")

    args = ap.parse_args()

    # Load cache for Gemini
    _cache_load()

    plan = _load_json(Path(args.plan)) or {}
    kg_rows = _load_meta(Path(args.kg_meta) if args.kg_meta else None)
    hy_rows = _load_meta(Path(args.hy_meta) if args.hy_meta else None)

    # Question bindings
    q_bindings: Dict[str, Any] = {}
    if args.params:
        q_bindings = _load_json(Path(args.params)) or {}

    # Load evidence
    if args.plan_with_evidence:
        whole = _load_json(Path(args.plan_with_evidence)) or {}
        evs = _ev_from_whole(whole, mode_override=args.mode_override)
    elif args.evidence:
        evs = _ev_from_jsonl(Path(args.evidence))
    else:
        print("ERROR: provide either --evidence or --plan_with_evidence", file=sys.stderr)
        sys.exit(2)

    # Map id->question (with bindings applied)
    q_by_id = _index_questions_from_plan(plan, q_bindings)

    # Generate per-item answers
    out_answers: List[Dict[str, Any]] = []

    for item in evs:
        cid = item.get("id")
        mode = (item.get("mode") or args.mode_override or plan.get("mode") or "KG").strip()

        # prefer question on the evidence item, but always apply substitution
        raw_q = item.get("question") or q_by_id.get(cid) or "Answer the question based on the context."
        question = simple_replace(raw_q, q_bindings)

        # optional meta template lookup (not required for output)
        tmpl = None
        if mode == "KG" and kg_rows:
            tmpl = (kg_rows.get(cid, {}) or {}).get("answer")
        elif mode == "Hybrid" and hy_rows:
            tmpl = (hy_rows.get(cid, {}) or {}).get("answer")

        # citations from KG URIs; URLs could be appended similarly if desired
        citations = _gather_kg_citations(item.get("bindings") or [], item.get("enrichment") or {})

        # Answer
        if args.llm_provider == "none":
            answer = _template_answer(item, question)
        else:
            sys_msg, user_msg = _prompt_for(
                item,
                question,
                style=args.style,
                use_url_content=args.use_url_content,
                max_url_snippets=args.max_url_snippets,
                snippet_chars=args.snippet_chars,
                budget=args.context_budget_chars,
            )
            answer = smart_llm_call(args.llm_provider, args.llm_model, sys_msg, user_msg, args.llm_base_url, args)

        rec = {
            "id": cid,
            "mode": mode,
            "beat": item.get("beat"),
            "question": question,
            "answer": answer,
            "citations": citations,
            "used": {
                "rows": min(len(item.get("bindings") or []), args.max_rows),
                "has_enrichment": bool(item.get("enrichment")),
                "has_url_info": bool(item.get("url_info")),
            },
        }
        if tmpl:
            rec["template"] = tmpl
        out_answers.append(rec)

    # Write answers JSONL
    with Path(args.out).open("w", encoding="utf-8") as fout:
        for rec in out_answers:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out} ({len(out_answers)} answers)")

    # Story
    if args.story_out:
        if args.compose == "chunked":
            story = compose_story_chunked(plan, out_answers, args)
        else:
            fmt_md = (args.story_format == "md")
            story = _story_from_answers(plan, out_answers, include_citations=args.include_citations, fmt_md=fmt_md)
        Path(args.story_out).write_text(story, encoding="utf-8")
        print(f"Wrote {args.story_out}")

if __name__ == "__main__":
    main()
