#!/usr/bin/env python3
"""
generator_dual.py
- Template mode (deterministic)
- Ollama mode (optional, as before)
- Gemini mode (large context, packs URL content from retriever)

Outputs:
  - narrative.md
  - narrative_trace.json
  - (Gemini mode) prompt_gemini.json  ← exact system/user/evidence used

Example:
  python generator_dual.py \
    --plan plan_final.json \
    --mode gemini \
    --gemini_model gemini-1.5-pro \
    --out_md run_trace/narrative.md \
    --out_json run_trace/narrative_trace.json
"""

from __future__ import annotations
import argparse, json, re, textwrap, hashlib, datetime, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- optional deps ----------
try:
    from rdflib import Graph, term
except Exception:
    Graph, term = None, None

try:
    from ollama import generate as ollama_generate  # pip install ollama
except Exception:
    ollama_generate = None

try:
    import google.generativeai as genai   # pip install google-generativeai
except Exception:
    genai = None


# =========================
# Persona & length styles
# =========================

PERSONA_STYLE = {
    "Emma": {
        "voice": "Warm, accessible, lightly explanatory; assumes non-expert reader.",
        "rhythm": "Short paragraphs; clear topic sentences.",
    },
    "Luca": {
        "voice": "Historian-leaning, contextual, a touch analytical.",
        "rhythm": "Medium paragraphs; weave cause/effect.",
    },
    "_default": {
        "voice": "Neutral, engaging, informative.",
        "rhythm": "Short-to-medium paragraphs.",
    }
}
def pick_style(persona: str) -> Dict[str, str]:
    return PERSONA_STYLE.get(persona, PERSONA_STYLE["_default"])

def target_sentence_hint(length: str) -> str:
    m = {"short":"~2–3 sentences per beat",
         "medium":"~3–5 sentences per beat",
         "long":"~5–7 sentences per beat"}
    return m.get((length or "").lower(), "~3–5 sentences per beat")


# =========================
# Optional KG enrichment
# =========================

def _apply_bindings(sparql: str, bindings: Dict[str, str]) -> str:
    q = sparql or ""
    for k, v in (bindings or {}).items():
        q = q.replace(f"[{k}]", v).replace(f"{{{k}}}", v)
    return q

def _ensure_limit(q: str, n: int) -> str:
    return q if re.search(r"\blimit\s+\d+\b", q, flags=re.I) else (q.rstrip() + f"\nLIMIT {n}")

def _n3(x: "term.Node") -> str:
    try: return x.n3()
    except Exception: return str(x)

def enrich_with_rows(items: List[Dict[str, Any]], rdf_files: List[str], bindings: Dict[str, str],
                     per_item_sample: int, include_executed_query: bool) -> None:
    """Mutates items: adds/overwrites 'rows' (list of dict) and 'row_count'."""
    if Graph is None:
        return
    g = Graph()
    for f in rdf_files:
        g.parse(f)
    for it in items:
        sparql = it.get("sparql") or ""
        if not sparql:
            it.setdefault("row_count", 0)
            it.setdefault("rows", [])
            continue
        q = _ensure_limit(_apply_bindings(sparql, bindings), per_item_sample)
        try:
            res = g.query(q)
            rows = []
            for b in getattr(res, "bindings", [])[:per_item_sample]:
                rows.append({k: _n3(v) for k, v in b.items()})
            it["rows"] = rows
            it["row_count"] = len(getattr(res, "bindings", []))
            if include_executed_query:
                it["executed_query"] = q
        except Exception as e:
            it["rows"] = []
            it["row_count"] = 0
            it["kg_error"] = f"{type(e).__name__}: {e}"
            if include_executed_query:
                it["executed_query"] = q  # still useful for debugging


# =========================
# Pre-processing (token resolution)
# =========================

TOKEN_RX = re.compile(r"\[([A-Za-z0-9_:-]+)\]")

def _strip_quotes(s: str) -> str:
    if s is None: return ""
    t = str(s).strip()
    if t.startswith("<") and t.endswith(">"):
        return ""
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1]
    return re.sub(r"\s+", " ", t).strip()

def _split_list_field(val: str) -> List[str]:
    if not val: return []
    parts = [p.strip() for p in str(val).split(",")]
    out = [_strip_quotes(p) for p in parts if _strip_quotes(p)]
    seen, uniq = set(), []
    for v in out:
        k = v.lower()
        if k not in seen:
            seen.add(k); uniq.append(v)
    return uniq

def _collect(rows: List[Dict[str, Any]], var: str) -> List[str]:
    vals = []
    for r in rows or []:
        if var in r:
            vals.append(_strip_quotes(r[var]))
    seen, out = set(), []
    for v in vals:
        k = v.lower()
        if k not in seen and v:
            seen.add(k); out.append(v)
    return out

def _first(rows: List[Dict[str, Any]], var: str) -> str:
    vals = _collect(rows, var)
    return vals[0] if vals else ""

def _normalize_instrument(name: str) -> str:
    if not name: return ""
    n = name.strip()
    repl = {
        "Bassguitar": "bass guitar",
        "Drums (Drum Set)": "drum kit",
        "Drums": "drums",
        "Acoustic Guitar": "acoustic guitar",
        "Electric Guitar": "electric guitar",
        "Voice": "vocals",
    }
    return repl.get(n, n).strip()

def _english_list(items: List[str], max_items: int | None = None) -> str:
    xs = items[:max_items] if max_items else list(items)
    if not xs: return ""
    if len(xs) == 1: return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]

def _resolve_answer(answer: str, rows: List[Dict[str, Any]]) -> str:
    if not answer: return ""
    a = str(answer)

    event_name  = _first(rows, "eventName")
    artist_name = _first(rows, "artistName")
    venues = _split_list_field(_first(rows, "allVenues"))
    instruments = [_normalize_instrument(v) for v in _collect(rows, "instrumentName")]
    instruments = [v for v in instruments if v]
    works = _split_list_field(_first(rows, "allWorkName"))
    genres = [g.lower() for g in _split_list_field(_first(rows, "earlyMusicInfluenceGenre"))]

    mapping: Dict[str, str] = {
        "Event": event_name or "the event",
        "MusicGroup": artist_name or "the band",
        "MusicArtist": artist_name or "the artist",
        "Instrument": _english_list(instruments, 4) or "instruments",
        "Venue1": venues[0] if len(venues) > 0 else "the venue",
        "Venue2": venues[1] if len(venues) > 1 else (venues[0] if venues else "the venue"),
        "Song1": works[0] if len(works) > 0 else "a song",
        "Song2": works[1] if len(works) > 1 else (works[0] if works else "a song"),
        "Song3": works[2] if len(works) > 2 else (works[0] if works else "a song"),
        "Genre1": genres[0] if len(genres) > 0 else "a genre",
        "Genre2": genres[1] if len(genres) > 1 else (genres[0] if genres else "a genre"),
        "Genre3": genres[2] if len(genres) > 2 else (genres[0] if genres else "a genre"),
    }

    def repl(m: re.Match):
        tok = m.group(1)
        return mapping.get(tok, m.group(0))
    a = TOKEN_RX.sub(repl, a)

    def sweep_generic(txt: str) -> str:
        def gen_replace(tok: str) -> str:
            t = tok.lower()
            if t.startswith("song"): return "a song"
            if t.startswith("genre"): return "a genre"
            if t.startswith("venue"): return "the venue"
            if t in ("event",): return "the event"
            if t in ("musicgroup",): return "the band"
            if t in ("musicartist",): return "the artist"
            if t in ("instrument",): return "instruments"
            return "it"
        return TOKEN_RX.sub(lambda m: gen_replace(m.group(1)), txt)

    a = sweep_generic(a)
    a = re.sub(r"\s+\.", ".", a)
    a = re.sub(r"\s+,", ",", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a

def preprocess_items(plan: Dict[str, Any]) -> None:
    for it in plan.get("items", []):
        rows = it.get("rows") or []
        it["answer_resolved"] = _resolve_answer(it.get("answer") or "", rows)
        it["question_resolved"] = _resolve_answer(it.get("question") or it.get("text") or "", rows)


# =========================
# Evidence packing (URL content → context)
# =========================

def collect_url_evidence(plan: Dict[str, Any],
                         max_docs: int = 18,
                         max_chars_per_doc: int = 4000) -> List[Dict[str, str]]:
    """
    Pulls content_text from retriever rows (__url_info[].content_text), dedupes by URL,
    trims per-doc, and prefers longer texts first.
    """
    docs: List[Dict[str, str]] = []
    for it in plan.get("items", []):
        for r in it.get("rows") or []:
            for info in r.get("__url_info", []):
                txt = info.get("content_text") or ""
                if not txt:
                    continue
                u = info.get("url") or ""
                dom = info.get("domain") or ""
                title = info.get("title") or ""
                docs.append({
                    "url": u,
                    "domain": dom,
                    "title": title,
                    "text": (txt[:max_chars_per_doc]).strip(),
                    "len": len(txt)
                })
    # sort by length desc, dedupe by url
    seen, out = set(), []
    for d in sorted(docs, key=lambda x: x["len"], reverse=True):
        if d["url"] in seen:
            continue
        seen.add(d["url"])
        out.append({k: d[k] for k in ("url","domain","title","text")})
        if len(out) >= max_docs:
            break
    return out


# =========================
# Rendering (template mode)
# =========================

def render_template(plan: Dict[str, Any]) -> str:
    beats_order: List[str] = plan.get("beats", [])
    persona = plan.get("persona") or ""
    length  = plan.get("length") or ""
    style = pick_style(persona)
    title = f"{persona} — {length} Narrative".strip(" —")

    by_beat: Dict[str, List[Dict[str, Any]]] = {b: [] for b in beats_order}
    for it in plan.get("items", []):
        labels = it.get("beat")
        labs = list(labels) if isinstance(labels, (list, tuple, set)) else [labels or "Unspecified"]
        for b in labs:
            if b in by_beat:
                by_beat[b].append(it)

    lines = [f"# {title}", ""]
    lines.append(f"_Style: {style['voice']} • {style['rhythm']}_")
    if length:
        lines.append(f"_Guidance: {target_sentence_hint(length)}_")
    lines.append("")

    for beat in beats_order:
        items = by_beat.get(beat, [])
        if not items:
            continue
        lines.append(f"## {beat}")
        for it in items:
            q = (it.get("question_resolved") or it.get("question") or it.get("text") or "").strip()
            a = (it.get("answer_resolved") or it.get("answer") or "").strip()
            lines.append(f"**Q:** {q}")
            lines.append(f"**A:** {a if a else '_(TBD)_'}")
            ev = it.get("rows") or []
            if ev:
                lines.append("<details><summary>Data sample</summary>")
                for r in ev[:2]:
                    pretty = "; ".join(f"{k}={v}" for k, v in r.items() if not str(k).startswith("__"))
                    lines.append(f"- {pretty}")
                lines.append("</details>")
            lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# =========================
# LLM prompts (shared rules)
# =========================

def build_shared_system(style: Dict[str, str], length: str) -> str:
    length_hint = target_sentence_hint(length)
    return "\n".join([
        "You are a narrative generator for a knowledge-grounded story.",
        "",
        "Strict rules",
        "- Do NOT invent facts beyond what's provided.",
        "- NEVER output unresolved bracketed tokens like [Event], [Venue1], [Song1]. If a value is missing, rewrite the sentence generically.",
        "- Use clear UK English. Persona voice and rhythm must be followed exactly.",
        "- Avoid bullet points unless data demands it; prefer cohesive paragraphs.",
        "- Keep each beat self-contained but add 1 short transitional phrase between beats.",
        "",
        "Grounding & data use",
        "- Prefer concrete values found in the provided rows/content. Strip any surrounding quotes from values.",
        "- Split comma-separated lists (e.g., allWorkName, allVenues, earlyMusicInfluenceGenre), trim, normalise, and pick a concise subset.",
        "- Never print IRIs or angle brackets.",
        "",
        "Normalisation rules",
        "- Instruments: normalise (e.g., “Bassguitar” → “bass guitar”; “Drums (Drum Set)” → “drum kit”).",
        "- Song titles: keep capitalisation; join with commas and “and” for the last item.",
        "- Genres: lower-case unless proper nouns; use 3–5.",
        "- Venues: natural phrasing (“Wembley Stadium and John F. Kennedy Stadium”).",
        "",
        "Redundancy control",
        "- If similar content appears in multiple beats, vary phrasing and focus.",
        "- Do not repeat the same list verbatim across beats; refer back briefly if needed.",
        "",
        f"Persona voice: {style['voice']}",
        f"Rhythm: {style['rhythm']}",
        f"Use {length_hint}.",
    ])

def build_user_prompt(persona: str, style: Dict[str,str], length: str,
                      beats_order: List[str], by_beat_payload: Dict[str, Any]) -> str:
    length_hint = target_sentence_hint(length)
    return textwrap.dedent(f"""
        Render a cohesive narrative organised by beats in this exact order:
        {beats_order}

        Persona: {persona}
        Voice & rhythm: {style['voice']} • {style['rhythm']}
        Length hint: {length_hint}

        JSON content:
        {json.dumps({"beats_order": beats_order, "content": by_beat_payload}, ensure_ascii=False)}
    """).strip()


# =========================
# Render (Ollama)
# =========================

def render_llm_ollama(plan: Dict[str, Any], model: str, temperature: float) -> str:
    if ollama_generate is None:
        return render_template(plan)
    persona = plan.get("persona") or ""
    length  = plan.get("length") or ""
    style = pick_style(persona)
    beats_order: List[str] = plan.get("beats", [])
    by_beat: Dict[str, List[Dict[str, Any]]] = {b: [] for b in beats_order}
    for it in plan.get("items", []):
        labels = it.get("beat")
        labs = list(labels) if isinstance(labels, (list, tuple, set)) else [labels or "Unspecified"]
        for b in labs:
            if b in by_beat:
                by_beat[b].append({
                    "q": it.get("question_resolved") or it.get("question") or it.get("text") or "",
                    "a": it.get("answer_resolved") or it.get("answer") or "",
                    "rows": it.get("rows") or []
                })

    sys_prompt = build_shared_system(style, length)
    user_prompt = build_user_prompt(persona, style, length, beats_order, by_beat)
    prompt = f"<<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt}"
    resp = ollama_generate(model=model, prompt=prompt, options={"temperature": temperature})
    return (resp.get("response") or "").strip() + "\n"


# =========================
# Render (Gemini)
# =========================

def render_llm_gemini(plan: Dict[str, Any],
                      model: str,
                      temperature: float,
                      evidence_docs: List[Dict[str,str]],
                      save_prompt_path: str | None = None) -> str:
    if genai is None:
        # Fallback if library isn't present
        return render_template(plan)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment.")
    genai.configure(api_key=api_key)

    persona = plan.get("persona") or ""
    length  = plan.get("length") or ""
    style = pick_style(persona)
    beats_order: List[str] = plan.get("beats", [])

    # Use resolved answers to avoid bracket leaks
    by_beat: Dict[str, List[Dict[str, Any]]] = {b: [] for b in beats_order}
    for it in plan.get("items", []):
        labels = it.get("beat")
        labs = list(labels) if isinstance(labels, (list, tuple, set)) else [labels or "Unspecified"]
        for b in labs:
            if b in by_beat:
                by_beat[b].append({
                    "q": it.get("question_resolved") or it.get("question") or it.get("text") or "",
                    "a": it.get("answer_resolved") or it.get("answer") or "",
                    "rows": it.get("rows") or []
                })

    sys_prompt = build_shared_system(style, length)
    user_prompt = build_user_prompt(persona, style, length, beats_order, by_beat)

    # Build evidence blocks (trimmed URL content)
    ctx_blocks = []
    for d in evidence_docs:
        header = f"[{d.get('domain','')} | {d.get('title','').strip() or 'Untitled'}] {d.get('url','')}"
        block = header + "\n" + d.get("text","")
        ctx_blocks.append(block)

    # Compose Gemini call
    # Use system_instruction + contents (context blocks first, then user instruction)
    model_obj = genai.GenerativeModel(model_name=model, system_instruction=sys_prompt)
    contents = []
    # Pack evidence in manageable chunks (Gemini can take lots; we still chunk to be safe)
    for blk in ctx_blocks:
        contents.append({"role": "user", "parts": [blk]})
    contents.append({"role": "user", "parts": [user_prompt]})

    gen_cfg = {"temperature": float(temperature)}
    resp = model_obj.generate_content(contents=contents, generation_config=gen_cfg)
    text = (getattr(resp, "text", None) or "").strip()
    if not text and getattr(resp, "candidates", None):
        # Some SDK versions keep text inside candidates
        try:
            text = resp.candidates[0].content.parts[0].text.strip()
        except Exception:
            text = ""

    # Save prompt pack for traceability
    if save_prompt_path:
        payload = {
            "system": sys_prompt,
            "user": user_prompt,
            "evidence_docs": evidence_docs[:10],  # sample first 10 to keep file readable
            "model": model,
            "temperature": temperature,
        }
        Path(save_prompt_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return (text or "").strip() + "\n"


# =========================
# Trace JSON builder
# =========================

def build_trace(plan: Dict[str, Any],
                rdf_files: List[str],
                bindings: Dict[str, Any],
                mode: str,
                include_executed_query: bool) -> Dict[str, Any]:
    stamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    run_id = hashlib.sha256((plan.get("persona","") + "|" + plan.get("length","") + "|" + stamp).encode("utf-8")).hexdigest()[:12]

    items: List[Dict[str, Any]] = []
    for it in plan.get("items", []):
        rec = {
            "id": it.get("id"),
            "beat": it.get("beat"),
            "question": it.get("question"),
            "question_resolved": it.get("question_resolved"),
            "answer": it.get("answer"),
            "answer_resolved": it.get("answer_resolved"),
            "sparql": it.get("sparql"),
            "kg_ok": it.get("kg_ok"),
            "kg_reason": it.get("kg_reason"),
            "row_count": it.get("row_count"),
            "rows": it.get("rows"),
        }
        if include_executed_query and "executed_query" in it:
            rec["executed_query"] = it["executed_query"]
        items.append(rec)

    trace = {
        "run_id": run_id,
        "generated_at_utc": stamp,
        "persona": plan.get("persona"),
        "length": plan.get("length"),
        "beats": plan.get("beats", []),
        "total_limit": plan.get("total_limit"),
        "mode": mode,
        "bindings": bindings or {},
        "rdf_files": rdf_files or [],
        "retriever_stats": plan.get("retriever_stats"),
        "topup_stats": plan.get("topup_stats"),
        "items": items
    }
    return trace


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="final plan JSON (after top-up)")
    ap.add_argument("--mode", choices=["template","ollama","gemini"], default="template")
    ap.add_argument("--out_md", default="narrative.md")
    ap.add_argument("--out_json", default="narrative_trace.json")
    # optional KG enrichment
    ap.add_argument("--rdf", nargs="*", default=None, help="ttl/nt/n3/rdf/xml files to re-run SPARQL")
    ap.add_argument("--bindings", default=None, help="JSON with placeholder bindings (include <> for IRIs)")
    ap.add_argument("--per_item_sample", type=int, default=5)
    ap.add_argument("--include_executed_query", action="store_true",
                    help="Embed the fully bound SPARQL query if we enrich here")
    # LLM options
    ap.add_argument("--ollama_model", default="llama3")
    ap.add_argument("--temperature", type=float, default=0.4)

    # Gemini options
    ap.add_argument("--provider", choices=["ollama","gemini"], default=None,
                    help="(deprecated) use --mode gemini/ollama; kept for compatibility.")
    ap.add_argument("--gemini_model", default="gemini-1.5-pro")
    ap.add_argument("--evidence_max_docs", type=int, default=18)
    ap.add_argument("--evidence_max_chars_per_doc", type=int, default=4000)
    ap.add_argument("--save_gemini_prompt", default="prompt_gemini.json")

    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    # Optional enrichment with fresh KG rows
    bindings = json.loads(Path(args.bindings).read_text(encoding="utf-8")) if args.bindings else {}
    if args.rdf:
        enrich_with_rows(plan.get("items", []), args.rdf, bindings, per_item_sample=args.per_item_sample, include_executed_query=args.include_executed_query)

    # Pre-process: resolve tokens BEFORE rendering
    preprocess_items(plan)

    # Render narrative
    mode = args.mode
    if mode == "ollama" or (args.provider == "ollama"):
        md = render_llm_ollama(plan, model=args.ollama_model, temperature=args.temperature)
    elif mode == "gemini" or (args.provider == "gemini"):
        evidence_docs = collect_url_evidence(
            plan,
            max_docs=args.evidence_max_docs,
            max_chars_per_doc=args.evidence_max_chars_per_doc
        )
        md = render_llm_gemini(
            plan,
            model=args.gemini_model,
            temperature=args.temperature,
            evidence_docs=evidence_docs,
            save_prompt_path=args.save_gemini_prompt
        )
    else:
        md = render_template(plan)

    Path(args.out_md).write_text(md, encoding="utf-8")

    # Build + write trace JSON (includes resolved fields)
    trace = build_trace(plan, rdf_files=args.rdf or [], bindings=bindings, mode=mode, include_executed_query=args.include_executed_query)
    Path(args.out_json).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"narrative  → {args.out_md}")
    print(f"trace JSON → {args.out_json}")
    if mode == "gemini" and genai is None:
        print("Gemini selected but google-generativeai is not installed; fell back to template.")

if __name__ == "__main__":
    main()
