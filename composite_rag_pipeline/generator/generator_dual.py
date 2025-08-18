#!/usr/bin/env python3
# generator_dual.py — dual-mode story generator (KG / Hybrid)
#
# Features:
#  - FACTLETS: compact, deduped fact snippets (numbers/years prioritized)
#  - Inline citations:
#       * numeric "[1]" style, OR
#       * CQ-ID style "[CQ-E1]" / "[CQ-E1, CQ-L11]"  (default)
#  - Per-sentence citation enforcement (optional)
#  - URL snippet support for Hybrid (optional)
#  - Writes per-beat answer records (JSONL) and claims.jsonl (claim → evidence)
#  - Strips LLM meta lead-ins like "Here's the story section: [CQ-...]"
#  - NEW: also writes a *clean* story file without sections or citations (CQ-IDs / numeric)
#
# Examples:
#   KG:
#     python3 generator_dual.py \
#       --plan ../planner/plan_KG.json \
#       --plan_with_evidence ../retriever/plan_with_evidence_KG.json \
#       --kg_meta ../index/KG/cq_metadata.json \
#       --params ../params.json \
#       --llm_provider ollama --llm_model llama3.1-128k \
#       --include_citations --citation_style cqid \
#       --max_rows 6 --max_facts_per_beat 12 --beat_sentences 4 --context_budget_chars 1600 \
#       --enforce_citation_each_sentence \
#       --claims_out claims_KG.jsonl \
#       --out answers_KG.jsonl --story_out story_KG.md \
#       --story_clean_out story_KG_clean.md
#
#   Hybrid (with URL snippets):
#     python3 generator_dual.py \
#       --plan ../planner/plan_Hybrid.json \
#       --plan_with_evidence ../retriever/plan_with_evidence_Hybrid.json \
#       --hy_meta ../index/Hybrid/cq_metadata.json \
#       --params ../params.json \
#       --llm_provider ollama --llm_model llama3.1-128k \
#       --include_citations --citation_style cqid \
#       --use_url_content --max_url_snippets 3 --snippet_chars 400 \
#       --max_rows 6 --max_facts_per_beat 12 --beat_sentences 4 --context_budget_chars 1600 \
#       --enforce_citation_each_sentence \
#       --claims_out claims_Hybrid.jsonl \
#       --out answers_Hybrid.jsonl --story_out story_Hybrid.md \
#       --story_clean_out story_Hybrid_clean.md

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------- fact-density + citation helpers ----------------

_NUM_RX  = re.compile(r"\b\d[\d,\.]*\b")
_YEAR_RX = re.compile(r"\b(19|20)\d{2}\b")

# numeric: [12]
_CITE_NUM_RX = re.compile(r"\[(\d+)\]")
# cqid: [CQ-AAA], or multiple: [CQ-AAA, CQ-BBB]
_CITE_CQ_RX  = re.compile(r"\[(CQ-[A-Za-z0-9-]+(?:\s*,\s*CQ-[A-Za-z0-9-]+)*)\]")
# either (for detection)
_CITE_ANY_RX = re.compile(r"\[(?:\d+|CQ-[A-Za-z0-9-]+(?:\s*,\s*CQ-[A-Za-z0-9-]+)*)\]")
# for removal in "clean" variant
_CITE_ANY_BLOCK_RX = re.compile(r"\s*\[(?:\d+(?:\s*,\s*\d+)*|CQ-[A-Za-z0-9-]+(?:\s*,\s*CQ-[A-Za-z0-9-]+)*)\]")

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

# --- meta/intro stripper regexes ---
INTRO_DROP_RX  = re.compile(
    r"""^\s*
        (?:
          here\s+is|here's|this|below\s+is|the\s+following
        )\s+(?:the\s+)?(?:introduction|story(?:\s+section)?|section)
        (?:\s+based\s+on\s+the\s+provided\s+factlets\s+and\s+references)?    # optional fluff
        \s*:\s*
        (?:\[(?:CQ-[A-Za-z0-9-]+(?:\s*,\s*CQ-[A-Za-z0-9-]+)*)\])?            # optional CQ citation tail
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

META_PREFIX_RX = re.compile(
    r"""^\s*
        (?:
          here\s+is|here's|this|below\s+is|the\s+following
        )\s+.*?(?:introduction|story(?:\s+section)?|section)\s*:\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

ONLY_CITE_RX = re.compile(
    r"""^\s*
        \[
          (?:\d+(?:\s*,\s*\d+)*|
             CQ-[A-Za-z0-9-]+(?:\s*,\s*CQ-[A-Za-z0-9-]+)*
          )
        \]
        \s*$
    """,
    re.VERBOSE,
)

def _strip_meta_leadins(text: str) -> str:
    """Remove 'Here's the story...' style lead-in lines and prefixes."""
    out_lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        # Drop whole line if it's just a meta intro (optionally with citations)
        if INTRO_DROP_RX.match(s):
            continue
        # Remove meta prefix like "Here's the story section: "
        s2 = META_PREFIX_RX.sub("", s, count=1)
        # If what remains is empty or only a citation block, drop it
        if not s2 or ONLY_CITE_RX.match(s2):
            continue
        out_lines.append(s2)
    return "\n".join(out_lines).strip()

def _score_row_for_fact_density(row: dict) -> int:
    """Score a row for 'factuality' — numbers/years/keywords."""
    text = " ".join(str(v) for v in row.values() if isinstance(v, str))
    s = 0
    if _NUM_RX.search(text): s += 3
    if _YEAR_RX.search(text): s += 2
    if any(k in text.lower() for k in ["wembley","stadium","final","attendance","minute","setlist","world cup"]): s += 1
    return s

def _pack_factlets(rows: list, max_factlets: int) -> list[str]:
    """Compact, deduped 'factlets' the LLM can integrate verbatim."""
    seen = set(); factlets = []
    rows_sorted = sorted(rows or [], key=_score_row_for_fact_density, reverse=True)
    for r in rows_sorted:
        vals = [v for v in r.values() if isinstance(v, str) and v and not str(v).startswith("__")]
        if not vals:
            continue
        sent = re.sub(r"\s+", " ", " — ".join(vals))[:280].strip()
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        factlets.append(sent)
        if len(factlets) >= max_factlets:
            break
    return factlets

def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text: return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]

def _ensure_sentence_citations_numeric(text: str, fallback_idx: Optional[int]) -> str:
    out_lines = []
    for para in (text or "").split("\n"):
        if not para.strip():
            out_lines.append(para); continue
        sents = _split_sentences(para)
        fixed = []
        for s in sents:
            if _CITE_NUM_RX.search(s):
                fixed.append(s)
            else:
                fixed.append(s + (f" [{fallback_idx}]" if fallback_idx else ""))
        out_lines.append(" ".join(fixed))
    return "\n".join(out_lines).strip()

def _ensure_sentence_citations_cqid(text: str, fallback_cqid: Optional[str]) -> str:
    out_lines = []
    for para in (text or "").split("\n"):
        if not para.strip():
            out_lines.append(para); continue
        sents = _split_sentences(para)
        fixed = []
        for s in sents:
            if _CITE_CQ_RX.search(s):  # already has CQ-style cites
                fixed.append(s)
            elif _CITE_NUM_RX.search(s):  # has numeric, keep it
                fixed.append(s)
            else:
                fixed.append(s + (f" [{fallback_cqid}]" if fallback_cqid else ""))
        out_lines.append(" ".join(fixed))
    return "\n".join(out_lines).strip()

def _extract_citations_numeric(sent: str) -> List[int]:
    return [int(n) for n in _CITE_NUM_RX.findall(sent or "")]

def _extract_citations_cqid(sent: str) -> List[str]:
    tags: List[str] = []
    for block in _CITE_CQ_RX.findall(sent or ""):
        for tok in block.split(","):
            t = tok.strip()
            if t:
                tags.append(t)
    return tags

# ---------------- small I/O utils ----------------

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ---------------- LLM adapters ----------------

def _call_ollama(model: str, prompt: str, num_ctx: Optional[int] = None) -> str:
    """Use local Ollama HTTP API if available; else 'ollama run' CLI."""
    import json as _json
    import urllib.request as _url
    try:
        payload = {"model": model, "prompt": prompt, "stream": False}
        if num_ctx is not None:
            payload["options"] = {"num_ctx": int(num_ctx)}
        req = _url.Request(
            "http://localhost:11434/api/generate",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with _url.urlopen(req, timeout=300) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return (data.get("response") or "").strip()
    except Exception:
        pass
    # Fallback: CLI
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as fp:
        fp.write(prompt); fp.flush()
        args = ["ollama", "run", model, "--prompt", fp.name]
        out = subprocess.check_output(args, text=True)
        return out.strip()

def _call_gemini(model: str, prompt: str) -> str:
    """Google Generative AI (requires GOOGLE_API_KEY)."""
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        raise RuntimeError("Gemini provider selected but 'google-generativeai' package is not installed.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini provider selected but GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    r = model_obj.generate_content(prompt)
    return (getattr(r, "text", None) or "").strip()

def llm_generate(provider: str, model: str, prompt: str, ollama_num_ctx: Optional[int] = None) -> str:
    provider = (provider or "").lower()
    if provider == "ollama":
        return _call_ollama(model, prompt, num_ctx=ollama_num_ctx)
    if provider == "gemini":
        return _call_gemini(model, prompt)
    raise ValueError(f"Unknown llm_provider: {provider}")

# ---------------- references ----------------

def _row_to_ref_string(row: Dict[str, Any]) -> str:
    parts = []
    for k,v in row.items():
        if str(k).startswith("__"): continue
        if isinstance(v, str) and v:
            parts.append(f"{k}: {v}")
    s = "; ".join(parts)
    return re.sub(r"\s+", " ", s).strip()

def _normalize_refs_numeric(
    rows: List[Dict[str, Any]],
    url_infos: List[Dict[str, Any]],
    use_url_content: bool,
    max_url_snippets: int,
    snippet_chars: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Flat numeric reference list 1..N (legacy style)."""
    references: List[Dict[str, Any]] = []
    ref_lines: List[str] = []

    for r in rows:
        references.append({"type":"kg","row":r})
        ref_lines.append(f"{len(references)}. KG: " + _row_to_ref_string(r)[:220])

    seen = set(); web = []
    for info in url_infos:
        u = (info.get("url") or "").strip()
        if not u or u in seen:
            continue
        seen.add(u); web.append(info)
    web = web[:max_url_snippets]

    for info in web:
        title = (info.get("title") or "").strip()
        domain = (info.get("domain") or "").strip()
        url = (info.get("url") or "").strip()
        line = f"{len(references)+1}. WEB: {title or domain or url}"
        if domain: line += f" ({domain})"
        references.append({
            "type":"web","url":url,"title":title,"domain":domain,
            "content_sha1": info.get("content_sha1") or "",
            "content_text": (info.get("content_text") or ""),
        })
        if use_url_content:
            snip = (info.get("content_text") or "")[:snippet_chars].strip()
            if snip: line += f" — “{snip}”"
        ref_lines.append(line[:320])

    return references, ref_lines

def _normalize_refs_by_cq(
    rows_by_cq: Dict[str, List[Dict[str, Any]]],
    web_by_cq: Dict[str, List[Dict[str, Any]]],
    use_url_content: bool,
    max_url_snippets: int,
    snippet_chars: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str], List[str]]:
    """
    Returns:
      ref_by_cq: {cq_id: [evidence dicts]}
      ref_lines: lines for prompt like "CQ-E1: KG: …" and "CQ-E1: WEB: …"
      ordered_cqids: order used in prompt
    """
    ref_by_cq: Dict[str, List[Dict[str, Any]]] = {}
    ref_lines: List[str] = []
    ordered: List[str] = []

    cqids = sorted(set(rows_by_cq.keys()) | set(web_by_cq.keys()))
    for cq in cqids:
        ordered.append(cq)
        bundle: List[Dict[str, Any]] = []

        # KG rows
        for r in rows_by_cq.get(cq, []):
            bundle.append({"type":"kg","row":r})
            ref_lines.append(f"{cq}: KG: " + _row_to_ref_string(r)[:220])

        # Web (dedupe by URL, clamp)
        seen = set(); web = []
        for info in web_by_cq.get(cq, []):
            u = (info.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u); web.append(info)
        web = web[:max_url_snippets]

        for info in web:
            title = (info.get("title") or "").strip()
            domain = (info.get("domain") or "").strip()
            url = (info.get("url") or "").strip()
            line = f"{cq}: WEB: {title or domain or url}"
            if domain: line += f" ({domain})"
            ev = {
                "type":"web","url":url,"title":title,"domain":domain,
                "content_sha1": info.get("content_sha1") or "",
                "content_text": (info.get("content_text") or ""),
            }
            if use_url_content:
                snip = (info.get("content_text") or "")[:snippet_chars].strip()
                if snip: line += f" — “{snip}”"
            ref_lines.append(line[:320])
            bundle.append(ev)

        ref_by_cq[cq] = bundle

    return ref_by_cq, ref_lines, ordered

# ---------------- prompt ----------------


# prompt_builder.py
import yaml

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

def persona_block(pack: dict) -> str:
    tone = ", ".join(pack["tone"])
    dos = "\n  - " + "\n  - ".join(pack["dos"]) if pack["dos"] else ""
    donts = "\n  - " + "\n  - ".join(pack["donts"]) if pack["donts"] else ""
    return (
        "Audience (write for them; do NOT roleplay):\n"
        f"- Description: {pack['description']}\n"
        f"- Tone/style: {tone}\n"
        f"- Reading level: {pack['reading_level']}\n"
        f"- Dos:{dos}\n"
        f"- Don’ts:{donts}\n"
    )

def build_instruction(persona_name: str, beat_idx: int, beat_title: str, n_factlets: int, path="personas.yaml"):
    p = load_persona_pack(persona_name, path)
    pb = persona_block(p)

    # compute coverage floor
    min_f = max(p["coverage"]["min_factlets"], int(round(p["coverage"]["min_pct"] * max(1, n_factlets))))
    lo, hi = p["length_words"]

    return [
        "You are writing a factual, engaging story section. Do NOT roleplay as the audience.",
        f"Section context — Beat {beat_idx + 1}: {beat_title}",
        "",
        pb,  # persona description block
        "Use the FACTLETS and REFERENCES provided below.",
        "- Faithfully incorporate as many FACTLETS as possible; you may paraphrase, merge, and condense while preserving meaning.",
        f"- Coverage target: use at least {min_f} distinct FACTLETS (or all if fewer). "
        + ("Aim for breadth across people/place/time/action/impact." if p["coverage"]["require_breadth_buckets"] else ""),
        "- Citations: After factual clauses, add bracketed IDs (e.g., “[CQ-E12b; CQ-L28]”). Multiple IDs per sentence are fine.",
        f"- Length: aim for {lo}–{hi} words.",
        "Narrate in third person; do not use first person or speak as the audience.",
        "Do NOT write meta lead-ins—start directly with the narrative.",
        "If relevant FACTLETS remain, append one line: “Also note: …” with citations.",
        "At the end, append: Used: [IDs]  Unused: [IDs]",
    ]


from typing import List, Dict, Tuple, Union
import textwrap

RefItem = Dict[str, str]  # expects keys like: id (optional), title, url, domain, type, snippet (optional)

def format_references(
    refs: Union[List[str], List[RefItem]],
    style: str = "cqid"  # "cqid" | "numeric"
) -> Tuple[str, Dict[str, str]]:
    """
    Returns (refs_block_text, id_map)
    - If refs are strings, passthrough (id_map is empty).
    - If refs are dicts, renders lines and returns id_map:
        * style="cqid": original 'id' (e.g., "CQ-E12") is used as the cite token
        * style="numeric": assigns "1","2",...; id_map maps original -> number
    Line format favors (Domain) Title — URL; includes type and short snippet if present.
    """
    if not refs:
        return "REFERENCES: (none)", {}

    # Case 1: already prepared lines
    if isinstance(refs[0], str):
        lines = "\n".join(refs)
        return f"REFERENCES:\n{lines}", {}

    # Case 2: dict refs
    items: List[RefItem] = refs  # type: ignore
    id_map: Dict[str, str] = {}
    rendered_lines: List[str] = []

    if style == "numeric":
        for i, r in enumerate(items, 1):
            display_id = str(i)
            orig = r.get("id") or f"ref-{i}"
            id_map[orig] = display_id
            line = f"[{display_id}] ({r.get('domain','')}) {r.get('title','Untitled')} — {r.get('url','')}"
            if r.get("type"):
                line += f"  <{r['type']}>"
            if r.get("snippet"):
                line += f"\n    {textwrap.shorten(r['snippet'], width=180)}"
            rendered_lines.append(line)
    else:  # cqid
        for r in items:
            cq = r.get("id") or ""
            display_id = cq if cq else r.get("title","ref").replace(" ", "")[:8]
            if cq:
                id_map[cq] = display_id
            line = f"[{display_id}] ({r.get('domain','')}) {r.get('title','Untitled')} — {r.get('url','')}"
            if r.get("type"):
                line += f"  <{r['type']}>"
            if r.get("snippet"):
                line += f"\n    {textwrap.shorten(r['snippet'], width=180)}"
            rendered_lines.append(line)

    return "REFERENCES:\n" + "\n".join(rendered_lines), id_map

import textwrap

def number_references(refs):
    """
    refs: List[dict] with keys: id, title, url, domain, type ("kg" or "web")
    Returns (id_map, lines, items):
      id_map: original_id -> "1"/"2"/...
      lines: human-readable numbered references
      items: sorted refs with their assigned numbers (for logging)
    """
    def sort_key(r):
        # keep KG first if you prefer; then domain/title for determinism
        return (0 if (r.get("type") or "").lower()=="kg" else 1,
                r.get("domain",""),
                r.get("title",""))
    items = sorted(refs, key=sort_key)
    id_map, lines = {}, []
    for i, r in enumerate(items, 1):
        orig = r.get("id") or f"ref-{i}"
        id_map[orig] = str(i)
        lines.append(f"[{i}] ({r.get('domain','')}) {r.get('title','Untitled')} — {r.get('url','')}")
    return id_map, lines, items


def build_prompt(
    persona_description: str,              # ← pass description, not just a name
    beat_idx: int,
    beat_title: str,
    factlets: List[str],
    refs: Union[List[str], List[Dict[str,str]]],  # ← either old lines or structured dicts
    include_citations: bool,
    beat_words: Tuple[int,int] = (180, 260),
    citation_style: str = "cqid"          # "cqid" or "numeric"
) -> str:
    # 1) References block (and optional ID remapping)
    refs_block, id_map = format_references(refs, style=citation_style)

    # 2) Instruction header (coverage + citations clarified)
    lo, hi = beat_words
    instruction = [
        "You are writing a factual, engaging story section. Do NOT roleplay as the audience.",
        f"Section context — Beat {beat_idx + 1}: {beat_title}",
        "",
        "Audience (write for them; do NOT roleplay):",
        f"- Description: {persona_description}",
        "- Tone/style: clear, precise, evidence-driven",
        "- Dos: lead with outcomes; use named entities, dates, and numbers; keep sentences tight",
        "- Don’ts: no first person, no speculation, no meta lead-ins",
        "",
        "Use the FACTLETS and REFERENCES provided below.",
        "- Faithfully incorporate as many FACTLETS as possible; you may paraphrase, merge, and condense while preserving meaning.",
        "- Coverage target: use at least 70% of the FACTLETS (or all if fewer). Prefer breadth (people/place/time/action/impact) when available.",
        f"- Length: aim for {lo}–{hi} words.",
        "Narrate in third person; do not speak as the audience.",
        "Do NOT write meta lead-ins like “Here is the introduction...”—start directly.",
    ]

    if include_citations:
        if citation_style == "cqid":
            instruction.append(
                "Citations: After factual clauses, add bracketed CQ IDs like [CQ-E1] or [CQ-E1; CQ-L11]. "
                "Combine multiple IDs when a sentence uses multiple items."
            )
        else:
            instruction.append(
                "Citations: After factual clauses, add numeric brackets like [1] or [1; 3]. "
                "Numbers refer to the REFERENCES list below."
            )

    # 3) Blocks
    header = "\n".join(instruction)
    fact_block = "FACTLETS:\n" + "\n".join(f"- {f}" for f in factlets) if factlets else "FACTLETS: (none)"

    return textwrap.dedent(f"""\
    {header}

    {fact_block}

    {refs_block}

    Now write the story section.
    """).strip()


# ---------------- generator core ----------------

@dataclass
class Args:
    plan: str
    plan_with_evidence: str
    kg_meta: Optional[str] = None
    hy_meta: Optional[str] = None
    params: Optional[str] = None
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1-128k"
    ollama_num_ctx: Optional[int] = None
    use_url_content: bool = False
    max_url_snippets: int = 2
    snippet_chars: int = 400
    include_citations: bool = False
    max_rows: int = 4
    max_facts_per_beat: int = 8
    beat_sentences: int = 3
    context_budget_chars: int = 1200
    enforce_citation_each_sentence: bool = False
    citation_style: str = "cqid"  # default to CQ-ID citations
    claims_out: Optional[str] = None
    out: str = "answers.jsonl"
    story_out: str = "story.md"
    story_clean_out: Optional[str] = None  # NEW — write a clean story (no sections, no citations)

def _trim_to_budget(lines: List[str], budget: int) -> List[str]:
    out = []; sofar = 0
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        n = len(ln) + 1
        if sofar + n > budget: break
        out.append(ln); sofar += n
    return out

def _clean_story_text_remove_sections_and_citations(text_md: str) -> str:
    """
    Produce a 'clean' story:
      - remove all markdown headings (lines starting with '#')
      - remove inline citations [1], [12], [CQ-XXX], [CQ-XXX, CQ-YYY]
      - normalize spacing and blank lines
    """
    lines = []
    for raw in (text_md or "").splitlines():
        if raw.lstrip().startswith("#"):
            continue  # drop headings like "## Beat ..."
        s = _CITE_ANY_BLOCK_RX.sub("", raw)  # remove any bracketed citation blocks
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)  # trim space before punctuation
        s = re.sub(r"\s{2,}", " ", s).strip()
        if s:
            lines.append(s)
        else:
            # preserve a blank line boundary (will be collapsed later)
            lines.append("")
    # collapse multiple blank lines
    out = []
    prev_blank = False
    for ln in lines:
        if not ln:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(ln)
            prev_blank = False
    text = "\n".join(out).strip() + "\n"
    return text

# prompt_logging.py
import hashlib, json, os, socket, subprocess, time
from pathlib import Path
from typing import Dict, Any

def prompt_hash(prompt: str, extra: Dict[str, Any] | None = None) -> str:
    """Stable short id for the prompt (+ any key knobs)."""
    m = hashlib.sha256()
    m.update(prompt.strip().encode("utf-8"))
    if extra:
        # Only stable keys affect the hash:
        keys = ["citation_style", "length_words", "min_factlets", "coverage_target"]
        payload = {k: extra.get(k) for k in keys if k in extra}
        m.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return m.hexdigest()[:12]

def git_commit_short() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return None

def log_prompt(record: Dict[str, Any], out_path: str | Path = "outputs/prompts_log.jsonl") -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def make_prompt_record(
    prompt_text: str,
    meta: Dict[str, Any],
    model: str,
    temperature: float,
    top_p: float,
    run_id: str | None = None,
) -> Dict[str, Any]:
    h = prompt_hash(prompt_text, meta)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "git_commit": git_commit_short(),
        "run_id": run_id or str(int(time.time())),
        "prompt_version": h,
        "prompt_text": prompt_text,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }
    rec.update(meta)  # persona, beat info, knobs
    return rec


def generate(
    *,
    mode: str,
    plan: Dict[str, Any],
    plan_with_evidence: Dict[str, Any],
    meta_path: str,
    params: Dict[str, Any],
    llm_provider: str,
    llm_model: str,
    ollama_num_ctx: Optional[int] = None,
    use_url_content: bool = False,
    max_url_snippets: int = 2,
    snippet_chars: int = 400,
    include_citations: bool = False,
    max_rows: int = 4,
    max_facts_per_beat: int = 8,
    beat_sentences: int = 3,
    context_budget_chars: int = 1200,
    enforce_citation_each_sentence: bool = False,
    citation_style: str = "cqid",
    claims_out: Optional[str] = None,
    story_clean_out: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:

    persona = plan.get("persona") or "Narrator"
    beats = plan.get("beats") or []
    items = plan_with_evidence.get("items") or []

    # group items by beat index
    by_beat: Dict[int, List[Dict[str, Any]]] = {}
    for it in items:
        b = it.get("beat") or {}
        idx = int(b.get("index", 0))
        by_beat.setdefault(idx, []).append(it)

    answers_stream: List[Dict[str, Any]] = []
    story_lines: List[str] = []
    claims_path = Path(claims_out) if claims_out else None

    for b in beats:
        beat_idx = int(b.get("index", 0))
        beat_title = b.get("title") or "Untitled"
        beat_items = by_beat.get(beat_idx, [])

        # pool rows + url infos and keep cq_id provenance
        rows_pool: List[Dict[str, Any]] = []
        web_infos_pool: List[Dict[str, Any]] = []

        rows_by_cq: Dict[str, List[Dict[str, Any]]] = {}
        web_by_cq: Dict[str, List[Dict[str, Any]]] = {}

        for it in beat_items:
            cqid = it.get("id") or "CQ-UNK"
            for r in (it.get("rows") or []):
                r2 = dict(r); r2["__cq_id"] = cqid
                rows_pool.append(r2)
                rows_by_cq.setdefault(cqid, []).append(r2)
                for info in r.get("__url_info", []) or []:
                    info2 = dict(info); info2["__cq_id"] = cqid
                    web_infos_pool.append(info2)
                    web_by_cq.setdefault(cqid, []).append(info2)
            for info in it.get("url_info", []) or []:
                info2 = dict(info); info2["__cq_id"] = cqid
                web_infos_pool.append(info2)
                web_by_cq.setdefault(cqid, []).append(info2)
            print(f"Beat {beat_idx+1} ({beat_title}): {len(rows_pool)} rows, {len(web_infos_pool)} web infos")

        # factlets
        rows_selected = sorted(rows_pool, key=_score_row_for_fact_density, reverse=True)
        factlets = _pack_factlets(rows_selected, max_facts_per_beat)

        # references
        if citation_style == "cqid":
            ref_by_cq, ref_lines_all, ordered_cqids = _normalize_refs_by_cq(
                rows_by_cq, web_by_cq, use_url_content, max_url_snippets, snippet_chars
            )
            ref_lines = _trim_to_budget(ref_lines_all, context_budget_chars)
            # keep only CQIDs that survived trimming
            kept_cqs = {ln.split(":")[0].strip() for ln in ref_lines if ":" in ln}
            ref_by_cq_kept = {cid: ref_by_cq[cid] for cid in kept_cqs if cid in ref_by_cq}
        else:
            references, ref_lines_all = _normalize_refs_numeric(
                rows_selected, web_infos_pool, use_url_content, max_url_snippets, snippet_chars
            )
            ref_lines = _trim_to_budget(ref_lines_all, context_budget_chars)
            if len(ref_lines) < len(references):
                references = references[:len(ref_lines)]

        persona_name = persona# or "Luca"
        pack = load_persona_pack(persona_name, path="config/personas.yaml")
        persona_desc = pack["description"]
        # prompt

        prompt = build_prompt(
            persona_description=persona_desc,
            beat_idx=beat_idx,
            beat_title=beat_title,
            factlets=factlets,
            refs=ref_lines,  # can be `ref_lines` OR `refs` dicts
            include_citations=True,
            beat_words=(180, 260),  # target length window
            citation_style="cqid",  # "cqid" | "numeric"
        )

        meta = {
            "persona": persona,
            "beat_index": beat_idx,
            "beat_title": beat_title,
            "citation_style": "cqid",
            "length_words": pack["length_words"],
            "min_factlets": max(3, int(round(0.7 * len(factlets)))),
            "coverage_target": 0.7,

        }

        # Log the prompt
        rec = make_prompt_record(
            prompt_text=prompt,
            meta=meta,
            model="gpt-X",  # fill with your model id
            temperature=0.5,
            top_p=1.0,
            run_id=run_id,  # if you have one for the whole story
        )
        log_prompt(rec, "outputs/prompts_log.jsonl")

        #print(f"prompt : {prompt}")

        text = llm_generate(llm_provider, llm_model, prompt, ollama_num_ctx=ollama_num_ctx)
        #text = prompt

        # enforce per-sentence citation if requested
        if enforce_citation_each_sentence:
            if citation_style == "cqid":
                # pick a fallback CQID with any evidence (first line's CQ)
                fallback_cqid = None
                for ln in ref_lines:
                    if ":" in ln:
                        fallback_cqid = ln.split(":")[0].strip()
                        if fallback_cqid:
                            break
                text = _ensure_sentence_citations_cqid(text, fallback_cqid)
            else:
                last_idx = len(ref_lines)  # numeric fallback
                text = _ensure_sentence_citations_numeric(text, last_idx if last_idx>0 else None)

        # strip meta lead-in filler lines/prefixes
        text = _strip_meta_leadins(text)

        # story aggregation (sectioned)
        story_lines.append(f"## {beat_title}\n\n{text}\n")

        # answers stream (debug)
        if citation_style == "cqid":
            answers_stream.append({
                "beat_index": beat_idx, "beat_title": beat_title,
                "facts_used": len(factlets),
                "references_by_cq": ref_by_cq_kept,
                "text": text,
            })
        else:
            answers_stream.append({
                "beat_index": beat_idx, "beat_title": beat_title,
                "facts_used": len(factlets),
                "references": references,
                "text": text,
            })

        # claims emission
        if claims_path:
            sents = _split_sentences(text)
            for s in sents:
                if not _CITE_ANY_RX.search(s):
                    continue
                if citation_style == "cqid":
                    tags = _extract_citations_cqid(s)
                    evid: List[Dict[str, Any]] = []
                    for tag in tags:
                        evid.extend(ref_by_cq_kept.get(tag, []))
                    _append_jsonl(claims_path, {
                        "mode": mode,
                        "beat_index": beat_idx,
                        "beat_title": beat_title,
                        "cq_id": None,
                        "sentence": s,
                        "citations": tags,
                        "evidence": evid,
                    })
                else:
                    nums = _extract_citations_numeric(s)
                    evid = []
                    for n in nums:
                        i = n-1
                        if i>=0 and i < len(references):
                            evid.append(references[i])
                    _append_jsonl(claims_path, {
                        "mode": mode,
                        "beat_index": beat_idx,
                        "beat_title": beat_title,
                        "cq_id": None,
                        "sentence": s,
                        "citations": nums,
                        "evidence": evid,
                    })

    story_md = "\n".join(story_lines).strip() + "\n"

    # Also produce a clean, unsectioned, uncited variant if requested
    if story_clean_out:
        clean_text = _clean_story_text_remove_sections_and_citations(story_md)
        Path(story_clean_out).write_text(clean_text, encoding="utf-8")

    return story_md, answers_stream

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Dual-mode story generator with numeric or CQ-ID citations, meta-leadin stripping, claims writer, and clean story output")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--plan_with_evidence", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--kg_meta")
    group.add_argument("--hy_meta")
    ap.add_argument("--params", required=True)

    ap.add_argument("--llm_provider", default="ollama", choices=["ollama","gemini"])
    ap.add_argument("--llm_model", default="llama3.1-128k")
    ap.add_argument("--ollama_num_ctx", type=int, default=None)

    ap.add_argument("--use_url_content", action="store_true")
    ap.add_argument("--max_url_snippets", type=int, default=2)
    ap.add_argument("--snippet_chars", type=int, default=400)

    ap.add_argument("--include_citations", action="store_true")
    ap.add_argument("--citation_style", choices=["numeric","cqid"], default="cqid")

    ap.add_argument("--max_rows", type=int, default=4)
    ap.add_argument("--max_facts_per_beat", type=int, default=8)
    ap.add_argument("--beat_sentences", type=int, default=3)
    ap.add_argument("--context_budget_chars", type=int, default=1200)

    ap.add_argument("--enforce_citation_each_sentence", action="store_true")
    ap.add_argument("--claims_out", default=None)

    ap.add_argument("--out", default="answers.jsonl")
    ap.add_argument("--story_out", default="story.md")
    ap.add_argument("--story_clean_out", default=None, help="Optional path to write a clean story (no sections, no citations)")
    args = ap.parse_args()

    plan = _load_json(Path(args.plan))
    plan_ev = _load_json(Path(args.plan_with_evidence))
    meta_path = args.kg_meta or args.hy_meta
    if meta_path:
        _ = _load_json(Path(meta_path))  # retained for compatibility/reference
    params = _load_json(Path(args.params)) if args.params else {}

    mode = plan.get("mode") or ("KG" if args.kg_meta else "Hybrid")

    story_md, answers = generate(
        mode=mode,
        plan=plan,
        plan_with_evidence=plan_ev,
        meta_path=meta_path or "",
        params=params,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_num_ctx=args.ollama_num_ctx,
        use_url_content=args.use_url_content,
        max_url_snippets=args.max_url_snippets,
        snippet_chars=args.snippet_chars,
        include_citations=args.include_citations,
        max_rows=args.max_rows,
        max_facts_per_beat=args.max_facts_per_beat,
        beat_sentences=args.beat_sentences,
        context_budget_chars=args.context_budget_chars,
        enforce_citation_each_sentence=args.enforce_citation_each_sentence,
        citation_style=args.citation_style,
        claims_out=args.claims_out,
        story_clean_out=args.story_clean_out,
    )

    Path(args.story_out).write_text(story_md, encoding="utf-8")
    with Path(args.out).open("w", encoding="utf-8") as fp:
        for rec in answers:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"story → {args.story_out}")
    if args.story_clean_out:
        print(f"story_clean → {args.story_clean_out}")
    print(f"answers → {args.out}")
    if args.claims_out:
        print(f"claims → {args.claims_out}")

if __name__ == "__main__":
    main()
