#!/usr/bin/env python3
# generator/eval_claims.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, List

SENT_RX = re.compile(r'(?<=\.|\?|!)\s+(?=[A-Z0-9])')  # simple sentence split
CITE_RX = re.compile(r"\[(\d+)\]")

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # keep paragraphs, split inside each
    paras = [p for p in text.split("\n") if p.strip()]
    out: List[str] = []
    for p in paras:
        parts = SENT_RX.split(p.strip())
        for s in parts:
            s = s.strip()
            if s:
                out.append(s)
    return out

def extract_citations(sent: str) -> List[int]:
    return [int(n) for n in CITE_RX.findall(sent or "")]

def normalize_references(evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turn per-beat evidence_docs into a flat, numbered reference list.
    If your generator already has 'references', pass that instead and skip this.
    """
    refs: List[Dict[str, Any]] = []
    for ev in evidence_docs or []:
        if "url" in ev or ev.get("type") == "web":
            refs.append({
                "type": "web",
                "url": ev.get("url") or ev.get("source_url") or "",
                "title": ev.get("title") or ev.get("source_title") or "",
                "domain": ev.get("domain") or "",
                "content_sha1": ev.get("content_sha1") or "",
                "content_excerpt": (ev.get("content_text") or ev.get("snippet") or "")[:400],
            })
        else:
            refs.append({
                "type": "kg",
                "row": ev.get("row") or ev.get("values") or ev,
                "executed_query": ev.get("executed_query",""),
                "label": ev.get("label") or "",
            })
    return refs

def emit_claims_jsonl(
    *,
    story_text: str,
    references: List[Dict[str, Any]],
    mode: str,
    beat_index: int,
    beat_title: str,
    cq_id: str | None,
    claims_out_path: Path,
) -> int:
    """
    Parse story_text, pull sentences + citations, map to 'references' by index,
    and append claim→evidence records. Returns count emitted.
    """
    claims_out_path.parent.mkdir(parents=True, exist_ok=True)
    num = 0
    sents = split_sentences(story_text)
    with claims_out_path.open("a", encoding="utf-8") as fp:
        for sent in sents:
            cites = extract_citations(sent)
            if not cites:
                continue  # only keep evidence-backed claims
            evid: List[Dict[str, Any]] = []
            for n in cites:
                i = n - 1
                if 0 <= i < len(references):
                    evid.append(references[i])
            rec = {
                "mode": mode,
                "beat_index": beat_index,
                "beat_title": beat_title,
                "cq_id": cq_id,
                "sentence": sent,
                "citations": cites,
                "evidence": evid,
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num += 1
    return num

def ensure_sentence_citations(text: str, fallback_cite: int | None) -> str:
    """
    If a sentence has no [n], append [fallback] (e.g., last ref in the beat).
    Use only when you explicitly enable strict mode.
    """
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append(para); continue
        sents = split_sentences(para)
        fixed = []
        for s in sents:
            if extract_citations(s):
                fixed.append(s)
            else:
                fixed.append(s + (f" [{fallback_cite}]" if fallback_cite else ""))
        lines.append(" ".join(fixed))
    return "\n".join(lines).strip()
