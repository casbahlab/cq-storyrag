from __future__ import annotations
from typing import Iterable, Tuple, Dict, List
import re

_WORDS = re.compile(r"[A-Za-z0-9']+")
BRACKETED_CQ = re.compile(r"\s*\[(?:CQ-[A-Za-z]+[0-9]+(?:\s*;\s*CQ-[A-Za-z]+[0-9]+)*)\]\s*")
RAW_CQ_TOKEN = re.compile(r"\bCQ-[A-Za-z]+[0-9]+\b")

def _tok(s: str) -> set:
    return set(w.lower() for w in _WORDS.findall(s or ""))

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def _strip_ids(text: str) -> str:
    # remove URLs, angle-bracket IRIs, ex: prefixes, CQ brackets/tokens
    t = text or ""
    t = re.sub(r"https?://\S+|<[^>]+>", "", t)
    t = re.sub(r"\bex:", "", t)
    t = BRACKETED_CQ.sub("", t)
    t = RAW_CQ_TOKEN.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _dedup_near(lines: List[str], jaccard: float = 0.80) -> List[str]:
    kept, toks = [], []
    for ln in lines:
        s = _strip_ids(ln)
        t = _tok(s)
        if any(_jaccard(t, k) >= jaccard for k in toks):  # near-dup
            continue
        kept.append(s)
        toks.append(t)
    return kept

def _length_target(length: str) -> int:
    key = (length or "Medium").strip().lower()
    return 8 if key == "short" else 14 if key == "medium" else 22

def build_context_block(
    items: Iterable[str],
    length: str,
    prefix: str = "C",
    label: str = "CONTEXT (numbered)",
) -> Tuple[str, Dict[str, str]]:
    """
    Collapse many inputs (factlets/snippets/triples) into ONE numbered CONTEXT block.
    Returns (block_text, id_to_text_map) where ids are C1..Cn.
    """
    L = [s for s in (items or []) if s and s.strip()]
    L = _dedup_near(L, jaccard=0.80)
    # feed the model a bit more than needed; it will compress
    max_items = _length_target(length)
    L = L[:max_items]
    id_map: Dict[str,str] = {}
    lines: List[str] = []
    for i, s in enumerate(L, start=1):
        cid = f"{prefix}{i}"
        id_map[cid] = s
        lines.append(f"{cid}: {s}")
    block = f"{label}\n" + "\n".join(lines) if lines else f"{label}\n(none)"
    return block, id_map
