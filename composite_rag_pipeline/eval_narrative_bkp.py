#!/usr/bin/env python3
# eval_narrative.py
# Minimal, dependency-light evaluator for narrative quality per beat.
# Reads a combined JSONL (one object per beat per run) and writes CSV/HTML reports.

import argparse, json, math, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd

SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+')
WORD_RE = re.compile(r"[A-Za-z0-9']+")
BRACKET = re.compile(r"\[(.*?)\]")             # matches [ ... ] (citations)
NUMERIC_RE = re.compile(r"\b\d[\d,\.]*\b")     # crude numeric detector
ROLEPLAY_RE = re.compile(r"\b(I|we|as a[n]?|my|our)\b", re.I)
META_RE = re.compile(r"^(here is|in this section|the following)", re.I | re.M)

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def tokenize_words(s): return WORD_RE.findall(s or "")
def split_sentences(s): return [x for x in SENT_SPLIT.split((s or "").strip()) if x]

def syllable_estimate(word):
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w: return 0
    vowels = "aeiouy"
    count, prev = 0, False
    for ch in w:
        is_v = ch in vowels
        if is_v and not prev: count += 1
        prev = is_v
    if w.endswith("e") and count > 1: count -= 1
    return max(1, count)

def flesch_kincaid(text):
    words = tokenize_words(text)
    sents = split_sentences(text)
    if not words or not sents:
        return {"fk_grade": 0.0, "fk_reading_ease": 0.0}
    syllables = sum(syllable_estimate(w) for w in words)
    W, S = len(words), len(sents)
    fk_grade = 0.39*(W/S) + 11.8*(syllables/W) - 15.59
    fre = 206.835 - 1.015*(W/S) - 84.6*(syllables/W)
    return {"fk_grade": round(fk_grade,2), "fk_reading_ease": round(fre,2)}

def extract_citation_tokens(text: str):
    tokens = []
    for m in BRACKET.finditer(text or ""):
        inside = m.group(1)
        parts = re.split(r'[;,]\s*', inside)
        for p in parts:
            t = p.strip()
            if t:
                tokens.append(t)
    return tokens

def valid_id_set(references_by_cq: dict):
    ids = set()
    for lst in (references_by_cq or {}).values():
        for it in (lst or []):
            rid = it.get("id") or it.get("cqid")
            if rid: ids.add(str(rid))
    return ids

def evidence_word_counts(references_by_cq: dict):
    ev_words, web, kg = 0, 0, 0
    for lst in (references_by_cq or {}).values():
        for it in (lst or []):
            t = it.get("content_text") or it.get("text") or ""
            if isinstance(t, str): ev_words += len(t.split())
            typ = (it.get("type") or "").lower()
            if typ == "web": web += 1
            elif typ == "kg": kg += 1
    return ev_words, web, kg

def entity_tokens(sentence: str):
    # crude entity proxy: capitalized tokens > 2 chars (skip common)
    stops = {"the","and","for","with","from","that","this","were","which","while","when","but"}
    toks = WORD_RE.findall(sentence or "")
    ents = [t for t in toks if t[:1].isupper() and len(t) > 2 and t.lower() not in stops]
    return set(ents)

def entity_continuity(sentences):
    if len(sentences) < 2: return 0.0
    overlaps, pairs = 0, 0
    prev_ents = entity_tokens(sentences[0])
    for s in sentences[1:]:
        ents = entity_tokens(s)
        if prev_ents or ents:
            inter = prev_ents.intersection(ents)
            union = prev_ents.union(ents)
            if union:
                j = len(inter)/len(union)
                if j >= 0.1: overlaps += 1
            pairs += 1
        prev_ents = ents
    return overlaps/pairs if pairs else 0.0

def parse_used_footer(text: str):
    # expects "Used: [A, B]  Unused: [C, D]" (robust to spaces)
    used = re.search(r"Used:\s*\[([^\]]*)\]", text or "", re.I)
    unused = re.search(r"Unused:\s*\[([^\]]*)\]", text or "", re.I)
    def split_ids(m):
        if not m: return set()
        inside = m.group(1).strip()
        if not inside: return set()
        return set([x.strip() for x in re.split(r"[,\s]+", inside) if x.strip()])
    return split_ids(used), split_ids(unused)

import re
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any
import pandas as pd

WORD = re.compile(r"[A-Za-z0-9']+")

def _tokens(s: str) -> List[str]:
    return WORD.findall((s or "").lower())

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

def _sim(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    # blend Jaccard (vocabulary overlap) + SequenceMatcher (order/phrasing)
    return 0.6 * _jaccard(ta, tb) + 0.4 * SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()

def _split_sents(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

def _refs_to_items(refs: Dict[str, Dict[str, Any]]) -> List[str]:
    """Flatten plan/reference dicts to short surface strings (no URLs/IDs)."""
    items: List[str] = []
    for _id, r in (refs or {}).items():
        if isinstance(r, dict):
            for k in ("text", "snippet", "title", "description", "name", "value"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    items.append(v.strip())
                    break
        elif isinstance(r, str) and r.strip():
            items.append(r.strip())
    return items

import re
import numpy as np
from typing import List, Dict, Any, Iterable, Tuple

# --- helpers (names kept to match your code) ---
def normalize_text_coverage(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def tokenize_words_coverage(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_text_coverage(s))

def jaccard_coverage(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def contains_ngram(line_tokens: List[str], text_tokens: List[str], n: int = 5) -> bool:
    if len(line_tokens) < n:
        n = max(1, len(line_tokens))
    return bool(set(_ngrams(line_tokens, n)) & set(_ngrams(text_tokens, n)))

def _split_sentences_basic(text: str) -> List[str]:
    # Lightweight sentence split that works without extra deps
    # Keeps punctuation with the sentence
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]

# --- main: returns exactly the fields you expect ---
def context_coverage(
    text: str,
    context_lines: List[str],
    jaccard_coverage_threshold: float = 0.5,
    ngram_n: int = 5
) -> Dict[str, Any]:
    """
    Returns:
      {
        "support_rate": supported / max(1, len(sents)),
        "coverage_rate": len(covered) / max(1, len(items)),
        "sentence_support": sent_hits,
        # plus a few helpful extras
        "covered_mask": [...],
        "uncovered_examples": [...]
      }
    where:
      - supported = number of sentences in 'text' that are supported by at least one context line
      - covered    = list of indices of context lines that appear in the text by exact/ngram/jaccard
      - items      = context_lines
      - sent_hits  = per-sentence hit counts (how many context lines matched each sentence)
    """
    text = text or ""
    items = context_lines or []

    # Prepare sentences and global tokens once
    sents = _split_sentences_basic(text)
    sents_norm = [normalize_text_coverage(s) for s in sents]
    sents_toks = [tokenize_words_coverage(s) for s in sents_norm]

    text_norm = normalize_text_coverage(text)
    text_tokens = tokenize_words_coverage(text_norm)

    covered_mask: List[bool] = []
    covered_indices: List[int] = []
    uncovered_examples: List[str] = []

    # Track per-sentence hits
    sent_hits = [0] * len(sents)

    for i, raw_line in enumerate(items):
        line = normalize_text_coverage(raw_line)
        if not line:
            covered_mask.append(False)
            uncovered_examples.append(raw_line)
            continue

        ltoks = tokenize_words_coverage(line)

        # Global text match
        matched = (
            (line in text_norm) or
            contains_ngram(ltoks, text_tokens, n=ngram_n) or
            (jaccard_coverage(ltoks, text_tokens) >= jaccard_coverage_threshold)
        )

        if not matched:
            # Try sentence-local match to give credit even if global failed
            for stoks in sents_toks:
                if contains_ngram(ltoks, stoks, n=ngram_n) or jaccard_coverage(ltoks, stoks) >= jaccard_coverage_threshold:
                    matched = True
                    break

        covered_mask.append(bool(matched))
        if matched:
            covered_indices.append(i)
            # Attribute the hit to the best matching sentence
            best_sent = -1
            best_score = -1.0
            for idx, stoks in enumerate(sents_toks):
                # simple score: max of ngram overlap or jaccard
                ngram_ok = contains_ngram(ltoks, stoks, n=ngram_n)
                jac = jaccard_coverage(ltoks, stoks)
                score = max(jac, 1.0 if ngram_ok else 0.0)
                if score > best_score:
                    best_score = score
                    best_sent = idx
            if best_sent >= 0:
                sent_hits[best_sent] += 1
        else:
            uncovered_examples.append(raw_line)

    # A sentence is "supported" if it has at least one matched context line
    supported = sum(1 for h in sent_hits if h > 0)

    # Expected fields
    support_rate = supported / max(1, len(sents))
    coverage_rate = len(covered_indices) / max(1, len(items))

    return {
        "support_rate": support_rate,
        "coverage_rate": coverage_rate,
        "sentence_support": sent_hits,
        # helpful extras
        "covered_mask": covered_mask,
        "uncovered_examples": uncovered_examples[:5],
    }




def _align_support_from_list(
    prose: str,
    items: List[str],
    threshold: float = 0.38,
    max_hits_per_sentence: int = 2,
) -> Dict[str, Any]:
    sents = _split_sents(prose)
    items = [x.strip() for x in (items or []) if x and x.strip()]
    if not items or not sents:
        return {
            "support_rate": 0.0,
            "coverage_rate": 0.0,
            "sentence_support": [],
        }
    sent_hits = []
    covered = set()
    supported = 0
    for s in sents:
        scored = sorted(
            ((i, _sim(s, it)) for i, it in enumerate(items)),
            key=lambda t: t[1],
            reverse=True,
        )
        picks = [(i, sc) for i, sc in scored[:max_hits_per_sentence] if sc >= threshold]
        if picks:
            supported += 1
            for i, _ in picks:
                covered.add(i)
        sent_hits.append({"sentence": s, "hits": picks})
    return {
        "support_rate": supported / max(1, len(sents)),
        "coverage_rate": len(covered) / max(1, len(items)),
        "sentence_support": sent_hits,
    }

def evaluate_rows(rows, persona: Optional[str]):
    recs = []
    for r in rows:
        text = r.get("text", "") or ""
        refs = r.get("references_by_cq", {}) or {}
        run = r.get("run") or r.get("variant") or "RUN"

        words = tokenize_words(text)
        sents = split_sentences(text)  # your existing splitter is fine
        fk = flesch_kincaid(text)

        # Inline citation path (legacy)
        tokens = extract_citation_tokens(text)
        uniq = set(tokens)
        ids = valid_id_set(refs)
        valid = sum(1 for t in tokens if t in ids or t.isdigit())
        citation_rate = (sum(1 for s in sents if BRACKET.search(s)) / len(sents)) if sents else 0.0

        ev_words, web_refs, kg_refs = evidence_word_counts(refs)

        # --- NEW: fallback when no inline tags present ---
        has_inline = bool(tokens) or any(BRACKET.search(s) for s in sents)

        support_rate = None
        coverage = None
        evidence_mode = "inline"
        context_size = None

        if has_inline:
            # Keep your existing coverage logic (based on cited IDs / footer)
            _picked, _used_capped, coverage = factlet_coverage(r)
            # For support, use % sentences with a bracket as a proxy (keeps old behavior)
            support_rate = citation_rate
            evidence_mode = "inline"
        else:
            # Build alignment source:
            # Prefer explicit context saved at generation; else plan factlets; else refs flattened
            context_items = r.get("context_items") or r.get("plan_facts") or r.get("context_lines")
            print(f"context_items : {context_items}")
            if not context_items:
                # As a last resort, fall back to references text
                context_items = _refs_to_items(refs)
            context_size = len(context_items or [])

            if r.get("context_lines"):
                aligned = context_coverage(text, context_items, jaccard_coverage_threshold=0.5, ngram_n=5)
            else:
                aligned = _align_support_from_list(text, context_items, threshold=0.38, max_hits_per_sentence=2)
            support_rate = aligned["support_rate"]
            coverage = aligned["coverage_rate"]
            evidence_mode = "plan" if r.get("pattern") in ("KG", "Hybrid") else "context"

        # fact density (unique citation tokens per sentence) – keep for continuity, even if zero
        fact_density = (len(uniq) / len(sents)) if sents else 0.0

        # ungrounded numeric/entity sentences (no inline cite) – keep metric
        ungrounded = 0
        for s in sents:
            has_num = bool(NUMERIC_RE.search(s))
            has_entity = len(entity_tokens(s)) >= 1
            has_cite = bool(BRACKET.search(s))
            if (has_num or has_entity) and not has_cite and has_inline:
                # Only penalize when inline mode is expected
                ungrounded += 1

        continuity = entity_continuity(sents)

        # persona readability gates
        target_grade = None
        if persona:
            pl = persona.lower()
            target_grade = 10.0 if pl.startswith("emma") else (12.0 if pl.startswith("luca") else None)
        grade_diff = abs(fk["fk_grade"] - target_grade) if target_grade is not None else 0.0

        # style violations
        roleplay = bool(ROLEPLAY_RE.search(text))
        meta = bool(META_RE.search(text))

        recs.append({
            "run": run,
            "persona": r.get("persona"),
            "pattern": r.get("pattern"),
            "beat_index": r.get("beat_index"),
            "beat_title": r.get("beat_title"),
            "words": len(words),
            "sentences": len(sents),
            "avg_words_per_sentence": round((len(words) / len(sents)), 2) if sents else 0.0,
            "fk_grade": fk["fk_grade"],
            "fk_reading_ease": fk["fk_reading_ease"],
            "citation_rate": round(citation_rate, 3),
            "citations_total": len(tokens),
            "citations_unique": len(uniq),
            "citations_valid": valid,
            "evidence_words": ev_words,
            "web_refs": web_refs,
            "kg_refs": kg_refs,
            "support_rate": round(support_rate or 0.0, 3),
            "coverage": round(coverage or 0.0, 3),
            "fact_density": round(fact_density, 3),
            "entity_continuity": round(continuity, 3),
            "ungrounded_claim_sents": ungrounded,
            "roleplay_violation": roleplay,
            "meta_violation": meta,
            "evidence_mode": evidence_mode,   # NEW
            "context_size": context_size,     # NEW (None in inline mode)
            "facts_used": r.get("facts_used", 0) or 0,
        })

    df = pd.DataFrame(recs)
    return df


    # Gates & Auto Score
    def gate(row):
        ok_cite = row["citation_rate"] >= 0.8
        ok_cov  = row["coverage"] >= 0.6
        ok_num  = row["ungrounded_claim_sents"] == 0
        ok_style = not (row["roleplay_violation"] or row["meta_violation"])
        return int(all([ok_cite, ok_cov, ok_num, ok_style]))

    def auto(row):
        # 0-100 score from subscores (0..1)
        subs = {
            "Coverage": min(1.0, row["coverage"]),
            "Grounding": min(1.0, row["citation_rate"]) * (1.0 if row["ungrounded_claim_sents"]==0 else 0.6),
            "Fusion": max(0.0, min(1.0, 1.5 - abs(row["fact_density"]-1.0))),  # peak at ~1.0 facts/sent
            "Coherence": max(0.0, min(1.0, row["entity_continuity"]/0.6)),      # 0.6 continuity ≈ full marks
            "Style": 1.0 if not (row["roleplay_violation"] or row["meta_violation"]) else 0.3,
        }
        return round(100*(0.30*subs["Coverage"] + 0.30*subs["Grounding"] + 0.15*subs["Fusion"] + 0.15*subs["Coherence"] + 0.10*subs["Style"]), 1)

    df["gate_pass"] = df.apply(gate, axis=1)
    df["auto_score"] = df.apply(auto, axis=1)
    return df

def write_reports(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # per-beat CSV
    csv_path = outdir / "narrative_eval_per_beat.csv"
    df.sort_values(["run","beat_index"]).to_csv(csv_path, index=False)
    # summary CSV
    print(f"df : {df.columns}")
    summ = df.groupby("run").agg({
        "auto_score":"mean",
        "gate_pass":"mean",
        "coverage":"mean",
        "citation_rate":"mean",
        "citations_valid":"sum",
        "evidence_words":"mean",
        "words":"mean",
        "fk_grade":"mean",
        "entity_continuity":"mean"
    }).round(3)
    summ_path = outdir / "narrative_eval_summary.csv"
    summ.to_csv(summ_path)

    # simple HTML
    html_path = outdir / "narrative_eval_report.html"
    parts = [
        "<h2>Narrative Evaluation — Summary</h2>",
        summ.to_html(),
        "<h2>Per-Beat</h2>",
        df.sort_values(["run","beat_index"]).to_html(index=False)
    ]
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return csv_path, summ_path, html_path

# --- add these helpers near the top of eval_narrative.py ---
def picked_kg_factlets(references_by_cq: dict) -> int:
    """Count KG factlets picked for this beat."""
    cnt = 0
    for lst in (references_by_cq or {}).values():
        for it in (lst or []):
            if (it.get("type") or "").lower() == "kg":
                cnt += 1
    return cnt

def factlet_coverage(row_obj: dict) -> tuple[int, int, float]:
    """Return (picked, used, coverage) using facts_used and KG factlets picked."""
    picked = picked_kg_factlets(row_obj.get("references_by_cq", {}))
    used = int(row_obj.get("facts_used") or 0)
    row_obj["kg_picked"] = picked
    row_obj["facts_used_effective"] = min(used, picked)
    row_obj["kg_utilization_raw"] = round((used / picked), 3) if picked else 0.0
    row_obj["coverage"] = round(row_obj["facts_used_effective"] / (picked or 1), 3)

    used_capped = min(used, picked)  # never exceed 1.0 coverage
    cov = (used_capped / picked) if picked else 0.0
    return picked, used_capped, cov


from pathlib import Path
import json

def combine_jsonls(in_paths, out_path, append=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)  # <-- ensure dir exists

    mode = "a" if append and out_path.exists() else "w"  # "w" creates/truncates
    with out_path.open(mode, encoding="utf-8") as out:
        print(f"[combine] writing -> {out_path} ({mode})")

        for p in map(Path, in_paths):
            print(f"[combine] reading <- {p}")
            if not p.exists():
                print(f"[combine][WARN] missing {p}, skipping")
                continue

            run_name = p.stem.replace("answers_", "")
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[combine][WARN] bad JSON in {p}: {e}")
                        continue
                    obj.setdefault("run", run_name)
                    out.write(json.dumps(obj, ensure_ascii=False))
                    out.write("\n")

    # At this point, the file exists; if inputs were empty, it will be an empty file.


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    eval = Path("runs/Emma-Medium-20250820-232808/KG/run-01/generator")



    combined = eval / "answers_combined.jsonl"
    combine_jsonls([eval / "answers_KG.jsonl"], combined)

    rows = read_jsonl(combined)

    df = evaluate_rows(rows,"Emma")
    csv_path, summ_path, html_path = write_reports(df, eval)
    print(f"Wrote:\n - {csv_path}\n - {summ_path}\n - {html_path}")

if __name__ == "__main__":
    main()
