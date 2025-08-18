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

def evaluate_rows(rows, persona: Optional[str]):
    recs = []
    for r in rows:
        text = r.get("text","") or ""
        refs = r.get("references_by_cq", {}) or {}
        run = r.get("run") or r.get("variant") or "RUN"
        words = tokenize_words(text); sents = split_sentences(text)
        fk = flesch_kincaid(text)
        tokens = extract_citation_tokens(text)
        uniq = set(tokens)
        ids = valid_id_set(refs)
        valid = sum(1 for t in tokens if t in ids or t.isdigit())  # allow numeric style
        citation_rate = (sum(1 for s in sents if BRACKET.search(s)) / len(sents)) if sents else 0.0
        ev_words, web_refs, kg_refs = evidence_word_counts(refs)
        # coverage from footer if present; else from citations ∩ ids
        # used_ids, unused_ids = parse_used_footer(text)
        # total_ids = len(used_ids | unused_ids) or (len(ids) or 1)
        # coverage = (len(used_ids) / total_ids) if (used_ids or unused_ids) else ((len(uniq & ids) / (len(ids) or 1)))
        picked, used_capped, coverage = factlet_coverage(r)
        # fact density
        fact_density = (len(uniq) / len(sents)) if sents else 0.0
        # ungrounded numeric/entity sentences
        ungrounded = 0
        for s in sents:
            has_num = bool(NUMERIC_RE.search(s))
            has_entity = len(entity_tokens(s)) >= 1
            has_cite = bool(BRACKET.search(s))
            if (has_num or has_entity) and not has_cite:
                ungrounded += 1
        continuity = entity_continuity(sents)
        # persona readability gates
        target_grade = None
        if persona:
            target_grade = 10.0 if persona.lower().startswith("emma") else (12.0 if persona.lower().startswith("luca") else None)
        grade_diff = abs(fk["fk_grade"] - target_grade) if target_grade is not None else 0.0
        # style violations
        roleplay = bool(ROLEPLAY_RE.search(text))
        meta = bool(META_RE.search(text))

        recs.append({
            "run": run,
            "persona": r.get("persona"),
            "beat_index": r.get("beat_index"),
            "beat_title": r.get("beat_title"),
            "words": len(words),
            "sentences": len(sents),
            "avg_words_per_sentence": round((len(words)/len(sents)),2) if sents else 0.0,
            "fk_grade": fk["fk_grade"],
            "fk_reading_ease": fk["fk_reading_ease"],
            "citation_rate": round(citation_rate,3),
            "citations_total": len(tokens),
            "citations_unique": len(uniq),
            "citations_valid": valid,
            "evidence_words": ev_words,
            "web_refs": web_refs,
            "kg_refs": kg_refs,
            "coverage": round(coverage,3),
            "fact_density": round(fact_density,3),
            "entity_continuity": round(continuity,3),
            "ungrounded_claim_sents": ungrounded,
            "roleplay_violation": roleplay,
            "meta_violation": meta,
            "facts_used": r.get("facts_used", 0) or 0,
        })
    df = pd.DataFrame(recs)

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

    eval = Path("runs/20250818_213845__emma-medium__seed42/generator")



    combined = eval / "answers_combined.jsonl"
    combine_jsonls([eval / "answers_KG.jsonl",
                    eval/ "answers_Hybrid.jsonl"], combined)

    rows = read_jsonl(combined)

    df = evaluate_rows(rows,"Emma")
    csv_path, summ_path, html_path = write_reports(df, eval)
    print(f"Wrote:\n - {csv_path}\n - {summ_path}\n - {html_path}")

if __name__ == "__main__":
    main()
