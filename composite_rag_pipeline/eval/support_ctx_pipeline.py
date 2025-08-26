#!/usr/bin/env python3
# support_ctx_pipeline.py
"""
Evaluate whether generated sentences are supported by the exact `context_lines` the model saw,
and optionally apply minimal, reversible fixes that inject tiny parenthetical tokens from the
best-matching context line. Works for both KG and Hybrid because both answers_*.jsonl include
`context_lines`.

USAGE EXAMPLES
==============
# 1) Evaluate (lexical-only; fast, no extra deps)
python3 composite_rag_pipeline/eval/support_ctx_pipeline.py eval \
  --answers outputs/answers_KG.jsonl \
  --out-csv outputs/eval/support_from_ctx_KG_sentences.csv \
  --out-summary outputs/eval/support_from_ctx_KG_summary.csv \
  --mode lexical

# 2) Evaluate (NLI; stronger)
pip install 'transformers>=4.40' torch            # once
python3 composite_rag_pipeline/eval/support_ctx_pipeline.py eval \
  --answers outputs/answers_Hybrid.jsonl \
  --out-csv outputs/eval/support_from_ctx_Hybrid_sentences.csv \
  --out-summary outputs/eval/support_from_ctx_Hybrid_summary.csv \
  --mode nli --nli-model roberta-large-mnli

# 3) Auto-fix unsupported sentences (reversible)
python3 composite_rag_pipeline/eval/support_ctx_pipeline.py fix \
  --answers outputs/answers_KG.jsonl \
  --out-fixed-answers outputs/answers_KG_fixed.jsonl \
  --out-fixed-story outputs/story_KG_fixed.md \
  --out-patches-csv outputs/eval/patches_KG.csv \
  --max-repairs-per-beat 2

# 4) One-shot pipeline: eval → fix → re-eval (lexical)
python3 composite_rag_pipeline/eval/support_ctx_pipeline.py pipeline \
  --answers outputs/answers_KG.jsonl \
  --workdir outputs/eval/pipeline_KG \
  --mode lexical

# 5) One-shot pipeline with NLI
python3 composite_rag_pipeline/eval/support_ctx_pipeline.py pipeline \
  --answers outputs/answers_Hybrid.jsonl \
  --workdir outputs/eval/pipeline_Hybrid \
  --mode nli --nli-model roberta-large-mnli
"""

from __future__ import annotations
import argparse, json, re, os, sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ----------------------- Light text utilities -----------------------

SENT_SPLIT   = re.compile(r'(?<=[.!?])\s+')
TOKEN_RX     = re.compile(r"[A-Za-z0-9']+")
PROPER_RX    = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b")
NUM_RX       = re.compile(r"\b\d[\d,\.]*\b")
URL_RX       = re.compile(r'https?://\S+|<[^>]+>')
TYPED_LIT_RX = re.compile(r'"\s*([^"]*?)\s*\^\^.*')

def strip_noise(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    m = TYPED_LIT_RX.fullmatch(s.strip())
    if m:
        s = m.group(1)
    s = URL_RX.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_sents(text: str) -> List[str]:
    if not text:
        return []
    return [p.strip() for p in SENT_SPLIT.split(text.strip()) if p.strip()]

def tokset(x: str) -> set:
    return set(TOKEN_RX.findall((x or "").lower()))

def extract_proper(s: str) -> List[str]:
    return [m.group(0) for m in PROPER_RX.finditer(s or "")]

def extract_years(s: str) -> List[str]:
    return YEAR_RX.findall(s or "")

def extract_numbers(s: str) -> List[str]:
    return NUM_RX.findall(s or "")

# ----------------------- Similarity features -----------------------

def tfidf_cosine(a: str, b: str) -> float:
    import numpy as np
    a = strip_noise(a); b = strip_noise(b)
    docs = [a, b]; vocab: Dict[str, int] = {}
    for d in docs:
        for tok in TOKEN_RX.findall(d.lower()):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    V = len(vocab)
    if V == 0:
        return 0.0
    tf = np.zeros((2, V), dtype=float)
    df = np.zeros(V, dtype=float)
    for i, d in enumerate(docs):
        counts: Dict[int, int] = {}
        for t in TOKEN_RX.findall(d.lower()):
            idx = vocab[t]
            counts[idx] = counts.get(idx, 0) + 1
        for idx, c in counts.items():
            tf[i, idx] = c
            df[idx] += 1
    idf = np.log((2 + 1) / (df + 1)) + 1.0  # smoothed
    X = tf * idf
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return float(X[0] @ X[1].T)

def char3_jaccard(a: str, b: str) -> float:
    def grams(x: str) -> set:
        t = " " + re.sub(r"\s+", " ", x.lower()) + " "
        return {t[i:i+3] for i in range(max(0, len(t) - 2))}
    A, B = grams(a), grams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def candidate_gate(sent: str, ctx: str) -> bool:
    """
    Small precision gate to drop obviously-bad context candidates.
    Tunable via env vars:
      NLI_GATE_OFF=1             -> disable gate
      NLI_GATE_TFIDF=0.30        -> tf-idf threshold
      NLI_GATE_CHAR3=0.25        -> char-3 threshold
    Also passes if a name/number/year overlaps.
    """
    import os
    if os.getenv("NLI_GATE_OFF", "0") == "1":
        return True

    tf_th = float(os.getenv("NLI_GATE_TFIDF", "0.30"))
    cj_th = float(os.getenv("NLI_GATE_CHAR3", "0.25"))

    s, c = strip_noise(sent), strip_noise(ctx)
    if tfidf_cosine(s, c) >= tf_th or char3_jaccard(s, c) >= cj_th:
        return True

    if set(extract_proper(s)) & set(extract_proper(c)): return True
    if set(extract_numbers(s)) & set(extract_numbers(c)): return True
    if set(extract_years(s))   & set(extract_years(c)):   return True
    return False



# def candidate_gate(sent: str, ctx: str) -> bool:
#     return True



# ---------- Context cleaner ----------
_KEYVAL_RX = re.compile(r'\b[A-Za-z][A-Za-z0-9_ ]{1,32}:\s*"([^"]+)"')
_PREFIX_RX = re.compile(r'^\s*:?\s*(KG|WEB)\s*:\s*', re.I)

def _clean_context_line(s: str, max_len: int = 180) -> str:
    """Make a noisy context line 'atomic': drop prefixes, pull values, prefer quoted spans, trim."""
    s1 = strip_noise(s)
    s1 = _PREFIX_RX.sub("", s1)                      # drop leading "KG:" / "WEB:"
    # prefer the longest quoted span if any
    qs = re.findall(r'"([^"]{3,240})"', s1)
    core = max(qs, key=len) if qs else s1
    # pull key:"value" pairs into a compact clause
    vals = _KEYVAL_RX.findall(core)
    if vals:
        core = " — ".join(vals)
    # final tidy
    core = re.sub(r"\s+", " ", core).strip()
    if len(core) > max_len:
        core = core[:max_len].rsplit(" ", 1)[0] + "…"
    return core

def _jaccard_tokens(a: str, b: str) -> float:
    A, B = tokset(a), tokset(b)
    return len(A & B) / max(1, len(A | B))

def normalize_context(ctxs: List[str], max_len: int = 180, near_dup: float = 0.85) -> List[str]:
    """Clean + near-dup dedupe."""
    out: List[str] = []
    for c in ctxs:
        cc = _clean_context_line(c, max_len=max_len)
        if not cc:
            continue
        if any(_jaccard_tokens(cc, x) >= near_dup for x in out):
            continue
        out.append(cc)
    return out

# ----------------------- NLI (optional) -----------------------

# ---- DROP-IN REPLACEMENT: no pipelines, direct MNLI scorer ----
def load_nli(model_name: str):
    """
    Returns a callable nli(premise, hypothesis) -> {'ent':..., 'con':..., 'neu':...}
    Uses AutoModelForSequenceClassification directly to avoid transformers.pipelines.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()

    # roberta-large-mnli label order: [contradiction, neutral, entailment]
    def nli(premise: str, hypothesis: str):
        inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = mdl(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()
        return {"con": float(probs[0]), "neu": float(probs[1]), "ent": float(probs[2])}

    return nli

def nli_entailment(nli, premise: str, hypothesis: str) -> dict:
    # nli is the callable returned by load_nli()
    return nli(premise, hypothesis)


# ----------------------- Evaluation core -----------------------

def evaluate_from_answers(
    answers_path: Path,
    out_csv: Path,
    out_summary: Path,
    mode: str = "lexical",                # "lexical" or "nli"
    nli_model: str = "roberta-large-mnli",
    topk: int = 5,
    ent_th: float = 0.62,
    con_th: float = 0.35,
    clean_context: bool = False,
) -> Tuple[int, int]:
    import pandas as pd
    recs: List[Dict[str, Any]] = []
    with answers_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass

    nli = load_nli(nli_model) if mode == "nli" else None

    rows: List[Dict[str, Any]] = []
    for rec in recs:
        bidx = rec.get("beat_index")
        bttl = rec.get("beat_title", f"Beat {bidx}")
        sents = split_sents(rec.get("text") or "")
        raw_ctxs = [strip_noise(x) for x in (rec.get("context_lines") or []) if strip_noise(x)]
        want_clean = clean_context or (os.getenv("CLEAN_CONTEXT", "0") == "1")

        if want_clean:
            clean_ctxs = normalize_context(raw_ctxs)  # atomic + dedup
            # Merge RAW (original phrasing) + CLEAN (atomic), keep order, dedupe by lowercase
            ctxs, seen = [], set()
            for c in (raw_ctxs + clean_ctxs):
                k = c.lower().strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                ctxs.append(c)
        else:
            ctxs = raw_ctxs

        for si, s in enumerate(sents):
            # Gate + top-k by tfidf
            cand = [(c, tfidf_cosine(s, c)) for c in ctxs if candidate_gate(s, c)]
            cand.sort(key=lambda x: x[1], reverse=True)
            cand = cand[:max(1, min(topk, len(cand)))]
            best_ctx = ""
            ent = con = tf = cj = float("nan")
            supported = False
            if mode == "lexical":
                best_tf = best_cj = -1.0
                for c, _ in cand or []:
                    tf2 = tfidf_cosine(s, c)
                    cj2 = char3_jaccard(s, c)
                    if max(tf2, cj2) > max(best_tf, best_cj):
                        best_ctx, best_tf, best_cj = c, tf2, cj2
                names_ok = bool(set(extract_proper(s)) & set(extract_proper(best_ctx or "")))
                nums_ok  = bool(set(extract_numbers(s)) & set(extract_numbers(best_ctx or "")))
                years_ok = bool(set(extract_years(s)) & set(extract_years(best_ctx or "")))
                tf, cj = best_tf, best_cj
                supported = (max(tf, cj) >= 0.62) or ((names_ok or nums_ok or years_ok) and max(tf, cj) >= 0.45)
            else:
                best_ent = best_con = -1.0
                best_ctx = ""
                best_score = -1.0
                for c, _ in cand or []:
                    r = nli_entailment(nli, premise=c, hypothesis=s)
                    ent, con = r["ent"], r["con"]
                    # add a small lexical prior
                    sim = max(tfidf_cosine(s, c), char3_jaccard(s, c))
                    names_ok = bool(set(extract_proper(s)) & set(extract_proper(c)))
                    nums_ok = bool(set(extract_numbers(s)) & set(extract_numbers(c)))
                    years_ok = bool(set(extract_years(s)) & set(extract_years(c)))
                    score = ent + 0.07 * sim + 0.03 * (names_ok + nums_ok + years_ok)

                    if score > best_score:
                        best_score = score
                        best_ctx = c
                        best_ent, best_con = ent, con

                ent, con = best_ent, best_con
                supported = (ent >= ent_th) and not (con >= con_th)

        rows.append({
                "beat_index": bidx, "beat_title": bttl, "sentence_idx": si,
                "sentence": s, "best_evidence": best_ctx,
                "entailment": ent, "contradiction": con,
                "tfidf": tf, "char3": cj,
                "supported": bool(supported),
            })

    df = pd.DataFrame(rows).sort_values(["beat_index","sentence_idx"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    if df.empty:
        summ = pd.DataFrame(columns=["beat_index","beat_title","sentences","supported_sentences","support_pct"])
        supported_count = 0
        total_sents = 0
    else:
        summ = (
            df.groupby(["beat_index", "beat_title"])
              .agg(sentences=("sentence_idx", "nunique"),
                   supported_sentences=("supported", "sum"),
                   support_pct=("supported", lambda x: 100.0 * x.mean()))
              .reset_index().sort_values("beat_index")
        )
        supported_count = int(df["supported"].sum())
        total_sents = int(df.shape[0])

    out_summary.write_text(summ.to_csv(index=False), encoding="utf-8")
    print(f"[EVAL] wrote {out_csv}")
    print(f"[EVAL] wrote {out_summary}")
    print(f"[EVAL] supported {supported_count}/{total_sents} ({(100.0*supported_count/max(1,total_sents)):.1f}%)")

    return supported_count, total_sents

# ----------------------- Fix (minimal, reversible) -----------------------

# --- Evidence map loader: (beat_index, sentence_idx) -> best_evidence ---
from typing import Optional

def read_evidence_map(sent_csv: Path) -> Dict[Tuple[int, int], str]:
    import pandas as pd
    m: Dict[Tuple[int, int], str] = {}
    df = pd.read_csv(sent_csv)
    need = {"beat_index", "sentence_idx", "best_evidence"}
    if not need.issubset(df.columns):
        return m
    for _, r in df.iterrows():
        try:
            b = int(r["beat_index"])
            si = int(r["sentence_idx"])
            ev = str(r.get("best_evidence") or "").strip()
            if ev:
                m[(b, si)] = ev
        except Exception:
            continue
    return m


def pick_best_context(sent: str, ctxs: List[str]) -> Tuple[str, float]:
    best, best_sim = "", -1.0
    for c in ctxs:
        sim = max(tfidf_cosine(sent, c), 0.0)
        if sim > best_sim:
            best, best_sim = c, sim
    return best, best_sim

def build_parenthetical(from_ctx: str, avoid_in_sentence: str, max_items: int = 2) -> str:
    """
    Build a compact parenthetical that boosts NLI by injecting concrete anchors.
    Heuristic priority: NAME + (YEAR or NUMBER). Dedup tokens already present.
    """
    used = tokset(avoid_in_sentence)

    # Collect candidates
    names  = extract_proper(from_ctx)
    years  = extract_years(from_ctx)
    nums   = [n for n in extract_numbers(from_ctx) if n not in years]  # don't double-count years as numbers

    # Light stoplist for very generic "names"
    stop_names = {"United", "Kingdom", "United Kingdom", "United States", "Global", "Broadcast", "Event",
                  "City", "Stadium", "Arena", "Festival"}
    def name_score(nm: str) -> tuple[int,int]:
        # prefer longer names; simple tie-break: token count, then length
        return (len(nm.split()), len(nm))

    # Filter out names already in sentence or too generic
    cand_names = [nm for nm in names if nm and nm.lower() not in used and nm not in stop_names]
    cand_names.sort(key=name_score, reverse=True)

    # Filter out years / nums already present
    cand_years = [y for y in years if y.lower() not in used]
    cand_nums  = [n for n in nums  if n.lower() not in used]

    parts: List[str] = []
    # 1) pick BEST name if any
    if cand_names:
        parts.append(cand_names[0])
    # 2) pick YEAR if any, else NUMBER
    if cand_years:
        parts.append(cand_years[0])
    elif cand_nums:
        parts.append(cand_nums[0])

    # If still empty, fall back to any two distinct tokens from ctx not in sentence
    if not parts:
        ctx_tokens = [t for t in tokset(from_ctx) if t not in used and len(t) > 2]
        parts = list(ctx_tokens)[:max_items]

    # Trim to max_items and return
    parts = [p for p in parts if p][:max_items]
    return ", ".join(parts)


def patch_text_with_context_tokens(
    text: str,
    ctxs: List[str],
    max_repairs: int = 2,
    beat_index: Optional[int] = None,
    evidence_map: Optional[Dict[Tuple[int, int], str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    sents = split_sents(text)
    fixed = sents[:]
    patches: List[Dict[str, Any]] = []
    repairs = 0
    for i, s in enumerate(sents):
        if repairs >= max_repairs:
            break

        # Prefer the exact best_evidence used by eval
        best_ctx = ""
        if evidence_map is not None and beat_index is not None:
            best_ctx = evidence_map.get((beat_index, i), "") or ""

        # Fallback to similarity if no evidence recorded for this sentence
        if not best_ctx:
            best_ctx, _ = pick_best_context(s, ctxs)
        if not best_ctx:
            continue

        st, ct = tokset(s), tokset(best_ctx)
        if len(st & ct) >= 2:  # already overlaps enough
            continue

        parenth = build_parenthetical(best_ctx, s, max_items=2)  # stronger builder already in your file
        if not parenth:
            continue

        if s.endswith((".", "!", "?")):
            new_s = s[:-1] + f" ({parenth})."
        else:
            new_s = s + f" ({parenth})."
        fixed[i] = new_s
        patches.append({"sentence_idx": i, "before": s, "after": new_s, "evidence": best_ctx})
        repairs += 1

    return " ".join(fixed), patches


def fix_answers(
    answers_path: Path,
    out_fixed_answers: Path,
    out_fixed_story: Path,
    out_patches_csv: Path,
    max_repairs_per_beat: int = 2,
    evidence_map: Optional[Dict[Tuple[int, int], str]] = None,   # <--- NEW
) -> int:
    import pandas as pd
    recs: List[Dict[str, Any]] = []
    with answers_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass

    out_recs: List[Dict[str, Any]] = []
    fixed_story_lines: List[str] = []
    patch_rows: List[Dict[str, Any]] = []
    total_fixes = 0

    for rec in recs:
        bidx = rec.get("beat_index")
        bttl = rec.get("beat_title", f"Beat {bidx}")

        # --- RAW + CLEAN merged context for fixing ---
        raw_ctxs = [c for c in (rec.get("context_lines") or []) if c and c.strip()]
        clean_ctxs = normalize_context(raw_ctxs)
        ctxs, seen = [], set()
        for c in (raw_ctxs + clean_ctxs):
            k = strip_noise(c).lower().strip()
            if not k or k in seen:
                continue
            seen.add(k)
            ctxs.append(strip_noise(c))
        # --------------------------------------------

        text = rec.get("text") or ""
        new_text, patches = patch_text_with_context_tokens(
            text, ctxs,
            max_repairs=max_repairs_per_beat,
            beat_index=bidx,
            evidence_map=evidence_map,
        )

        rec_out = dict(rec)
        rec_out["text_fixed"] = new_text
        rec_out["fixes_applied"] = len(patches)
        out_recs.append(rec_out)
        fixed_story_lines.append(f"## {bttl}\n\n{new_text}\n")
        for p in patches:
            patch_rows.append({"beat_index": bidx, "beat_title": bttl, **p})
        total_fixes += len(patches)

    out_fixed_answers.parent.mkdir(parents=True, exist_ok=True)
    out_fixed_answers.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_recs), encoding="utf-8")
    out_fixed_story.parent.mkdir(parents=True, exist_ok=True)
    out_fixed_story.write_text("\n".join(fixed_story_lines), encoding="utf-8")

    dfp = pd.DataFrame(patch_rows)
    out_patches_csv.parent.mkdir(parents=True, exist_ok=True)
    dfp.to_csv(out_patches_csv, index=False, encoding="utf-8")

    print(f"[FIX] wrote fixed answers → {out_fixed_answers}")
    print(f"[FIX] wrote fixed story   → {out_fixed_story}")
    print(f"[FIX] wrote patches       → {out_patches_csv}")
    print(f"[FIX] total patches applied: {total_fixes}")
    return total_fixes


# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate and optionally auto-fix support against context_lines.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # eval
    ev = sub.add_parser("eval", help="Evaluate support of sentences against context_lines.")
    ev.add_argument("--answers", required=True)
    ev.add_argument("--out-csv", required=True)
    ev.add_argument("--out-summary", required=True)
    ev.add_argument("--mode", choices=["lexical","nli"], default="lexical")
    ev.add_argument("--nli-model", default="roberta-large-mnli")
    ev.add_argument("--topk", type=int, default=5)
    ev.add_argument("--ent-th", type=float, default=0.62)
    ev.add_argument("--con-th", type=float, default=0.35)
    ev.add_argument("--clean-context", action="store_true")

    # fix
    fx = sub.add_parser("fix", help="Auto-fix unsupported sentences by injecting small parentheticals from context.")
    fx.add_argument("--answers", required=True)
    fx.add_argument("--out-fixed-answers", required=True)
    fx.add_argument("--out-fixed-story", required=True)
    fx.add_argument("--out-patches-csv", required=True)
    fx.add_argument("--max-repairs-per-beat", type=int, default=2)
    fx.add_argument("--evidence-from", default=None,
                    help="Path to a sentences CSV (from eval) to source best_evidence.")

    # pipeline
    pl = sub.add_parser("pipeline", help="One-shot: eval → fix → re-eval (writes into a workdir).")
    pl.add_argument("--answers", required=True)
    pl.add_argument("--workdir", required=True, help="Directory to write all artifacts into.")
    pl.add_argument("--mode", choices=["lexical","nli"], default="lexical")
    pl.add_argument("--nli-model", default="roberta-large-mnli")
    pl.add_argument("--topk", type=int, default=5)
    pl.add_argument("--ent-th", type=float, default=0.62)
    pl.add_argument("--con-th", type=float, default=0.35)
    pl.add_argument("--max-repairs-per-beat", type=int, default=2)
    pl.add_argument("--clean-context", action="store_true")

    args = ap.parse_args()

    if args.cmd == "eval":
        evaluate_from_answers(
            answers_path=Path(args.answers),
            out_csv=Path(args.out_csv),
            out_summary=Path(args.out_summary),
            mode=args.mode,
            nli_model=args.nli_model,
            topk=args.topk,
            ent_th=args.ent_th,
            con_th=args.con_th,
            clean_context=args.clean_context,
        )
        return

    if args.cmd == "fix":
        fix_answers(
            answers_path=Path(args.answers),
            out_fixed_answers=Path(args.out_fixed_answers),
            out_fixed_story=Path(args.out_fixed_story),
            out_patches_csv=Path(args.out_patches_csv),
            max_repairs_per_beat=args.max_repairs_per_beat,
            evidence_map=evidence_map,
        )
        return

    if args.cmd == "pipeline":
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        # 1) Eval original
        out_csv = workdir / "support_sentences.csv"
        out_sum = workdir / "support_summary.csv"
        sup_before, tot_before = evaluate_from_answers(
            answers_path=Path(args.answers),
            out_csv=out_csv,
            out_summary=out_sum,
            mode=args.mode,
            nli_model=args.nli_model,
            topk=args.topk,
            ent_th=args.ent_th,
            con_th=args.con_th,
            clean_context=args.clean_context,
        )

        evidence_map = read_evidence_map(out_csv)

        # 2) Fix
        fixed_answers = workdir / "answers_fixed.jsonl"
        fixed_story   = workdir / "story_fixed.md"
        patches_csv   = workdir / "patches.csv"
        fix_answers(
            answers_path=Path(args.answers),
            out_fixed_answers=fixed_answers,
            out_fixed_story=fixed_story,
            out_patches_csv=patches_csv,
            max_repairs_per_beat=args.max_repairs_per_beat,
        )

        # 3) Re-eval fixed
        out_csv2 = workdir / "support_sentences_fixed.csv"
        out_sum2 = workdir / "support_summary_fixed.csv"
        sup_after, tot_after = evaluate_from_answers(
            answers_path=fixed_answers,
            out_csv=out_csv2,
            out_summary=out_sum2,
            mode=args.mode,
            nli_model=args.nli_model,
            topk=args.topk,
            ent_th=args.ent_th,
            con_th=args.con_th,
            clean_context=args.clean_context,
        )

        print("\n--- PIPELINE SUMMARY ---")
        print(f"Before: {sup_before}/{tot_before} supported")
        print(f"After : {sup_after}/{tot_after} supported")
        return

if __name__ == "__main__":
    main()
