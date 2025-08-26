#!/usr/bin/env python3
# Deterministic, model-free support checker for answers_*.jsonl

from __future__ import annotations
import argparse, json, re, os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from datetime import datetime

SENT_SPLIT   = re.compile(r'(?<=[.!?])\s+')
TOKEN_RX     = re.compile(r"[A-Za-z0-9']+")
PROPER_RX    = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
YEAR_RX      = re.compile(r"\b(19|20)\d{2}\b")
NUM_RX       = re.compile(r"\b\d[\d,\.]*\b")
URL_RX       = re.compile(r'https?://\S+|<[^>]+>')
TYPED_LIT_RX = re.compile(r'"\s*([^"]*?)\s*\^\^.*')
_PREFIX_RX   = re.compile(r'^\s*:?\s*(KG|WEB)\s*:\s*', re.I)
_KEYVAL_RX   = re.compile(r'\b[A-Za-z][A-Za-z0-9_ ]{1,32}:\s*"([^"]+)"')

def strip_noise(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("“", '"').replace("”", '"').replace("’", "'")
    m = TYPED_LIT_RX.fullmatch(s.strip())
    if m: s = m.group(1)
    s = URL_RX.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_sents(text: str) -> List[str]:
    if not text: return []
    return [p.strip() for p in SENT_SPLIT.split(text.strip()) if p.strip()]

def tokset(x: str) -> set:
    return set(TOKEN_RX.findall((x or "").lower()))

def extract_proper(s: str) -> List[str]:
    return [m.group(0) for m in PROPER_RX.finditer(s or "")]

def extract_years(s: str) -> List[str]:
    return YEAR_RX.findall(s or "")

def extract_numbers(s: str) -> List[str]:
    return NUM_RX.findall(s or "")

def _clean_context_line(s: str, max_len: int = 180) -> str:
    s1 = strip_noise(s)
    s1 = _PREFIX_RX.sub("", s1)
    qs = re.findall(r'"([^"]{3,240})"', s1)
    core = max(qs, key=len) if qs else s1
    vals = _KEYVAL_RX.findall(core)
    if vals: core = " — ".join(vals)
    core = re.sub(r"\s+", " ", core).strip()
    if len(core) > max_len:
        core = core[:max_len].rsplit(" ", 1)[0] + "…"
    return core

def _jaccard_tokens(a: str, b: str) -> float:
    A, B = tokset(a), tokset(b)
    return len(A & B) / max(1, len(A | B))

def normalize_context(ctxs: List[str], max_len: int = 180, near_dup: float = 0.85) -> List[str]:
    out: List[str] = []
    for c in ctxs:
        cc = _clean_context_line(c, max_len=max_len)
        if not cc: continue
        if any(_jaccard_tokens(cc, x) >= near_dup for x in out):
            continue
        out.append(cc)
    return out

def tfidf_cosine(a: str, b: str) -> float:
    a, b = strip_noise(a), strip_noise(b)
    docs = [a, b]; vocab: Dict[str, int] = {}
    for d in docs:
        for tok in TOKEN_RX.findall(d.lower()):
            if tok not in vocab: vocab[tok] = len(vocab)
    V = len(vocab)
    if V == 0: return 0.0
    tf = np.zeros((2, V), dtype=float)
    df = np.zeros(V, dtype=float)
    for i, d in enumerate(docs):
        counts: Dict[int, int] = {}
        for t in TOKEN_RX.findall(d.lower()):
            idx = vocab[t]; counts[idx] = counts.get(idx, 0) + 1
        for idx, c in counts.items():
            tf[i, idx] = c; df[idx] += 1
    idf = np.log((2 + 1) / (df + 1)) + 1.0
    X = tf * idf
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return float(X[0] @ X[1].T)

def char3_jaccard(a: str, b: str) -> float:
    def grams(x: str) -> set:
        t = " " + re.sub(r"\s+", " ", x.lower()) + " "
        return {t[i:i+3] for i in range(max(0, len(t) - 2))}
    A, B = grams(a), grams(b)
    if not A or not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def det_supported(sentence: str, contexts: List[str],
                  tf_th: float, cj_th: float) -> Tuple[bool, str, Dict[str,float]]:
    """Return (supported?, best_ctx, features) deterministically."""
    s = sentence
    best_ctx = ""
    best_sig = -1.0
    best_feats = {"tfidf": 0.0, "char3": 0.0, "names": 0.0, "nums": 0.0, "years": 0.0}

    s_names = set(extract_proper(s))
    s_nums  = set(extract_numbers(s))
    s_years = set(extract_years(s))

    for c in contexts:
        tf = tfidf_cosine(s, c)
        cj = char3_jaccard(s, c)
        c_names = set(extract_proper(c))
        c_nums  = set(extract_numbers(c))
        c_years = set(extract_years(c))

        names_ok = float(len(s_names & c_names) > 0)
        nums_ok  = float(len(s_nums  & c_nums ) > 0)
        years_ok = float(len(s_years & c_years) > 0)

        # signal combines lexical+anchors so short KG lines can still win
        sig = max(tf, cj) + 0.06*names_ok + 0.04*max(nums_ok, years_ok)

        if sig > best_sig:
            best_sig = sig
            best_ctx = c
            best_feats = {"tfidf": tf, "char3": cj, "names": names_ok, "nums": nums_ok, "years": years_ok}

    # Rule A: entity+number/year overlap with any context
    anchor_ok = (best_feats["names"] >= 1.0) and (best_feats["nums"] >= 1.0 or best_feats["years"] >= 1.0)
    # Rule B: lexical overlap strong enough
    lexical_ok = (best_feats["tfidf"] >= tf_th) or (best_feats["char3"] >= cj_th)

    return bool(anchor_ok or lexical_ok), best_ctx, best_feats

def main():
    ap = argparse.ArgumentParser(description="Deterministic support checker (no transformers).")
    ap.add_argument("--answers", required=True, help="answers_*.jsonl")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--clean-context", action="store_true", help="clean/dedupe context lines first")
    ap.add_argument("--tf-th", type=float, default=0.35, help="TF-IDF cosine threshold")
    ap.add_argument("--cj-th", type=float, default=0.28, help="char-3 Jaccard threshold")
    args = ap.parse_args()

    answers_path = Path(args.answers)
    recs: List[Dict[str, Any]] = []
    with answers_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                recs.append(json.loads(ln))
            except Exception:
                pass

    rows: List[Dict[str, Any]] = []
    for rec in recs:
        bidx = rec.get("beat_index")
        bttl = rec.get("beat_title", f"Beat {bidx}")
        text = rec.get("text") or ""
        ctx_raw = [strip_noise(x) for x in (rec.get("context_lines") or []) if strip_noise(x)]
        ctx = normalize_context(ctx_raw) if args.clean_context else ctx_raw
        for si, s in enumerate(split_sents(text)):
            supported, best_ctx, feats = det_supported(s, ctx, tf_th=args.tf_th, cj_th=args.cj_th)
            rows.append({
                "beat_index": bidx, "beat_title": bttl, "sentence_idx": si,
                "sentence": s, "best_evidence": best_ctx,
                "tfidf": feats["tfidf"], "char3": feats["char3"],
                "name_overlap": feats["names"], "num_overlap": feats["nums"], "year_overlap": feats["years"],
                "supported": bool(supported),
            })

    df = pd.DataFrame(rows).sort_values(["beat_index","sentence_idx"]).reset_index(drop=True)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    if df.empty:
        summ = pd.DataFrame(columns=["beat_index","beat_title","sentences","supported_sentences","support_pct"])
    else:
        summ = (
            df.groupby(["beat_index","beat_title"])
              .agg(sentences=("sentence_idx","nunique"),
                   supported_sentences=("supported","sum"),
                   support_pct=("supported", lambda x: 100.0 * x.mean()))
              .reset_index()
              .sort_values("beat_index")
        )
    out_summary = Path(args.out_summary)
    summ.to_csv(out_summary, index=False, encoding="utf-8")

    # provenance meta sidecar
    meta = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "answers_path": str(answers_path),
        "clean_context": bool(args.clean_context),
        "tf_th": args.tf_th,
        "cj_th": args.cj_th,
        "sentences_total": int(df.shape[0]),
        "sentences_supported": int(df["supported"].sum()) if not df.empty else 0,
    }
    out_summary.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DET] wrote {out_csv}")
    print(f"[DET] wrote {out_summary}")
    print(f"[DET] support: {meta['sentences_supported']}/{meta['sentences_total']} ({(100.0*meta['sentences_supported']/max(1,meta['sentences_total'])):.1f}%)")

if __name__ == "__main__":
    main()
