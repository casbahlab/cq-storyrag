#!/usr/bin/env python3
# support_ctx_reset.py — deterministic, model-free support checker
# Now ALSO computes "coverage" (evidence -> story) when --emit-coverage is passed.
# Adds: alias/date/number normalization, optional BM25 filtering, light/heavy clean

from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------- Regexes & basic utils -----------------------

SENT_SPLIT   = re.compile(r'(?<=[.!?])\s+')
TOKEN_RX     = re.compile(r"[A-Za-z0-9']+")
PROPER_RX    = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
YEAR_ONLY_RX = re.compile(r"\b(19|20)\d{2}\b")
NUM_RX       = re.compile(r"\b\d[\d,\.]*\b")
URL_RX       = re.compile(r'https?://\S+|<[^>]+>')
TYPED_LIT_RX = re.compile(r'"\s*([^"]*?)\s*\^\^.*')
_PREFIX_RX   = re.compile(r'^\s*:?\s*(KG|WEB)\s*:\s*', re.I)
_KEYVAL_RX   = re.compile(r'\b[A-Za-z][A-Za-z0-9_ ]{1,32}:\s*"([^"]+)"')

MONTHS = {
    "january": "01","february":"02","march":"03","april":"04","may":"05","june":"06",
    "july":"07","august":"08","september":"09","october":"10","november":"11","december":"12"
}

SPELLED_NUM = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90
}
SCALE_WORDS = {"thousand":1_000, "million":1_000_000, "billion":1_000_000_000,
               "k":1_000, "m":1_000_000, "bn":1_000_000_000, "b":1_000_000_000}

# Lightweight default aliases; extend with --alias-file
DEFAULT_ALIASES = {
    "john f. kennedy stadium": ["jfk stadium", "john kennedy stadium", "live aid stadium philadelphia", "jfk"],
    "british broadcasting corporation": ["bbc"],
    "mtv": ["music television"],
    "wembley stadium": ["wembley"],
    "philadelphia": ["philly"],
}

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
    return YEAR_ONLY_RX.findall(s or "")

def extract_numbers(s: str) -> List[str]:
    return NUM_RX.findall(s or "")

# ----------------------- Canonicalization -----------------------

def _normalize_dates(text: str) -> str:
    t = text

    pat1 = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b")   # 13 July 1985
    pat2 = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b")  # July 13, 1985

    def to_iso(d: int, mon: str, y: int) -> str:
        mm = MONTHS.get(mon.lower())
        if not mm: return ""
        return f"{y:04d}-{mm}-{int(d):02d}"

    out = t
    for m in pat1.finditer(t):
        d, mon, y = int(m.group(1)), m.group(2), int(m.group(3))
        iso = to_iso(d, mon, y)
        if iso: out = out.replace(m.group(0), f"{m.group(0)} {iso}")
    for m in pat2.finditer(t):
        mon, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        iso = to_iso(d, mon, y)
        if iso: out = out.replace(m.group(0), f"{m.group(0)} {iso}")
    return out

def _words_to_number_phrase(w: List[str]) -> int:
    if not w: return 0
    total = 0
    current = 0
    for token in w:
        tl = token.lower()
        if tl in SPELLED_NUM:
            val = SPELLED_NUM[tl]
            if val >= 20 and current > 0: current += val
            elif val >= 20: current = val
            else: current += val
        elif tl in SCALE_WORDS:
            scale = SCALE_WORDS[tl]
            if current == 0: current = 1
            total += current * scale
            current = 0
        else:
            return 0
    return total + current

def _normalize_spelled_numbers(text: str) -> str:
    toks = re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text)
    out = []
    i = 0
    while i < len(toks):
        best = None
        for L in range(2, 6):
            span = toks[i:i+L]
            if not span: break
            if not all(re.fullmatch(r"[A-Za-z]+", x or "") for x in span):
                continue
            val = _words_to_number_phrase(span)
            if val > 0:
                best = (L, val)
        if best:
            L, val = best
            out.extend(toks[i:i+L])
            out.append(str(val))
            i += L
        else:
            out.append(toks[i]); i += 1
    return re.sub(r"\s+", " ", " ".join(out)).strip()

def _apply_aliases(text: str, alias_map: Dict[str, List[str]]) -> str:
    t = text
    for canon, variants in alias_map.items():
        canon_l = canon.lower()
        for v in [canon] + list(variants):
            v_l = v.lower()
            if not v_l: continue
            pattern = re.compile(rf"\b{re.escape(v_l)}\b", re.I)
            t = pattern.sub(f"{canon_l} {v_l}", t)
    return t

def canonicalize_text(text: str, alias_map: Dict[str, List[str]]) -> str:
    t = strip_noise(text)
    t = _apply_aliases(t, alias_map)
    t = _normalize_dates(t)
    t = _normalize_spelled_numbers(t)
    return t

# ----------------------- Context cleaning -----------------------

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

# --- light clean helpers (recommended) ---
PREFIX_RX = re.compile(r'^\s*:?\s*(KG|WEB)\s*:\s*', re.I)
URL_OR_IRI_RX = re.compile(r'https?://\S+|<[^>]+>')

def light_clean_line(s: str) -> str:
    if not s: return ""
    s = str(s).replace("“", '"').replace("”", '"').replace("’", "'")
    s = URL_OR_IRI_RX.sub("", s)
    s = PREFIX_RX.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def light_clean_context(ctxs: List[str]) -> List[str]:
    out, seen = [], set()
    for c in ctxs or []:
        cc = light_clean_line(c)
        if not cc: continue
        key = cc.lower()
        if key in seen: continue
        seen.add(key); out.append(cc)
    return out

# ----------------------- Similarities (TF-IDF / char-3) -----------------------

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

# ----------------------- BM25 (optional candidate filter) -----------------------

def _bm25_idf(N: int, df: int) -> float:
    return max(0.0, np.log((N - df + 0.5) / (df + 0.5) + 1e-9))

def bm25_rank(query: str, docs: List[str], k1: float = 1.2, b: float = 0.50) -> List[Tuple[int, float]]:
    q_toks = [t.lower() for t in TOKEN_RX.findall(query)]
    if not docs or not q_toks: return []
    D_toks = [[t.lower() for t in TOKEN_RX.findall(d)] for d in docs]
    N = len(docs)
    lens = np.array([max(1, len(toks)) for toks in D_toks], dtype=float)
    avgL = float(np.mean(lens)) if len(lens) else 1.0
    from collections import Counter
    dfs: Dict[str,int] = Counter()
    for toks in D_toks:
        for t in set(toks):
            dfs[t] += 1
    scores = np.zeros(N, dtype=float)
    for term in set(q_toks):
        idf = _bm25_idf(N, dfs.get(term, 0))
        if idf <= 0: continue
        tfs = np.array([D_toks[i].count(term) for i in range(N)], dtype=float)
        denom = tfs + k1 * (1 - b + b * (lens / avgL))
        contrib = idf * (tfs * (k1 + 1)) / np.where(denom == 0, 1.0, denom)
        scores += contrib
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

# ----------------------- Matching primitives -----------------------

def compute_feats(a: str, b: str) -> Dict[str, float]:
    tf = tfidf_cosine(a, b)
    cj = char3_jaccard(a, b)
    a_names, b_names = set(extract_proper(a)), set(extract_proper(b))
    a_nums , b_nums  = set(extract_numbers(a)), set(extract_numbers(b))
    a_years, b_years = set(extract_years(a)), set(extract_years(b))
    return {
        "tfidf": tf,
        "char3": cj,
        "names": float(len(a_names & b_names) > 0),
        "nums":  float(len(a_nums  & b_nums ) > 0),
        "years": float(len(a_years & b_years) > 0),
    }

def best_match(query: str, candidates: List[str]) -> Tuple[int, str, Dict[str,float]]:
    best_i, best_c, best_sig, best_feats = -1, "", -1.0, {"tfidf":0,"char3":0,"names":0,"nums":0,"years":0}
    for i, c in enumerate(candidates):
        feats = compute_feats(query, c)
        sig = max(feats["tfidf"], feats["char3"]) + 0.06*feats["names"] + 0.04*max(feats["nums"], feats["years"])
        if sig > best_sig:
            best_sig = sig; best_i = i; best_c = c; best_feats = feats
    return best_i, best_c, best_feats

def decision_from_feats(feats: Dict[str,float], tf_th: float, cj_th: float) -> Tuple[bool, float]:
    anchor_ok  = (feats["names"] >= 1.0) and (feats["nums"] >= 1.0 or feats["years"] >= 1.0)
    lexical_ok = (feats["tfidf"] >= tf_th) or (feats["char3"] >= cj_th)
    L = max(feats["tfidf"], feats["char3"])
    return bool(anchor_ok or lexical_ok), float(L)

# ----------------------- Main eval -----------------------

def main():
    ap = argparse.ArgumentParser(description="Deterministic support + (optional) coverage checker with normalization and BM25.")
    ap.add_argument("--answers", required=True, help="answers_*.jsonl")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)

    # support thresholds
    ap.add_argument("--tf-th", type=float, default=0.35, help="Support TF-IDF cosine threshold")
    ap.add_argument("--cj-th", type=float, default=0.28, help="Support char-3 Jaccard threshold")

    # coverage thresholds (defaults to support if unspecified)
    ap.add_argument("--emit-coverage", action="store_true", help="Also compute coverage (evidence→story)")
    ap.add_argument("--cov-tf-th", type=float, default=None, help="Coverage TF-IDF cosine threshold")
    ap.add_argument("--cov-cj-th", type=float, default=None, help="Coverage char-3 Jaccard threshold")
    ap.add_argument("--cov-out-csv", default=None, help="Path for coverage_evidence.csv")
    ap.add_argument("--cov-out-summary", default=None, help="Path for coverage_summary.csv")

    # cleaning
    ap.add_argument("--clean-context", action="store_true", help="heavy clean/dedupe (older)")
    ap.add_argument("--light-clean", action="store_true", help="light clean (recommended)")
    ap.add_argument("--no-canon", action="store_true", help="disable alias/date/number canonicalization")
    ap.add_argument("--alias-file", default=None, help="JSON {canon:[variants,...]} or [[canon,variant],...]")

    # near-miss logging
    ap.add_argument("--near-low", type=float, default=0.28, help="Near-miss L window low")
    ap.add_argument("--near-high", type=float, default=0.35, help="Near-miss L window high")
    ap.add_argument("--near-topk", type=int, default=200)
    ap.add_argument("--emit-near", action="store_true", help="Write near_misses.csv (support)")

    args = ap.parse_args()
    t0 = time.time()

    # coverage thresholds defaulting
    cov_tf = args.cov_tf_th if args.cov_tf_th is not None else args.tf_th
    cov_cj = args.cov_cj_th if args.cov_cj_th is not None else args.cj_th

    # aliases
    alias_map: Dict[str, List[str]] = DEFAULT_ALIASES.copy()
    if args.alias_file:
        try:
            raw = json.loads(Path(args.alias_file).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    alias_map[str(k).lower()] = [str(x).lower() for x in (v or [])]
            elif isinstance(raw, list):
                for pair in raw:
                    if not (isinstance(pair, list) and len(pair) == 2): continue
                    canon = str(pair[0]).lower(); var = str(pair[1]).lower()
                    alias_map.setdefault(canon, [])
                    if var not in alias_map[canon]:
                        alias_map[canon].append(var)
        except Exception:
            pass

    # read answers
    ans_path = Path(args.answers)
    print(f"ans_path : {ans_path}")
    recs: List[Dict[str, Any]] = []
    with ans_path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    recs.append(obj)
                else:
                    # skip non-dict JSON (e.g., a stray string)
                    continue
            except Exception:
                # ignore bad lines
                continue

    # output paths
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary)

    # coverage outputs (defaults next to out_csv/summary)
    cov_out_csv = Path(args.cov_out_csv) if args.cov_out_csv else out_csv.with_name("coverage_evidence.csv")
    cov_out_summary = Path(args.cov_out_summary) if args.cov_out_summary else out_summary.with_name("coverage_summary.csv")

    # guard: choose exactly one cleaner
    if args.clean_context and args.light_clean:
        raise SystemExit("Choose only one: --clean-context (heavy) OR --light-clean (recommended).")

    support_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for rec in recs:
        bidx = rec.get("beat_index")
        bttl = rec.get("beat_title", f"Beat {bidx}")
        text_raw = rec.get("text") or ""

        # prepare contexts
        ctx_raw = [strip_noise(x) for x in (rec.get("context_lines") or []) if strip_noise(x)]
        if args.light_clean:
            ctx_pre = light_clean_context(ctx_raw)
        elif args.clean_context:
            ctx_pre = normalize_context(ctx_raw)
        else:
            ctx_pre = ctx_raw

        # canonicalize
        if not args.no_canon:
            ctx_clean = [canonicalize_text(c, alias_map) for c in ctx_pre]
            text = canonicalize_text(text_raw, alias_map)
        else:
            ctx_clean = [strip_noise(c) for c in ctx_pre]
            text = strip_noise(text_raw)

        sents = split_sents(text)

        # -------- SUPPORT (sentence -> context) --------
        for si, s in enumerate(sents):
            # BM25 filter contexts (optional)
            candidate_ctx = ctx_clean

            best_i, best_ctx, feats = best_match(s, candidate_ctx)
            supported, L = decision_from_feats(feats, args.tf_th, args.cj_th)
            support_rows.append({
                "beat_index": bidx, "beat_title": bttl, "sentence_idx": si,
                "sentence": s, "best_evidence": best_ctx,
                "tfidf": feats["tfidf"], "char3": feats["char3"],
                "name_overlap": feats["names"], "num_overlap": feats["nums"], "year_overlap": feats["years"],
                "L": L,
                "canon_enabled": (not args.no_canon),
                "supported": bool(supported),
            })

        # -------- COVERAGE (context -> sentence) --------
        if args.emit_coverage:
            for ci, c in enumerate(ctx_clean):
                candidate_sents = sents

                best_i, best_sent, feats = best_match(c, candidate_sents)
                covered, L = decision_from_feats(feats, cov_tf, cov_cj)
                coverage_rows.append({
                    "beat_index": bidx, "beat_title": bttl, "evidence_idx": ci,
                    "evidence": c, "best_sentence": best_sent,
                    "tfidf": feats["tfidf"], "char3": feats["char3"],
                    "name_overlap": feats["names"], "num_overlap": feats["nums"], "year_overlap": feats["years"],
                    "L": L, "covered": bool(covered),
                    "canon_enabled": (not args.no_canon),
                })

    # ---------- Write SUPPORT outputs ----------
    df_sup = pd.DataFrame(support_rows).sort_values(["beat_index","sentence_idx"]).reset_index(drop=True)
    df_sup.to_csv(out_csv, index=False, encoding="utf-8")

    if df_sup.empty:
        summ_sup = pd.DataFrame(columns=["beat_index","beat_title","sentences","supported_sentences","support_pct"])
        tot = sup = 0
    else:
        summ_sup = (
            df_sup.groupby(["beat_index","beat_title"])
                  .agg(sentences=("sentence_idx","nunique"),
                       supported_sentences=("supported","sum"),
                       support_pct=("supported", lambda x: 100.0 * x.mean()))
                  .reset_index()
                  .sort_values("beat_index")
        )
        tot = int(df_sup.shape[0]); sup = int(df_sup["supported"].sum())

    summ_sup.to_csv(out_summary, index=False, encoding="utf-8")

    # near-misses (support)
    if getattr(args, "emit_near", False) and not df_sup.empty:
        near = df_sup[(~df_sup["supported"]) & (df_sup["L"] >= args.near_low) & (df_sup["L"] < args.near_high)].copy()
        near = near.sort_values(["L", "beat_index", "sentence_idx"], ascending=[False, True, True]).head(args.near_topk)
        near_cols = ["beat_index", "beat_title", "sentence_idx", "L", "tfidf", "char3",
                     "name_overlap", "num_overlap", "year_overlap", "sentence", "best_evidence"]
        near_path = Path(args.out_csv).with_name("near_misses.csv")
        near.to_csv(near_path, index=False, encoding="utf-8", columns=[c for c in near_cols if c in near.columns])
        print(f"[DET] wrote {near_path} (near-miss window [{args.near_low:.2f}, {args.near_high:.2f}))")

    # ---------- Write COVERAGE outputs (optional) ----------
    if args.emit_coverage:
        df_cov = pd.DataFrame(coverage_rows).sort_values(["beat_index","evidence_idx"]).reset_index(drop=True)
        df_cov.to_csv(cov_out_csv, index=False, encoding="utf-8")

        if df_cov.empty:
            summ_cov = pd.DataFrame(columns=["beat_index","beat_title","evidence_lines","covered_evidence","coverage_pct"])
            ev_tot = ev_cov = 0
        else:
            summ_cov = (
                df_cov.groupby(["beat_index","beat_title"])
                      .agg(evidence_lines=("evidence_idx","nunique"),
                           covered_evidence=("covered","sum"),
                           coverage_pct=("covered", lambda x: 100.0 * x.mean()))
                      .reset_index()
                      .sort_values("beat_index")
            )
            ev_tot = int(df_cov.shape[0]); ev_cov = int(df_cov["covered"].sum())
        summ_cov.to_csv(cov_out_summary, index=False, encoding="utf-8")
    else:
        df_cov = None
        ev_tot = ev_cov = 0

    # ---------- meta + console ----------
    meta = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "answers_path": str(ans_path),
        "tf_th": args.tf_th, "cj_th": args.cj_th,
        "cov_tf_th": cov_tf, "cov_cj_th": cov_cj,
        "canon_enabled": (not args.no_canon),
        "clean_context": bool(args.clean_context), "light_clean": bool(args.light_clean),
        "sentences_total": int(df_sup.shape[0]) if not df_sup.empty else 0,
        "sentences_supported": int(df_sup["supported"].sum()) if not df_sup.empty else 0,
        "evidence_total": ev_tot, "evidence_covered": ev_cov,
        "duration_sec": round(time.time() - t0, 3),
    }
    out_summary.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[DET] wrote {out_csv}")
    print(f"[DET] wrote {out_summary}")
    if args.emit_coverage:
        print(f"[DET] wrote {cov_out_csv}")
        print(f"[DET] wrote {cov_out_summary}")
    pct_sup = (100.0*sup/max(1,tot))
    if args.emit_coverage:
        pct_cov = (100.0*ev_cov/max(1,ev_tot))
        print(f"[DET] support: {sup}/{tot} ({pct_sup:.1f}%)  coverage: {ev_cov}/{ev_tot} ({pct_cov:.1f}%)  "
              f"(canon={'on' if not args.no_canon else 'off'})")
    else:
        print(f"[DET] support: {sup}/{tot} ({pct_sup:.1f}%)  (canon={'on' if not args.no_canon else 'off'})")

if __name__ == "__main__":
    main()
