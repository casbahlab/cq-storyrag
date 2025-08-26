#!/usr/bin/env python3
# support_ctx_reset.py — deterministic, model-free support checker
# Adds: (A) alias/date/number normalization, (B) optional BM25 candidate filter
# Fixes: year regex, persist L in rows, preserve chosen clean context during canonicalization,
#        early mutual-exclusion check for --clean-context vs --light-clean.

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------- Regexes & basic utils -----------------------

SENT_SPLIT   = re.compile(r'(?<=[.!?])\s+')
TOKEN_RX     = re.compile(r"[A-Za-z0-9']+")
PROPER_RX    = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
YEAR_RX      = re.compile(r"\b(?:19|20)\d{2}\b")       # non-capturing, returns full year
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
    "john f. kennedy stadium": ["jfk stadium", "john kennedy stadium"],
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
    return [m.group(0) for m in YEAR_RX.finditer(s or "")]

def extract_numbers(s: str) -> List[str]:
    return NUM_RX.findall(s or "")

# ----------------------- Canonicalization (A) -----------------------

def _normalize_dates(text: str) -> str:
    """Add ISO forms for common date phrases (keep originals)."""
    t = text

    # "13 July 1985" or "13 Jul 1985"
    pat1 = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b")
    # "July 13, 1985"
    pat2 = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b")

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
    # support simple patterns: "two billion", "twenty five thousand"
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
    """Convert common spelled counts with scales to digits, keep both forms."""
    toks = re.findall(r"[A-Za-z]+|\d+|[^\w\s]", text)
    out = []
    i = 0
    while i < len(toks):
        best = None
        for L in range(2, 6):  # up to 5 tokens
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

# --- NEW: identifier normalization for KG-style names ---

_CAMEL_RX   = re.compile(r'(?<!^)(?=[A-Z])')   # split CamelCase
_TAIL_NUM_RX = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\s+([0-9])\b')

def _split_camel_and_digits(text: str) -> str:
    """
    Insert spaces into CamelCase and between letters+digits:
      'RogerTaylor2' -> 'Roger Taylor 2'
    """
    toks = []
    for tok in re.findall(r"[A-Za-z0-9]+|[^\w\s]", text):
        if tok.isalpha() and tok.lower() != tok and tok.upper() != tok:
            tok = " ".join(_CAMEL_RX.split(tok))  # RogerTaylor -> Roger Taylor
        # letters followed by digits: 'Taylor2' -> 'Taylor 2'
        tok = re.sub(r'([A-Za-z])(\d+)\b', r'\1 \2', tok)
        toks.append(tok)
    return re.sub(r"\s+", " ", " ".join(toks)).strip()

def _drop_disambig_tails(text: str) -> str:
    """
    Drop single-digit disambiguators that are typical of KG URIs:
      'Roger Taylor 2' -> 'Roger Taylor'
    Avoid touching years (four digits) or real counts.
    """
    return _TAIL_NUM_RX.sub(r'\1', text)

def normalize_identifiers(text: str) -> str:
    # Keep both forms to maximize overlap: original + normalized
    split = _split_camel_and_digits(text)
    dropped = _drop_disambig_tails(split)
    if dropped != split:
        return f"{text} {split} {dropped}"
    if split != text:
        return f"{text} {split}"
    return text


def canonicalize_text(text: str, alias_map: Dict[str, List[str]]) -> str:
    t = strip_noise(text)
    t = _apply_aliases(t, alias_map)
    t = _normalize_dates(t)
    t = _normalize_spelled_numbers(t)
    t = normalize_identifiers(t)  # <-- add this line
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

# --- light clean helpers ---
PREFIX_RX2    = re.compile(r'^\s*:?\s*(KG|WEB)\s*:\s*', re.I)
URL_OR_IRI_RX = re.compile(r'https?://\S+|<[^>]+>')

def light_clean_line(s: str) -> str:
    if not s: return ""
    s = str(s).replace("“", '"').replace("”", '"').replace("’", "'")
    s = URL_OR_IRI_RX.sub("", s)
    s = PREFIX_RX2.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def light_clean_context(ctxs: List[str]) -> List[str]:
    out, seen = [], set()
    for c in ctxs or []:
        cc = light_clean_line(c)
        if not cc: continue
        key = cc.lower()
        if key in seen: continue  # exact dup only
        seen.add(key)
        out.append(cc)
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

# ----------------------- BM25 (B) — optional candidate filter -----------------------

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

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked

# ----------------------- Decision rule -----------------------

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

        sig = max(tf, cj) + 0.06*names_ok + 0.04*max(nums_ok, years_ok)

        if sig > best_sig:
            best_sig = sig
            best_ctx = c
            best_feats = {"tfidf": tf, "char3": cj, "names": names_ok, "nums": nums_ok, "years": years_ok}

    anchor_ok = (best_feats["names"] >= 1.0) and (best_feats["nums"] >= 1.0 or best_feats["years"] >= 1.0)
    lexical_ok = (best_feats["tfidf"] >= tf_th) or (best_feats["char3"] >= cj_th)
    return bool(anchor_ok or lexical_ok), best_ctx, best_feats



CAMEL_RX = re.compile(r'(?<!^)(?=[A-Z])')  # split Before Caps
DASHLIKE_RX = re.compile(r"[–—−]+")        # normalize dash variants

def _split_camel(s: str) -> str:
    # Only split longish tokens to avoid “UK” → “U K”
    parts = []
    for tok in TOKEN_RX.findall(s):
        if len(tok) >= 6 and tok.lower() == tok:  # already lowercase word
            parts.append(tok)
        elif len(tok) >= 6 and re.search(r"[A-Z][a-z]", tok):
            parts.append(" ".join(CAMEL_RX.split(tok)))
        else:
            parts.append(tok)
    # Re-stitch into text-ish string
    out = s
    for t in set(TOKEN_RX.findall(s)):
        if len(t) >= 6 and re.search(r"[A-Z][a-z]", t):
            out = re.sub(rf"\b{re.escape(t)}\b", " ".join(CAMEL_RX.split(t)), out)
    return out

def _smooth_punct_names(s: str) -> str:
    # unify dash variants
    s = DASHLIKE_RX.sub("-", s)
    # Run-D.M.C. family → a single canonical surface "run d m c"
    s = re.sub(r"(?i)\brun[\s\.-]*d[\s\.-]*m[\s\.-]*c\b", "run d m c", s)
    # collapse repeated punctuation gaps
    s = re.sub(r"[.\-_/]{1,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _light_verbalize_kg_edges(s: str) -> str:
    # very conservative: only a couple of predicates
    s = s.replace("→ location →", " at ")
    s = s.replace("-> location ->", " at ")
    s = s.replace("→ member →", " member ")
    return s


# ----------------------- Main eval -----------------------

def main():
    ap = argparse.ArgumentParser(description="Deterministic support checker with optional normalization and BM25 filtering.")
    ap.add_argument("--answers", required=True, help="answers_*.jsonl")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)

    # core thresholds
    ap.add_argument("--clean-context", action="store_true", help="clean/dedupe context lines first (heavy)")
    ap.add_argument("--light-clean", action="store_true", help="light clean (recommended): strip KG/WEB prefix, URLs/IRIs, collapse whitespace")
    ap.add_argument("--tf-th", type=float, default=0.35, help="TF-IDF cosine threshold")
    ap.add_argument("--cj-th", type=float, default=0.28, help="char-3 Jaccard threshold")

    # normalization
    ap.add_argument("--no-canon", action="store_true", help="disable alias/date/number canonicalization")
    ap.add_argument("--alias-file", default=None, help="JSON mapping for aliases. Format: {canon: [variants,...], ...} or [[canon,variant],...]")

    # BM25 candidate filter
    ap.add_argument("--bm25-mode", choices=["off","filter"], default="off")
    ap.add_argument("--bm25-k1", type=float, default=1.2)
    ap.add_argument("--bm25-b", type=float, default=0.50, help="Try 0.25 for KG; 0.50 for Hybrid")
    ap.add_argument("--bm25-topk", type=int, default=20)
    ap.add_argument("--bm25-log-margin", type=float, default=0.15, help="Only for logging/audits")

    # near-miss reporting
    ap.add_argument("--near-low", type=float, default=0.28, help="Lower bound for near-miss L window (inclusive).")
    ap.add_argument("--near-high", type=float, default=0.35, help="Upper bound for near-miss L window (exclusive).")
    ap.add_argument("--near-topk", type=int, default=200, help="Max rows to write to near_misses.csv (sorted by L desc).")
    ap.add_argument("--emit-near", action="store_true", help="Write near_misses.csv next to out-csv.")

    args = ap.parse_args()

    # Early mutual-exclusion check
    if args.clean_context and args.light_clean:
        raise SystemExit("Choose only one: --clean-context (heavy) OR --light-clean (recommended).")

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
                recs.append(json.loads(ln))
            except Exception:
                pass

    rows: List[Dict[str, Any]] = []

    #print(f"recs : {recs}")
    for rec in recs:
        bidx = rec.get("beat_index")
        bttl = rec.get("beat_title", f"Beat {bidx}")
        text_raw = rec.get("text") or ""

        # --- prepare context once ---
        ctx_raw = [strip_noise(x) for x in (rec.get("context_lines") or []) if strip_noise(x)]
        if args.light_clean:
            base_ctx = light_clean_context(ctx_raw)
        elif args.clean_context:
            base_ctx = normalize_context(ctx_raw)
        else:
            base_ctx = ctx_raw

        # --- canonicalize text + chosen context consistently ---
        if not args.no_canon:
            ctx  = [canonicalize_text(c, alias_map) for c in base_ctx]
            text = canonicalize_text(text_raw, alias_map)
        else:
            ctx  = base_ctx
            text = strip_noise(text_raw)

        sents = split_sents(text)

        for si, s in enumerate(sents):
            candidate_ctx = ctx

            bm25_best = bm25_second = bm25_margin = float("nan")
            if args.bm25_mode == "filter" and candidate_ctx:
                ranked = bm25_rank(s, candidate_ctx, k1=args.bm25_k1, b=args.bm25_b)
                if ranked:
                    bm25_best = ranked[0][1]
                    if len(ranked) >= 2:
                        bm25_second = ranked[1][1]
                        if bm25_best > 0:
                            bm25_margin = (bm25_best - bm25_second) / max(1e-9, bm25_best)
                keep = [idx for idx,_ in ranked[:max(1, min(args.bm25_topk, len(ranked)))]]
                candidate_ctx = [candidate_ctx[i] for i in keep]

            supported, best_ctx, feats = det_supported(s, candidate_ctx, tf_th=args.tf_th, cj_th=args.cj_th)
            L = max(feats["tfidf"], feats["char3"])  # lexical support score

            rows.append({
                "beat_index": bidx, "beat_title": bttl, "sentence_idx": si,
                "sentence": s, "best_evidence": best_ctx,
                "tfidf": feats["tfidf"], "char3": feats["char3"], "L": L,   # keep L in rows
                "name_overlap": feats["names"], "num_overlap": feats["nums"], "year_overlap": feats["years"],
                "bm25_best": bm25_best, "bm25_second": bm25_second, "bm25_margin": bm25_margin,
                "bm25_mode": args.bm25_mode, "bm25_b": args.bm25_b, "bm25_k1": args.bm25_k1, "bm25_topk": args.bm25_topk,
                "canon_enabled": (not args.no_canon),
                "clean_context": bool(args.clean_context), "light_clean": bool(args.light_clean),
                "supported": bool(supported),
            })

    df = pd.DataFrame(rows).sort_values(["beat_index","sentence_idx"]).reset_index(drop=True)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    if df.empty:
        summ = pd.DataFrame(columns=["beat_index","beat_title","sentences","supported_sentences","support_pct"])
        tot = sup = 0
    else:
        summ = (
            df.groupby(["beat_index","beat_title"])
              .agg(sentences=("sentence_idx","nunique"),
                   supported_sentences=("supported","sum"),
                   support_pct=("supported", lambda x: 100.0 * x.mean()))
              .reset_index().sort_values("beat_index")
        )
        tot = int(df.shape[0]); sup = int(df["supported"].sum())

        # per-beat stats for unsupported L (optional but useful)
        unsup = df[~df["supported"]].copy()
        if not unsup.empty and "L" in unsup.columns:
            agg_unsup = (unsup.groupby(["beat_index", "beat_title"])
                         .agg(unsup_sentences=("sentence_idx", "nunique"),
                              L_mean_unsup=("L", "mean"),
                              L_median_unsup=("L", "median"))
                         .reset_index())
            summ = summ.merge(agg_unsup, on=["beat_index", "beat_title"], how="left")
        else:
            summ["unsup_sentences"] = 0
            summ["L_mean_unsup"] = float("nan")
            summ["L_median_unsup"] = float("nan")

    out_summary = Path(args.out_summary)
    summ.to_csv(out_summary, index=False, encoding="utf-8")

    meta = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "answers_path": str(ans_path),
        "tf_th": args.tf_th, "cj_th": args.cj_th,
        "canon_enabled": (not args.no_canon),
        "clean_context": bool(args.clean_context), "light_clean": bool(args.light_clean),
        "bm25_mode": args.bm25_mode, "bm25_k1": args.bm25_k1, "bm25_b": args.bm25_b, "bm25_topk": args.bm25_topk,
        "sentences_total": tot, "sentences_supported": sup,
    }
    out_summary.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if getattr(args, "emit_near", False) and not df.empty:
        near = df[(~df["supported"]) & (df["L"] >= args.near_low) & (df["L"] < args.near_high)].copy()
        near = near.sort_values(["L", "beat_index", "sentence_idx"], ascending=[False, True, True]).head(args.near_topk)
        near_cols = ["beat_index", "beat_title", "sentence_idx", "L", "tfidf", "char3",
                     "name_overlap", "num_overlap", "year_overlap", "sentence", "best_evidence"]
        near_path = Path(args.out_csv).with_name("near_misses.csv")
        near.to_csv(near_path, index=False, encoding="utf-8", columns=[c for c in near_cols if c in near.columns])
        print(f"[DET] wrote {near_path} (near-miss window [{args.near_low:.2f}, {args.near_high:.2f}))")

    print(f"[DET] wrote {out_csv}")
    print(f"[DET] wrote {out_summary}")
    pct = (100.0*sup/max(1,tot))
    print(f"[DET] support: {sup}/{tot} ({pct:.1f}%)  (canon={'on' if not args.no_canon else 'off'}, bm25={args.bm25_mode})")

if __name__ == "__main__":
    main()
