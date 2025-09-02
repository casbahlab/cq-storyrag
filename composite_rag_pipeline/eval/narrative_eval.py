#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
narrative_eval.py  —  Narrative-only quality metrics (deterministic-first)
Outputs:
  - narrative_summary.csv (overall)
  - narrative_per_section.csv (per-beat metrics incl. optional NQI, NQI_LITE)
  - narrative_report.md (compact summary)
  - narrative_per_section_annotated.csv (if --annotate)
  - narrative_report_annotated.md (if --annotate)

Usage:
  python3 narrative_eval.py -i story.md -o out_dir --beats auto --with-coherence --annotate --nqi-lite \
    --neardupe-th 0.85 --domain-stopwords live aid concert performance wembley philadelphia
"""

import os, re, math, csv, argparse, statistics
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

############################################
# Regexes, lists, constants
############################################

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=(?:["“”\'\)]*\s*)?[A-Z0-9])')
WORD_RE = re.compile(r"[A-Za-z0-9']+")
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

STOPWORDS = set("""
a an the and or but if while of for to in on at from by with without over under
this that these those is are was were be been being am do does did doing have has had
i you he she it we they them him her us our your their my mine yours his hers ours theirs
as into about above below up down out off again further then once here there when where
why how all any both each few more most other some such no nor not only own same so than
too very s t can will just don don’t should now
""".split())

FILLERS = set(["very","really","just","quite","kind","sort","kind of","sort of","basically","actually","literally","simply","somewhat","rather"])

TRANSITION_CUES = [
    "however","therefore","meanwhile","later","then","next","finally","first","second",
    "in contrast","overall","in summary","ultimately","afterward","beforehand","subsequently",
    "additionally","moreover","nonetheless","nevertheless"
]

CONCLUDE_CUES = ["overall","in the end","ultimately","in summary","to sum up","in conclusion"]
LEAD_STARTERS = ["in ","on ","at "]  # e.g., "In 1985", "On July 13", "At Wembley"

BE_VERBS = r"(?:am|is|are|was|were|be|been|being)"
PASSIVE_RE = re.compile(rf"\b{BE_VERBS}\b\s+\w+(?:ed|en)\b", re.IGNORECASE)

REL_LATER = {"later","afterwards","after","subsequently"}
REL_EARLIER = {"earlier","before","previously","prior"}

PROPN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:of|the|and)\s+)?)+(?:[A-Z][a-z]+)\b")

############################################
# Basic text utilities
############################################

def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sents = SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]

def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in WORD_RE.finditer(text or "")]

def lower_tokens(tokens: List[str]) -> List[str]:
    return [t.lower() for t in tokens]

def simple_lemma(token: str) -> str:
    t = token.lower()
    for suf in ["'s","’s"]:
        if t.endswith(suf): t = t[:-2]
    if len(t) > 4 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 3 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 2 and t.endswith("s"):
        t = t[:-1]
    return t

def content_words(tokens: List[str], extra_stop:set) -> List[str]:
    out = []
    for t in tokens:
        tl = t.lower()
        if tl in STOPWORDS or tl in extra_stop:
            continue
        if not tl.isalpha():
            continue
        out.append(simple_lemma(tl))
    return out

def char_trigrams(s: str) -> set:
    s = re.sub(r"\s+", " ", (s or "").lower()).strip()
    return set(s[i:i+3] for i in range(len(s)-2))

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

############################################
# Readability
############################################

VOWELS = "aeiouy"

def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    if not w: return 0
    count, prev_is_v = 0, False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_is_v:
            count += 1
        prev_is_v = is_v
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def flesch_reading_ease(total_sent, total_words, total_syll):
    if total_sent == 0 or total_words == 0:
        return 0.0
    return 206.835 - 1.015 * (total_words / total_sent) - 84.6 * (total_syll / total_words)

def flesch_kincaid_grade(total_sent, total_words, total_syll):
    if total_sent == 0 or total_words == 0:
        return 0.0
    return 0.39 * (total_words / total_sent) + 11.8 * (total_syll / total_words) - 15.59

############################################
# Entities, temporal, numbers
############################################

def extract_entities(sentence: str) -> List[str]:
    ents = []
    for m in PROPN_RE.finditer(sentence or ""):
        ent = m.group(0).strip()
        if ent in ["In","On","At","The"]:  # avoid trivial sentence starters
            continue
        ents.append(ent)
    ents.extend(re.findall(r"\b[A-Z]{2,}\b", sentence or ""))  # acronyms
    return ents

def canonical_entity(ent: str) -> str:
    toks = (ent or "").split()
    root = toks[-1] if len(toks) >= 2 else (ent or "")
    root = re.sub(r"[^A-Za-z0-9]", "", root).lower()
    return root

def parse_years(s: str) -> List[int]:
    return [int(y) for y in YEAR_RE.findall(s or "")]

NUM_NOUN_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s+([A-Za-z][a-z\-]+)\b")

def number_conflict_in_window(sentences: List[str], idx: int, window: int = 2) -> int:
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    noun_to_vals = defaultdict(set)
    for i in range(start, end):
        for m in NUM_NOUN_RE.finditer(sentences[i]):
            num, noun = m.groups()
            try:
                val = float(num.replace(",", ""))
            except:
                continue
            noun_lemma = simple_lemma(noun)
            noun_to_vals[noun_lemma].add(val)
    return sum(1 for v in noun_to_vals.values() if len(v) > 1)

############################################
# TF-IDF (optional coherence)
############################################

def build_vocab(docs: List[List[str]]) -> Dict[str, int]:
    vocab = {}
    for doc in docs:
        for t in doc:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

def tfidf_vectors(docs: List[List[str]], vocab: Dict[str,int]) -> List[List[float]]:
    N = len(docs)
    df = [0]*len(vocab)
    for doc in docs:
        seen = set(doc)
        for t in seen:
            df[vocab[t]] += 1
    idf = [math.log((N + 1) / (dfi + 1)) + 1.0 for dfi in df]
    vecs = []
    for doc in docs:
        counts = Counter(doc)
        vec = [0.0]*len(vocab)
        max_tf = max(counts.values()) if counts else 1
        for t, c in counts.items():
            j = vocab[t]
            tf = c / max_tf
            vec[j] = tf * idf[j]
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        vecs.append([v/norm for v in vec])
    return vecs

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

############################################
# Metrics
############################################

def ngrams(tokens: List[str], n: int) -> List[Tuple[str,...]]:
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

def p95(xs: List[float]) -> float:
    if not xs: return 0.0
    xs_sorted = sorted(xs)
    k = int(math.ceil(0.95*len(xs_sorted))) - 1
    k = max(0, min(k, len(xs_sorted)-1))
    return float(xs_sorted[k])

def iqr_over_median(values: List[int]) -> float:
    if not values: return 0.0
    try:
        q = statistics.quantiles(values, n=4, method="inclusive")
        q1, q3 = q[0], q[2]
        med = statistics.median(values) or 1.0
        return (q3 - q1) / med if med else 0.0
    except Exception:
        return 0.0

def trigram_rep_rate(tokens: List[str]) -> float:
    tris = ngrams(tokens, 3)
    total = len(tris)
    if total == 0: return 0.0
    counts = Counter(tris)
    repeated_occ = sum(c-1 for c in counts.values() if c > 1)
    return repeated_occ / total

def distinct_n(tokens: List[str], n: int) -> float:
    grams = ngrams(tokens, n)
    total = len(grams)
    return (len(set(grams))/total) if total else 0.0

def near_dup_adjacent_rate(sentences: List[str], th: float) -> float:
    if len(sentences) < 2: return 0.0
    cnt = 0
    for i in range(len(sentences)-1):
        sim = jaccard(char_trigrams(sentences[i]), char_trigrams(sentences[i+1]))
        if sim >= th:
            cnt += 1
    return cnt / (len(sentences)-1)

def max_repeat_streak(sentences: List[str], th: float) -> int:
    if not sentences: return 0
    longest, cur = 1, 1
    for i in range(1, len(sentences)):
        sim = jaccard(char_trigrams(sentences[i-1]), char_trigrams(sentences[i]))
        if sim >= th:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest

def entity_overlap_adjacent(sentences: List[str]) -> Tuple[float,float]:
    overlaps = []
    for i in range(len(sentences)-1):
        e1 = [canonical_entity(e) for e in extract_entities(sentences[i])]
        e2 = [canonical_entity(e) for e in extract_entities(sentences[i+1])]
        overlaps.append(jaccard(set(e1), set(e2)))
    if not overlaps: return (0.0, 0.0)
    overlaps_sorted = sorted(overlaps)
    p25_idx = max(0, int(len(overlaps_sorted)*0.25)-1)
    return (sum(overlaps)/len(overlaps), overlaps_sorted[p25_idx])

def content_overlap_adjacent(sentences: List[str], extra_stop:set) -> Tuple[float,float]:
    overlaps = []
    for i in range(len(sentences)-1):
        w1 = content_words(tokenize(sentences[i]), extra_stop)
        w2 = content_words(tokenize(sentences[i+1]), extra_stop)
        overlaps.append(jaccard(set(w1), set(w2)))
    if not overlaps: return (0.0, 0.0)
    overlaps_sorted = sorted(overlaps)
    p25_idx = max(0, int(len(overlaps_sorted)*0.25)-1)
    return (sum(overlaps)/len(overlaps), overlaps_sorted[p25_idx])

def transition_rate(sentences: List[str]) -> float:
    if not sentences: return 0.0
    cnt = 0
    for s in sentences:
        start = (s or "").strip().lower()
        if any(start.startswith(cue) for cue in TRANSITION_CUES):
            cnt += 1
    return cnt / len(sentences)

def passive_ratio(sentences: List[str]) -> float:
    if not sentences: return 0.0
    hits = sum(1 for s in sentences if PASSIVE_RE.search(s or ""))
    return hits / len(sentences)

def type_token_ratio(tokens: List[str]) -> float:
    return (len(set(t.lower() for t in tokens)) / len(tokens)) if tokens else 0.0

def lexical_density(tokens: List[str], extra_stop:set) -> float:
    if not tokens: return 0.0
    cnt = sum(1 for t in tokens if t.isalpha() and t.lower() not in STOPWORDS and t.lower() not in extra_stop)
    return cnt / len(tokens)

def lead_marker(first_sentence: str) -> bool:
    s = (first_sentence or "").strip()
    sl = s.lower()
    if any(sl.startswith(prefix) for prefix in LEAD_STARTERS): return True
    if YEAR_RE.search(s): return True
    if re.match(r"^(At|In|On)\s+[A-Z]", s): return True
    return False

def conclude_marker(last_sentence: str) -> bool:
    s = (last_sentence or "").strip().lower()
    if any(cue in s for cue in CONCLUDE_CUES): return True
    if s.startswith("overall"): return True
    return False

def entity_alias_switches_per_1k(sentences: List[str], tokens: List[str]) -> float:
    root_to_forms = defaultdict(set)
    for s in sentences:
        for e in extract_entities(s):
            root = canonical_entity(e)
            root_to_forms[root].add(e)
    switches = sum(max(0, len(forms)-1) for forms in root_to_forms.values())
    words = max(1, len(tokens))
    return switches / words * 1000.0

def temporal_order_violations(sentences: List[str]) -> int:
    vio = 0
    for i in range(1, len(sentences)):
        prev, cur = sentences[i-1], sentences[i]
        y_prev, y_cur = parse_years(prev), parse_years(cur)
        if "later" in (cur or "").lower() and y_prev and y_cur and max(y_cur) < min(y_prev):
            vio += 1
        if "earlier" in (cur or "").lower() and y_prev and y_cur and min(y_cur) > max(y_prev):
            vio += 1
        if any(w in (cur or "").lower() for w in REL_LATER) and any(w in (prev or "").lower() for w in REL_EARLIER):
            vio += 1
    last_year = None
    for s in sentences:
        ys = parse_years(s)
        if ys:
            y = min(ys)
            if last_year is not None:
                sl = (s or "").lower()
                if "later" in sl and y < last_year: vio += 1
                if "earlier" in sl and y > last_year: vio += 1
            last_year = y
    return vio

def number_conflicts(sentences: List[str]) -> int:
    return sum(number_conflict_in_window(sentences, i, window=2) for i in range(len(sentences)))

def conciseness_proxies(tokens: List[str], sentences: List[str], extra_stop:set) -> Dict[str, float]:
    non_stop = sum(1 for t in tokens if t.lower() not in STOPWORDS and t.lower() not in extra_stop and t.isalpha())
    density = (non_stop / len(tokens)) if tokens else 0.0
    filler_cnt = sum(1 for t in tokens if t.lower() in FILLERS)
    filler_rate = (filler_cnt / len(tokens)) if tokens else 0.0
    clause_counts = []
    for s in sentences:
        c = 1
        c += s.count(",")
        c += len(re.findall(r"\b(and|but|or)\b", (s or "").lower()))
        c += s.count(";")
        c += len(re.findall(r"\b(which|that)\b", (s or "").lower()))
        clause_counts.append(c)
    avg_clauses = sum(clause_counts)/len(clause_counts) if clause_counts else 0.0
    return {
        "non_stopword_density": density,
        "filler_rate": filler_rate,
        "avg_clauses_per_sent": avg_clauses
    }

############################################
# Beat parsing
############################################

MD_HEAD_RE = re.compile(r"^\s{0,3}#{2,3}\s+(.+)$", re.MULTILINE)

def parse_beats_auto(text: str) -> List[Dict[str,str]]:
    beats = []
    matches = list(MD_HEAD_RE.finditer(text or ""))
    if not matches:
        return [{"id":"B1","title":"Beat 1","text":(text or "").strip()}]
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(text) and i+1 < len(matches) else len(text)
        title = m.group(1).strip()
        body = (text[start:end] if start <= len(text) else "").strip()
        if body:
            beats.append({"id": f"B{i+1}", "title": title, "text": body})
    if not beats:
        beats = [{"id":"B1","title":"Beat 1","text":(text or "").strip()}]
    return beats

def split_equal_beats(text: str, k: int) -> List[Dict[str,str]]:
    sents = sentence_split(text or "")
    if not sents:
        return [{"id":"B1","title":"Beat 1","text":(text or "").strip()}]
    n = len(sents)
    size = max(1, n // max(1,k))
    beats, idx, b = [], 0, 1
    while idx < n:
        chunk = sents[idx:idx+size]
        beats.append({"id":f"B{b}", "title":f"Beat {b}", "text":" ".join(chunk)})
        idx += size; b += 1
    return beats

############################################
# Compute metrics for text and structure
############################################

def compute_metrics_for_text(text: str, extra_stop:set, near_th: float) -> Dict[str, Any]:
    sentences = sentence_split(text)
    tokens = tokenize(text)
    tokens_l = lower_tokens(tokens)
    # redundancy
    tgr = trigram_rep_rate(tokens_l)
    novel_sent_ratio = (len(set(s.strip().lower() for s in sentences)) / len(sentences)) if sentences else 0.0
    near_dup_rate = near_dup_adjacent_rate(sentences, near_th)
    d1 = distinct_n(tokens_l, 1)
    d2 = distinct_n(tokens_l, 2)
    max_streak = max_repeat_streak(sentences, near_th)
    # cohesion
    ent_mean, ent_p25 = entity_overlap_adjacent(sentences)
    cont_mean, cont_p25 = content_overlap_adjacent(sentences, extra_stop)
    trans_rate = transition_rate(sentences)
    # readability
    syll = sum(count_syllables(w) for w in tokens)
    fre = flesch_reading_ease(len(sentences), len(tokens), syll)
    fk = flesch_kincaid_grade(len(sentences), len(tokens), syll)
    sent_lens = [len(tokenize(s)) for s in sentences]
    sent_avg = (sum(sent_lens)/len(sent_lens)) if sent_lens else 0.0
    sent_p95 = p95(sent_lens) if sent_lens else 0.0
    pass_ratio = passive_ratio(sentences)
    ttr = type_token_ratio(tokens)
    lex_den = lexical_density(tokens, extra_stop)
    # consistency
    alias_switches = entity_alias_switches_per_1k(sentences, tokens)
    temp_vios = temporal_order_violations(sentences)
    num_conf = number_conflicts(sentences)
    # conciseness
    conc = conciseness_proxies(tokens, sentences, extra_stop)
    return {
        "trigram_rep_rate": tgr,
        "novel_sentence_ratio": novel_sent_ratio,
        "adj_near_dup_rate": near_dup_rate,
        "distinct_1": d1,
        "distinct_2": d2,
        "max_repeat_span": max_streak,
        "ent_overlap_mean": ent_mean,
        "ent_overlap_p25": ent_p25,
        "content_overlap_mean": cont_mean,
        "content_overlap_p25": cont_p25,
        "transition_rate": trans_rate,
        "flesch_reading_ease": fre,
        "fk_grade": fk,
        "sent_len_avg": sent_avg,
        "sent_len_p95": sent_p95,
        "passive_ratio": pass_ratio,
        "type_token_ratio": ttr,
        "lexical_density": lex_den,
        "alias_switches_per_entity_per_1k": alias_switches,
        "temporal_order_violations": temp_vios,
        "number_conflicts": num_conf,
        "non_stopword_density": conc["non_stopword_density"],
        "filler_rate": conc["filler_rate"],
        "avg_clauses_per_sent": conc["avg_clauses_per_sent"]
    }

def compute_structure_metrics(beats: List[Dict[str,str]]) -> Dict[str, Any]:
    non_empty = sum(1 for b in beats if (b.get("text","").strip()))
    non_empty_pct = non_empty / len(beats) if beats else 0.0
    beat_tokens = [len(tokenize(b.get("text",""))) for b in beats]
    len_balance = iqr_over_median(beat_tokens)
    leads = 0
    concludes = 0
    for b in beats:
        sents = sentence_split(b.get("text",""))
        if sents:
            if lead_marker(sents[0]): leads += 1
            if conclude_marker(sents[-1]): concludes += 1
    lead_rate = leads / len(beats) if beats else 0.0
    concl_rate = concludes / len(beats) if beats else 0.0
    return {
        "beats_non_empty_pct": non_empty_pct,
        "beat_len_iqr_over_median": len_balance,
        "lead_marker_rate": lead_rate,
        "conclude_marker_rate": concl_rate
    }

def optional_coherence(beats: List[Dict[str,str]], extra_stop:set) -> Dict[str,float]:
    docs = []
    for b in beats:
        toks = [simple_lemma(t) for t in tokenize(b.get("text",""))]
        toks = [t for t in toks if t and t not in STOPWORDS and t.isalpha()]
        docs.append(toks)
    if not any(docs):
        return {"topic_var": 0.0, "beat_to_global_sim_mean": 0.0}
    vocab = build_vocab(docs)
    vecs = tfidf_vectors(docs, vocab)
    dists = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            dists.append(1.0 - cosine(vecs[i], vecs[j]))
    topic_var = sum(dists)/len(dists) if dists else 0.0
    all_doc = []
    for d in docs: all_doc.extend(d)
    global_vec = tfidf_vectors([all_doc], vocab)[0]
    sims = [cosine(v, global_vec) for v in vecs] if vecs else [0.0]
    return {"topic_var": topic_var, "beat_to_global_sim_mean": sum(sims)/len(sims)}

############################################
# I/O helpers
############################################

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str,Any]]):
    if not rows: return
    # union of keys across rows so added fields (NQI/NQI_LITE) are included
    keys = set()
    for r in rows: keys |= set(r.keys())
    keys = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def write_report_md(path: str, overall: Dict[str,Any], per_beats: List[Dict[str,Any]]):
    def fmt(name, v):
        return f"{name}: {v:.3f}" if isinstance(v, float) else f"{name}: {v}"
    lines = ["# Narrative Quality Report", "", "## Overall", ""]
    for k in sorted(overall.keys()):
        lines.append(f"- {fmt(k, overall[k])}")
    lines.append("")
    lines.append("## Per Beat")
    lines.append("")
    for b in per_beats:
        lines.append(f"### {b.get('beat_id','')} – {b.get('beat_title','')}")
        for k in sorted(b.keys()):
            if k in ("beat_id","beat_title","text"): continue
            lines.append(f"- {fmt(k, b[k])}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

############################################
# Annotation flags + NQI (relative, min-max across beats)
############################################

def _flag_threeway(val, good, ok, higher_is_better=True):
    try:
        v = float(val)
    except Exception:
        return "grey"
    if higher_is_better:
        if v >= good: return "green"
        if v >= ok:   return "amber"
        return "red"
    else:
        if v <= good: return "green"
        if v <= ok:   return "amber"
        return "red"

def _combine_flags(flags):
    order = {"green": 0, "amber": 1, "red": 2, "grey": 1}
    return max(flags, key=lambda c: order.get(c, 1)) if flags else "grey"

def _suggest_for_flags(row):
    tips = []
    if row.get("redundancy_flag") in ("amber","red"):
        tips.append("Trim repetition; vary phrasing; watch 3-gram repeats.")
    if row.get("cohesion_flag") in ("amber","red"):
        tips.append("Add bridges between adjacent sentences (repeat key terms/synonyms).")
    if row.get("readability_flag") in ("amber","red"):
        tips.append("Shorten sentences; aim ~18–20 tokens; prefer concrete verbs.")
    if row.get("consistency_flag") in ("amber","red"):
        tips.append("Normalize entity names; verify dates/numbers within ±2 sentences.")
    return " ".join(tips)

def _minmax_norm(values: List[float], higher_is_better: bool) -> List[float]:
    clean = [float(v) if isinstance(v,(int,float)) or (isinstance(v,str) and v.strip()!='') else float('nan') for v in values]
    finite = [v for v in clean if not math.isnan(v)]
    if not finite or max(finite) == min(finite):
        return [0.5]*len(values)
    lo, hi = min(finite), max(finite)
    out = []
    for v in clean:
        if math.isnan(v):
            out.append(0.5)
            continue
        z = (v - lo) / (hi - lo)
        out.append(z if higher_is_better else (1.0 - z))
    return out

def annotate_perbeat_rows(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # Flags
    for r in rows:
        r["redundancy_trigrams_flag"] = _flag_threeway(r.get("trigram_rep_rate"), 0.02, 0.05, higher_is_better=False)
        r["redundancy_neardup_flag"]  = _flag_threeway(r.get("adj_near_dup_rate"), 0.02, 0.05, higher_is_better=False)
        r["redundancy_novel_flag"]    = _flag_threeway(r.get("novel_sentence_ratio"), 0.95, 0.90, higher_is_better=True)
        r["redundancy_div_flag"]      = _flag_threeway(r.get("distinct_1"), 0.45, 0.38, higher_is_better=True)
        r["redundancy_flag"]          = _combine_flags([r["redundancy_trigrams_flag"], r["redundancy_neardup_flag"], r["redundancy_novel_flag"], r["redundancy_div_flag"]])

        r["cohesion_ent_flag"]        = _flag_threeway(r.get("ent_overlap_mean"), 0.30, 0.15, higher_is_better=True)
        r["cohesion_content_flag"]    = _flag_threeway(r.get("content_overlap_mean"), 0.10, 0.05, higher_is_better=True)
        r["cohesion_trans_flag"]      = _flag_threeway(r.get("transition_rate"), 0.15, 0.08, higher_is_better=True)
        r["cohesion_flag"]            = _combine_flags([r["cohesion_ent_flag"], r["cohesion_content_flag"], r["cohesion_trans_flag"]])

        r["readability_fre_flag"]     = _flag_threeway(r.get("flesch_reading_ease"), 60.0, 30.0, higher_is_better=True)
        r["readability_len_flag"]     = _flag_threeway(r.get("sent_len_avg"), 20.0, 25.0, higher_is_better=False)
        r["readability_passive_flag"] = _flag_threeway(r.get("passive_ratio"), 0.08, 0.15, higher_is_better=False)
        r["readability_flag"]         = _combine_flags([r["readability_fre_flag"], r["readability_len_flag"], r["readability_passive_flag"]])

        r["consistency_alias_flag"]   = _flag_threeway(r.get("alias_switches_per_entity_per_1k"), 1.0, 3.0, higher_is_better=False)
        r["consistency_temp_flag"]    = _flag_threeway(r.get("temporal_order_violations"), 0.0, 1.0, higher_is_better=False)
        r["consistency_num_flag"]     = _flag_threeway(r.get("number_conflicts"), 0.0, 1.0, higher_is_better=False)
        r["consistency_flag"]         = _combine_flags([r["consistency_alias_flag"], r["consistency_temp_flag"], r["consistency_num_flag"]])

        r["suggestions"]              = _suggest_for_flags(r)

    # Relative NQI across beats (min-max per metric)
    metrics = {
        "novel_sentence_ratio": True,
        "distinct_1": True,
        "ent_overlap_mean": True,
        "content_overlap_mean": True,
        "transition_rate": True,
        "flesch_reading_ease": True,
        "trigram_rep_rate": False,
        "adj_near_dup_rate": False,
        "sent_len_avg": False,
        "passive_ratio": False,
        "alias_switches_per_entity_per_1k": False,
        "number_conflicts": False,
    }
    # prepare columns
    cols = {m: [float(r.get(m, float('nan'))) if r.get(m,"")!="" else float('nan') for r in rows] for m in metrics.keys()}
    # normalize
    norm = {m: _minmax_norm(vals, higher) for m,(vals,higher) in zip(metrics.keys(), [(cols[m], metrics[m]) for m in metrics])}
    for i in range(len(rows)):
        vals = [norm[m][i] for m in metrics]
        rows[i]["NQI"] = round(sum(vals)/len(vals)*100.0, 1)
    return rows

############################################
# NQI-Lite (absolute-banded, research-grounded, 6 metrics)
############################################

def _linmap(x, lo, hi, higher_is_better=True) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if higher_is_better:
        if v <= lo: return 0.0
        if v >= hi: return 100.0
        return (v - lo) / (hi - lo) * 100.0
    else:
        # for lower-better: map [hi (good) .. lo (bad)] to [100..0]
        if v <= hi: return 100.0
        if v >= lo: return 0.0
        return (lo - v) / (lo - hi) * 100.0

def score_nqi_lite_row(row: Dict[str,Any], total_tokens: int) -> Dict[str,float]:
    # 1) FRE: 30→0, 60→100
    s_fre  = _linmap(row.get("flesch_reading_ease",0.0), 30.0, 60.0, True)
    # 2) Entity overlap mean: 0.05→0, 0.35→100
    s_ent  = _linmap(row.get("ent_overlap_mean",0.0), 0.05, 0.35, True)
    # 3) Content overlap mean: 0.02→0, 0.15→100
    s_cont = _linmap(row.get("content_overlap_mean",0.0), 0.02, 0.15, True)
    # 4) Distinct-2: 0.20→0, 0.50→100
    s_d2   = _linmap(row.get("distinct_2",0.0), 0.20, 0.50, True)
    # 5) Trigram repetition (lower better): 0.10→0, 0.02→100
    s_tri  = _linmap(row.get("trigram_rep_rate",0.0), 0.10, 0.02, False)
    # 6) Temporal order violations per 1k tokens: 2.0→0, 0.0→100
    tok = max(1, int(total_tokens))
    vio_k = float(row.get("temporal_order_violations",0.0))/tok*1000.0
    s_tmp  = _linmap(vio_k, 2.0, 0.0, False)

    subs = {
        "nqi_lite_fre": round(s_fre,1),
        "nqi_lite_entity": round(s_ent,1),
        "nqi_lite_content": round(s_cont,1),
        "nqi_lite_distinct2": round(s_d2,1),
        "nqi_lite_trigram": round(s_tri,1),
        "nqi_lite_temporal": round(s_tmp,1),
    }
    subs["NQI_LITE"] = round(sum(subs.values())/6.0, 1)
    return subs

def add_nqi_lite_to_rows(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    for r in rows:
        tokens = len(tokenize(r.get("text","")))
        r.update(score_nqi_lite_row(r, tokens))
    return rows

############################################
# Orchestration
############################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True, help="Input narrative (.txt/.md)")
    ap.add_argument("-o","--outdir", default="narr_eval_out", help="Output directory")
    ap.add_argument("--beats", choices=["auto","none","equal"], default="auto", help="Beat parsing mode")
    ap.add_argument("--equal-k", type=int, default=4, help="Number of equal bins when beats=equal")
    ap.add_argument("--neardupe-th", type=float, default=0.85, help="Char-3 Jaccard threshold for near-duplicate sentences")
    ap.add_argument("--domain-stopwords", nargs="*", default=[], help="Extra stopwords for domain terms")
    ap.add_argument("--with-coherence", action="store_true", help="Also compute topic variance & beat-to-global similarity")
    ap.add_argument("--annotate", action="store_true", help="Emit flags, suggestions, and NQI per beat")
    ap.add_argument("--nqi-lite", action="store_true", help="Compute NQI_LITE (6-metric, absolute-banded)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    text = read_text_file(args.input)
    extra_stop = set(t.lower() for t in args.domain_stopwords)

    if args.beats == "auto":
        beats = parse_beats_auto(text)
    elif args.beats == "equal":
        beats = split_equal_beats(text, args.equal_k)
    else:
        beats = [{"id":"B1","title":"Beat 1","text":(text or "").strip()}]

    # per-beat metrics
    per_rows = []
    for b in beats:
        m = compute_metrics_for_text(b["text"], extra_stop, args.neardupe_th)
        m.update({"beat_id": b["id"], "beat_title": b["title"], "text": b["text"]})
        per_rows.append(m)

    # NQI-Lite (absolute) and/or Annotation+NQI (relative)
    if args.nqi_lite:
        per_rows = add_nqi_lite_to_rows(per_rows)
    if args.annotate:
        per_rows = annotate_perbeat_rows(per_rows)

    # structure metrics
    struct = compute_structure_metrics(beats)

    # overall (full text)
    overall = compute_metrics_for_text(" ".join(b["text"] for b in beats), extra_stop, args.neardupe_th)
    overall.update(struct)
    if args.with_coherence:
        overall.update(optional_coherence(beats, extra_stop))

    # write outputs
    write_csv(os.path.join(args.outdir, "narrative_per_section.csv"), per_rows)
    write_csv(os.path.join(args.outdir, "narrative_summary.csv"), [overall])
    write_report_md(os.path.join(args.outdir, "narrative_report.md"), overall, per_rows)
    if args.annotate:
        # annotated CSV + short MD
        write_csv(os.path.join(args.outdir, "narrative_per_section_annotated.csv"), per_rows)
        md_path = os.path.join(args.outdir, "narrative_report_annotated.md")
        lines = ["# Narrative Quality – Annotated per beat", ""]
        for r in per_rows:
            lines.append(f"## {r.get('beat_id','')} — {r.get('beat_title','')}")
            if "NQI" in r: lines.append(f"- **NQI:** {r['NQI']}/100")
            if "NQI_LITE" in r: lines.append(f"- **NQI_LITE:** {r['NQI_LITE']}/100")
            for cat in ("redundancy_flag","cohesion_flag","readability_flag","consistency_flag"):
                if cat in r:
                    # Show flags compactly
                    pass
            if all(k in r for k in ("redundancy_flag","cohesion_flag","readability_flag","consistency_flag")):
                lines.append(f"- Flags → Redundancy: **{r['redundancy_flag']}**, Cohesion: **{r['cohesion_flag']}**, Readability: **{r['readability_flag']}**, Consistency: **{r['consistency_flag']}**")
            if r.get("suggestions"):
                lines.append(f"- Suggestions: {r['suggestions']}")
            lines.append("")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    main()
