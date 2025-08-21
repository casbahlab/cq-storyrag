from __future__ import annotations
from pathlib import Path
import json, re, statistics
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

# ----------------------------- Utilities -----------------------------

_SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+|\n+')
_WS_RE = re.compile(r'\s+')

STOPWORDS = set("""
a an the and or for of in on at to from by with as is are was were be been being
that this those these it its their his her they them we you i our your not but so
because therefore however though although while when where which who whom whose
into over under without within if then also just very really more most much
""".split())

CAPITAL_STOP = set("""
I We You He She They It Live Aid Wembley Stadium Queen The Who U2
""".split())

BRIDGE_WORDS = [
    "because","so","therefore","as a result","thus",
    "while","although","however","meanwhile","later","earlier","before","after"
]

def sent_tokenize(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text or '') if p.strip()]
    return parts

def normalize(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip()).lower()

def tokenize_content(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text or "")
    return [t.lower() for t in toks if t.lower() not in STOPWORDS]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

# -------------------------- Evidence parsing -------------------------

def parse_facts(evidence):
    """Accepts:
       - {"type":"fact"|"triple","value":"S | P | O"}
       - {"subject":..., "predicate":..., "object":...}
       - {"type":"text","value": "... A → P → B ..."}
    """
    facts = []
    for e in (evidence or []):
        # canonical pipe-separated
        val = e.get("value")
        typ = (e.get("type") or "").lower()
        if isinstance(val, str) and "|" in val and typ in ("fact","triple"):
            parts = [p.strip() for p in val.split("|")]
            if len(parts) == 3:
                facts.append((parts[0], parts[1], parts[2]))
                continue
        # fielded triples
        t = _triple_from_dict(e)
        if t:
            facts.append(t)
            continue
        # arrow-blocks inside text evidence
        if typ == "text" and isinstance(val, str):
            facts.extend(parse_arrow_block(val))
    # humanize S/O for surface matching
    return [(humanize_label(s), p or "", humanize_label(o)) for (s,p,o) in facts]

# --- key-agnostic text harvest + entity extraction ---

import re

def harvest_texts_anywhere(obj) -> list[str]:
    """Collect all human-readable strings inside nested dict/list structures."""
    out = []
    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x: walk(v)
    walk(obj)

    # normalize + dedupe
    norm = lambda s: re.sub(r"\s+", " ", s.strip())
    keep, seen = [], set()
    for t in out:
        n = norm(t)
        if n and n not in seen:
            keep.append(n); seen.add(n)
    return keep

# Dates & quoted titles (generic)
DATE_LONG_RE = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+((?:19|20)\d{2})\b", re.I)
DATE_ISO_RE  = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
QUOTED_RE    = re.compile(r"[“\"]([^”\"]+)[”\"]")

def extract_entities_from_texts(texts: list[str]) -> list[str]:
    """Entities as proper nouns + dates + quoted titles from arbitrary text blocks."""
    ents = []
    for t in texts:
        ents.extend(extract_capitalized_names(t))  # you already have this
        for m in DATE_LONG_RE.finditer(t):
            ents.append(f"{m.group(1)} {m.group(2)} {m.group(3)}")
        ents.extend(DATE_ISO_RE.findall(t))
        for m in QUOTED_RE.finditer(t):
            ents.append(m.group(1).strip())
    # de-dup case-insensitively
    seen, out = set(), []
    for e in ents:
        k = e.lower().strip()
        if k and k not in seen:
            out.append(e); seen.add(k)
    return out

def sentence_supported_comention(sentence: str, evidence_entities: list[str]) -> bool:
    """Schema-free support: a sentence is 'supported' if it co-mentions ≥2 evidence entities."""
    s = sentence.lower()
    hits = [e for e in evidence_entities if e and e.lower() in s]
    return len(set(hits)) >= 2


def extract_facts_and_entities_item(item: dict) -> tuple[list[tuple[str,str,str]], list[str]]:
    """
    1) Use your current extract_facts_from_plan_item(item) to get structured facts if available.
    2) Also hoover *all* text from the item, parse graph-arrow lines into facts,
       and extract generic entities from text as a fallback.
    """
    facts = []
    try:
        facts = extract_facts_from_plan_item(item)  # you already have this
    except Exception:
        facts = []

    # Harvest arbitrary text blocks (KG rows, Hybrid text, Graph neighborhood)
    texts = harvest_texts_anywhere(item)

    # Extra facts from arrow-like graph lines (A → P → B)
    extra_facts = []
    for t in texts:
        if "→" in t or "->" in t or "Graph neighborhood" in t:
            extra_facts.extend(parse_arrow_block(t))  # you already have this

    # Merge facts + extra_facts (dedupe)
    seen, merged = set(), []
    for (s,p,o) in (facts + extra_facts):
        key = (s.lower(), (p or "").lower(), o.lower())
        if key not in seen:
            seen.add(key); merged.append((s, p or "", o))

    # Entities from facts if we have any; otherwise from text
    ents = evidence_entities(merged) if merged else extract_entities_from_texts(texts)
    return merged, ents


import re

EM_DASH = " — "

def _is_private_key(k: str) -> bool:
    return isinstance(k, str) and k.startswith("__")

def strip_quotes(s: str) -> str:
    if not isinstance(s, str): return s
    s2 = s.strip()
    if len(s2) >= 2 and ((s2[0], s2[-1]) in {('"','"'), ("'","'")}):
        return s2[1:-1]
    return s2

def content_tokens_plain(text: str) -> list[str]:
    # (reuse your STOPWORDS if already defined)
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text or "")
    return [t.lower() for t in toks if t.lower() not in STOPWORDS]

# Graph neighborhood parser: lines like “X → rel → Y” or “… -> …”
_ARROW_RE = re.compile(r'^\s*[-•]?\s*(.+?)\s*(?:→|->)\s*(.+?)\s*(?:→|->)\s*(.+?)\s*$')

def parse_arrow_block_to_factlets(s: str) -> list[str]:
    outs = []
    for line in (s or "").splitlines():
        m = _ARROW_RE.match(line.strip())
        if not m: continue
        a, p, b = [strip_quotes(t.strip()) for t in m.groups()]
        if a and b:
            outs.append(f"{a}{EM_DASH}{p}{EM_DASH}{b}")
    return outs

# in _row_to_factlet(...)
def _row_to_factlet(row, include_urls=False, max_len=280):
    parts = []
    for k, v in row.items():
        if k.startswith("__"):
            continue
        if isinstance(v, str):
            if not include_urls and v.strip().lower().startswith(("http://","https://","<http")):
                continue
            parts.append(strip_quotes(v))
        elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            parts.append(strip_quotes(" / ".join(v)))
    if not parts:
        return None
    factlet = " — ".join(p for p in parts if p)
    return factlet[:max_len].rsplit(" ", 1)[0] + "…" if len(factlet) > max_len else factlet


def build_factlets_from_item(item: dict) -> list[str]:
    factlets = []
    # 1) Rows → factlets (KG/Hybrid path)
    rows = item.get("rows") or []
    for r in rows:
        fl = _row_to_factlet(r, include_urls=True)
        if fl: factlets.append(fl)

    # 2) Graph evidence “text” blocks → arrow factlets
    ev = item.get("evidence") or []
    for e in ev:
        if (e.get("type") or "").lower() == "text" and isinstance(e.get("value"), str):
            # only convert if it looks like a neighborhood block
            if "→" in e["value"] or "->" in e["value"] or "Graph neighborhood" in e["value"]:
                factlets.extend(parse_arrow_block_to_factlets(e["value"]))

    # 3) De-dup case-insensitive
    seen = set(); uniq = []
    for fl in factlets:
        k = fl.lower().strip()
        if k and k not in seen:
            seen.add(k); uniq.append(fl)
    return uniq

def sentence_supported_by_factlets(sentence: str, factlets: list[str], jaccard_min: float = 0.22, min_hits: int = 2) -> bool:
    """A sentence is supported if it overlaps meaningfully with any factlet."""
    s_tokens = set(content_tokens_plain(sentence))
    if not s_tokens:
        return False
    for fl in factlets:
        f_tokens = set(content_tokens_plain(fl))
        inter = len(s_tokens & f_tokens)
        if inter >= min_hits:
            # optional: jaccard to avoid spurious hits on short lines
            union = len(s_tokens | f_tokens)
            if union == 0 or (inter / union) >= jaccard_min:
                return True
    return False

def factlet_coverage_ratio(section_text: str, factlets: list[str], min_hits: int = 2) -> float:
    """Fraction of factlets that are touched by the section text."""
    sec_tokens = set(content_tokens_plain(section_text))
    if not factlets:
        return 0.0
    touched = 0
    for fl in factlets:
        f_tokens = set(content_tokens_plain(fl))
        if len(sec_tokens & f_tokens) >= min_hits:
            touched += 1
    return touched / len(factlets)



def extract_facts_from_plan_item(item: dict) -> list[tuple[str,str,str]]:
    # 1) Preferred: evidence[] (KG/Graph)

    ev = item.get("evidence")
    facts = parse_facts(ev) if ev else []
    if facts:
        return facts

    # 2) Harvest any triple-like structures anywhere in the item (KG variants)
    harvested = harvest_triples_anywhere(item)
    if harvested:
        return [(humanize_label(s), p or "", humanize_label(o)) for (s,p,o) in harvested]

    # 3) Hybrid/CQ fallback: synthesize minimal triples from rows[]
    rows = item.get("rows") or []
    if rows:
        subj = strip_quotes(rows[0].get("eventName") or "") \
               or strip_quotes(item.get("beat",{}).get("title") or "") \
               or "Topic"
        derived = []
        for r in rows:
            text = r.get("text") or ""
            for name in extract_capitalized_names(text):
                if name and name.lower() != subj.lower():
                    derived.append((subj, "mentions", name))
            url = r.get("url") or ""
            if url:
                derived.append((subj, "cites", url))
        if derived:
            return [(humanize_label(s), p, humanize_label(o)) for (s,p,o) in derived]

    return []



def evidence_entities(facts: List[Tuple[str,str,str]]) -> List[str]:
    ents = []
    for s, _, o in facts:
        ents.append(s)
        ents.append(o)
    seen, out = set(), []
    for e in ents:
        key = normalize(e)
        if key not in seen:
            out.append(e)
            seen.add(key)
    return out

def top_entities_by_degree(facts: List[Tuple[str,str,str]], topn: int = 10) -> List[str]:
    from collections import Counter
    deg = Counter()
    for s, _, o in facts:
        deg[normalize(s)] += 1
        deg[normalize(o)] += 1
    ranked = [e for e,_ in deg.most_common(topn)]
    return ranked

# ------------------------- Sentence scoring --------------------------

def sentence_supported(sentence: str, facts: List[Tuple[str,str,str]]) -> bool:
    s_norm = normalize(sentence)
    for s,p,o in facts:
        s_ok = normalize(s) in s_norm
        o_ok = normalize(o) in s_norm
        if s_ok and o_ok:
            return True
    return False

NOISE_NAMES = {
    "This","That","These","Those","Many","One","As","On","At","In","Of","For","And",
    "During","Across","Between","Within","By","From","Into","Over","Under",
    "January","February","March","April","May","June","July","August","September",
    "October","November","December","London","Philadelphia"
}

def extract_capitalized_names(sentence: str) -> list[str]:
    raw = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", sentence or "")
    out, seen = [], set()
    for r in raw:
        r2 = r.strip()
        if r2 in CAPITAL_STOP or r2 in NOISE_NAMES:
            continue
        if r2 not in seen:
            out.append(r2); seen.add(r2)
    return out

# -------------------------- Cohesiveness -----------------------------

def content_tokens_plain(text: str) -> List[str]:
    stop = set(STOPWORDS)
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", text or "")
    return [t.lower() for t in toks if t.lower() not in stop]

def jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(content_tokens_plain(a)), set(content_tokens_plain(b))
    if not sa and not sb: return 1.0
    u = sa | sb
    return (len(sa & sb) / len(u)) if u else 0.0

def local_coherence_band(sentences: List[str]) -> float:
    pairs = [(sentences[i-1], sentences[i]) for i in range(1, len(sentences))]
    good = 0; total = 0
    for a,b in pairs:
        s = jaccard_tokens(a,b)
        total += 1
        if 0.15 <= s <= 0.65: good += 1
    return (good / total) if total else 1.0

def bridge_rate(sentences: List[str]) -> float:
    hits = 0
    for s in sentences:
        ls = " " + s.lower() + " "
        if any((" " + w + " ") in ls for w in BRIDGE_WORDS):
            hits += 1
    raw = hits / max(1, len(sentences))
    clamped = min(0.6, max(0.2, raw))
    return (clamped - 0.2) / 0.4  # 0..1

def extract_years(sentence: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", sentence)]

def temporal_consistency(sentences: List[str]) -> float:
    cues = ("earlier","before","previously","flashback")
    inversions = 0; dated_pairs = 0
    prev_years = extract_years(sentences[0]) if sentences else []
    for i in range(1, len(sentences)):
        cur_years = extract_years(sentences[i])
        if prev_years and cur_years:
            dated_pairs += 1
            if max(cur_years) < min(prev_years):
                if not any(c in sentences[i].lower() for c in cues):
                    inversions += 1
        if cur_years: prev_years = cur_years
    if dated_pairs == 0: return 1.0
    return 1 - inversions / dated_pairs

def reference_stability(sentences: List[str], evidence_entities: List[str]) -> float:
    import re
    norm = lambda s: re.sub(r"[^A-Za-z0-9 ]+", "", s).strip().lower()
    forms = {norm(e): set() for e in evidence_entities}
    text = " ".join(sentences)
    for e in evidence_entities:
        n = norm(e)
        if not n: continue
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
            if n in norm(m.group(0)):
                forms[n].add(m.group(0))
    extras = []
    for k, vs in forms.items():
        if not vs: continue
        extra = max(0, len(vs) - 2)
        extras.append(extra)
    if not extras: return 1.0
    avg_extra = sum(extras) / len(extras)
    return max(0.0, 1.0 - (avg_extra * 0.25))

def compute_cohesiveness(section_text: str, evidence_facts: List[Tuple[str,str,str]], evidence_entities: List[str]) -> Dict[str, float]:
    sents = [s for s in _SENT_SPLIT_RE.split(section_text or '') if s.strip()]
    if not sents:
        return {
            "cohesiveness": 0.0, "local_band": 0.0, "entity_flow": 0.0,
            "bridge_rate": 0.0, "temporal_consistency": 1.0, "ref_stability": 1.0
        }
    def has_shared_entity(a: str, b: str) -> bool:
        A = set(e.lower() for e in evidence_entities if e and e.lower() in a.lower())
        B = set(e.lower() for e in evidence_entities if e and e.lower() in b.lower())
        return len(A & B) > 0
    pairs = [(sents[i-1], sents[i]) for i in range(1, len(sents))]
    share = sum(1 for a,b in pairs if has_shared_entity(a,b)) / max(1, len(pairs))
    def shared_count(a: str, b: str) -> int:
        A = set(e.lower() for e in evidence_entities if e and e.lower() in a.lower())
        B = set(e.lower() for e in evidence_entities if e and e.lower() in b.lower())
        return min(2, len(A & B))
    avg_shared = (sum(shared_count(a,b) for a,b in pairs) / max(1, len(pairs))) / 2.0
    entity_flow = 0.7*share + 0.3*avg_shared

    local_band = local_coherence_band(sents)
    bridges = bridge_rate(sents)
    time_ok = temporal_consistency(sents)
    ref_ok = reference_stability(sents, evidence_entities)

    cohesiveness = (
        0.35*local_band + 0.25*entity_flow + 0.15*bridges +
        0.15*time_ok + 0.10*ref_ok
    )
    return {
        "cohesiveness": round(cohesiveness,3),
        "local_band": round(local_band,3),
        "entity_flow": round(entity_flow,3),
        "bridge_rate": round(bridges,3),
        "temporal_consistency": round(time_ok,3),
        "ref_stability": round(ref_ok,3),
    }

# ---------------------- Narrative Quality (story) --------------------

_VOWEL_RE = re.compile(r"[aeiouy]+", re.I)

def _sentences(s: str) -> List[str]:
    return [x.strip() for x in _SENT_SPLIT_RE.split(s or "") if x.strip()]

def _tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", s)

def _content_tokens(s: str) -> List[str]:
    return [t.lower() for t in _tokens(s) if t.lower() not in STOPWORDS]

def _count_syllables(word: str) -> int:
    w = word.lower()
    if len(w) <= 3: return 1
    w2 = re.sub(r"e\b", "", w)
    chunks = _VOWEL_RE.findall(w2)
    return max(1, len(chunks))

def flesch_reading_ease(text: str) -> float:
    sents = _sentences(text); words = _tokens(text)
    if not sents or not words: return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    W = len(words); S = len(sents)
    return 206.835 - 1.015*(W/S) - 84.6*(syllables/W)

def fk_grade_level(text: str) -> float:
    sents = _sentences(text); words = _tokens(text)
    if not sents or not words: return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    W = len(words); S = len(sents)
    return 0.39*(W/S) + 11.8*(syllables/W) - 15.59

from pathlib import Path

def _force_clean_story(cfg: dict) -> dict:
    # Derive the directory from the answers file we *know* is correct
    ans_path = Path(cfg["answers"])
    run_dir = ans_path.parent
    pat = cfg.get("name") or re.search(r"answers_(\w+)\.jsonl$", ans_path.name, re.I).group(1)
    clean = run_dir / f"story_{pat}_clean.md"
    plain = run_dir / f"story_{pat}.md"

    if clean.exists():
        cfg["story"] = str(clean)  # force clean
    elif plain.exists():
        cfg["story"] = str(plain)  # fallback
    else:
        # leave as-is (maybe upstream passed a different path)
        pass
    return cfg

def narrative_quality(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    all_sents = _sentences(text)
    has_bullets = any(re.match(r"^\s*[-\*\d]+\s", line) for line in (text or "").splitlines())
    has_headings = any(re.match(r"^\s*#{1,6}\s", line) for line in (text or "").splitlines())
    has_uris = bool(re.search(r"https?://", text or ""))
    has_instance_ids = bool(re.search(r"\b(Membership_|lit::|http://wembrewind\.live/ex#)", text or ""))

    sent_per_para = [len(_sentences(p)) for p in paras]
    single_sent_para_ratio = sum(1 for n in sent_per_para if n == 1) / max(1, len(paras))

    fre = flesch_reading_ease(text); fkgl = fk_grade_level(text)
    avg_sent_len = (sum(len(_content_tokens(s)) for s in all_sents) / max(1, len(all_sents))) if all_sents else 0.0

    all_tokens = [t.lower() for t in _tokens(text)]
    ttr = (len(set(all_tokens)) / max(1, len(all_tokens)))
    content_all = [t for t in _content_tokens(text)]
    content_ttr = (len(set(content_all)) / max(1, len(content_all)))

    bigrams = [" ".join(content_all[i:i+2]) for i in range(len(content_all)-1)]
    trigrams = [" ".join(content_all[i:i+3]) for i in range(len(content_all)-2)]
    from collections import Counter
    def repetition_rate(ngrams):
        c = Counter(ngrams)
        repeats = sum(1 for k,v in c.items() if v>=2)
        return repeats / max(1, len(c))
    bigram_rep = repetition_rate(bigrams); trigram_rep = repetition_rate(trigrams)

    passive_hits = len(re.findall(r"\b(?:was|were|is|are|been|being|be)\s+\w+ed\b", text or "", flags=re.I))
    passive_rate = passive_hits / max(1, len(all_sents))

    BRIDGE = BRIDGE_WORDS
    bridge_sents = sum(1 for s in all_sents if any((" "+w+" ") in (" "+s.lower()+" ") for w in BRIDGE))
    bridge_rate = bridge_sents / max(1, len(all_sents))

    adj_overlap = []
    for i in range(1, len(all_sents)):
        a = _content_tokens(all_sents[i-1]); b = _content_tokens(all_sents[i])
        sa, sb = set(a), set(b); u = sa | sb
        adj_overlap.append((len(sa & sb) / len(u)) if u else 0.0)
    redundancy_rate = sum(1 for r in adj_overlap if r >= 0.8) / max(1, len(adj_overlap))

    def extract_years(s):
        return [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", s)]
    inversions = 0; dated_pairs = 0
    prev_years = extract_years(all_sents[0]) if all_sents else []
    for i in range(1, len(all_sents)):
        cur_years = extract_years(all_sents[i])
        if prev_years and cur_years:
            dated_pairs += 1
            if max(cur_years) < min(prev_years) and not any(w in all_sents[i].lower() for w in ("earlier","before","previously","flashback")):
                inversions += 1
        if cur_years: prev_years = cur_years
    temporal_consistency = 1 - inversions / max(1, dated_pairs) if dated_pairs>0 else 1.0

    thresholds = {
        "no_bullets": True,
        "no_headings": True,
        "no_uris": True,
        "no_instance_ids": True,
        "fre_min": 40.0,
        "fkgl_max": 16.0,
        "avg_sent_len_min": 18,
        "avg_sent_len_max": 35,
        "single_sent_para_ratio_max": 0.60,
        "bigram_rep_max": 0.25,
        "trigram_rep_max": 0.15,
        "passive_rate_max": 0.35,
        "bridge_rate_min": 0.15,
        "bridge_rate_max": 0.60,
        "redundancy_rate_max": 0.20,
        "temporal_consistency_min": 0.70,
        "content_ttr_min": 0.35,
    }

    checks = {
        "no_bullets": not has_bullets,
        "no_headings": not has_headings,
        "no_uris": not has_uris,
        "no_instance_ids": not has_instance_ids,
        "fre_min": fre >= thresholds["fre_min"],
        "fkgl_max": fkgl <= thresholds["fkgl_max"],
        "avg_sent_len_band": thresholds["avg_sent_len_min"] <= avg_sent_len <= thresholds["avg_sent_len_max"],
        "single_sent_para_ratio_max": single_sent_para_ratio <= thresholds["single_sent_para_ratio_max"],
        "bigram_rep_max": bigram_rep <= thresholds["bigram_rep_max"],
        "trigram_rep_max": trigram_rep <= thresholds["trigram_rep_max"],
        "passive_rate_max": passive_rate <= thresholds["passive_rate_max"],
        "bridge_rate_band": thresholds["bridge_rate_min"] <= bridge_rate <= thresholds["bridge_rate_max"],
        "redundancy_rate_max": redundancy_rate <= thresholds["redundancy_rate_max"],
        "temporal_consistency_min": temporal_consistency >= thresholds["temporal_consistency_min"],
        "content_ttr_min": content_ttr >= thresholds["content_ttr_min"],
    }

    summary = {
        "paragraphs": len(paras),
        "sentences": len(all_sents),
        "avg_sent_len_tokens": round(avg_sent_len,2),
        "Flesch_RE": round(fre,2),
        "FK_Grade": round(fkgl,2),
        "type_token_ratio": round(ttr,3),
        "content_TTR": round(content_ttr,3),
        "bigram_repetition_rate": round(bigram_rep,3),
        "trigram_repetition_rate": round(trigram_rep,3),
        "passive_rate": round(passive_rate,3),
        "bridge_rate": round(bridge_rate,3),
        "redundancy_rate_adjacent": round(redundancy_rate,3),
        "temporal_consistency": round(temporal_consistency,3),
        "single_sentence_para_ratio": round(single_sent_para_ratio,3),
        "bullets_found": has_bullets,
        "headings_found": has_headings,
        "uris_found": has_uris,
        "instance_ids_found": has_instance_ids,
    }

    return summary, {"checks": checks, "thresholds": thresholds}

# ------------------------------ Main eval ----------------------------

def evaluate_run(answers_jsonl: Path, plan_json: Path, pattern_name: str = "Run", story_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # Load answers
    recs = []
    with open(answers_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass

    plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    items = plan.get("items", [])

    rows = []
    for rec in recs:
        idx = rec.get("beat_index", 0)
        text = rec.get("text", "")
        title = rec.get("beat_title") or (items[idx]["beat"] if idx < len(items) else f"Section {idx+1}")
        item = items[idx] if idx < len(items) else {}

        # A) Build factlets (KG/Hybrid exact prompt facts; Graph arrow lines)
        factlets = build_factlets_from_item(item)
        factlets_count = len(factlets)

        # B) Keep your existing triples (for when they exist)
        facts = extract_facts_from_plan_item(item) if 'extract_facts_from_plan_item' in globals() else []
        facts_count = len(facts)

        # C) Choose evidence mode: prefer factlets if available
        text = rec.get("text", "")
        sents = sent_tokenize(text)

        if factlets_count > 0:
            supported_flags = [sentence_supported_by_factlets(s, factlets) for s in sents]
            coverage_ratio = factlet_coverage_ratio(text, factlets)
            evidence_mode = "factlets"
        elif facts_count > 0:
            supported_flags = [sentence_supported(s, facts) for s in sents]  # your existing strict S/O co-mention
            # your existing coverage from facts/entities:
            section_text_norm = normalize(text)
            from collections import Counter
            deg = Counter()
            for s_, _, o_ in facts:
                deg[normalize(s_)] += 1;
                deg[normalize(o_)] += 1
            top_ents = [e for e, _ in deg.most_common(min(10, len(deg)))]
            coverage_ratio = sum(1 for te in top_ents if te in section_text_norm) / max(1, len(top_ents))
            evidence_mode = "triples"
        else:
            # final fallback: co-mention using entities mined from the section itself
            ents = extract_capitalized_names(text)

            def _supported_comention(s):
                hits = [e for e in ents if e.lower() in s.lower()]
                return len(set(hits)) >= 2

            supported_flags = [_supported_comention(s) for s in sents]
            coverage_ratio = 0.0
            evidence_mode = "fallback"

        # --- redundancy (adjacent sentence overlap) ---
        toks = [tokenize_content(s) for s in sents]
        if len(toks) >= 2:
            overlaps = [jaccard(toks[i - 1], toks[i]) for i in range(1, len(toks))]
            redundancy_rate = sum(1 for r in overlaps if r >= 0.8) / max(1, len(overlaps))
        else:
            redundancy_rate = 0.0

        coh = compute_cohesiveness(text, facts, evidence_entities(facts) if facts_count > 0 else sents)

        rows.append({
            "pattern": pattern_name,
            "section_index": idx,
            "section_title": title,
            "sentences": len(sents),
            "supported_sentences": sum(1 for b in supported_flags if b),
            "support_ratio": (sum(1 for b in supported_flags if b) / max(1, len(sents))),
            "entity_coverage_ratio": coverage_ratio,
            "redundancy_rate_adjacent": redundancy_rate,
            "cohesiveness": coh["cohesiveness"],
            "coh_local_band": coh["local_band"],
            "coh_entity_flow": coh["entity_flow"],
            "coh_bridge_rate": coh["bridge_rate"],
            "coh_temporal": coh["temporal_consistency"],
            "coh_ref_stability": coh["ref_stability"],
            "evidence_mode": evidence_mode,  # NEW: visible in per_section.csv
            "factlets_count": factlets_count,  # NEW
            "facts_count": facts_count,  # keep existing
        })

    per_section = pd.DataFrame(rows)
    NUM_COLS = [
        "sentences", "supported_sentences", "support_ratio",
        "entity_coverage_ratio", "redundancy_rate_adjacent",
        "cohesiveness", "coh_local_band", "coh_entity_flow",
        "coh_bridge_rate", "coh_temporal", "coh_ref_stability",
        "facts_count", "factlets_count"
    ]
    for c in [c for c in NUM_COLS if c in per_section.columns]:
        per_section[c] = pd.to_numeric(per_section[c], errors="coerce")

    # overall summary
    if not per_section.empty:
        summary = per_section.groupby("pattern").agg(
            sections=("section_index","count"),
            avg_support_ratio=("support_ratio","mean"),
            median_support_ratio=("support_ratio","median"),
            avg_coverage=("entity_coverage_ratio","mean"),
            avg_redundancy=("redundancy_rate_adjacent","mean"),
            avg_sentences=("sentences","mean"),
            avg_cohesiveness=("cohesiveness","mean"),
        ).reset_index()
    else:
        summary = pd.DataFrame([{
            "pattern": pattern_name,
            "sections": 0,
            "avg_support_ratio": 0.0,
            "median_support_ratio": 0.0,
            "avg_coverage": 0.0,
            "avg_redundancy": 0.0,
            "avg_sentences": 0.0,
            "avg_cohesiveness": 0.0,
        }])

    # Narrative-quality (final clean story) if provided
    story_report = None
    if story_path and Path(story_path).exists():
        text = Path(story_path).read_text(encoding="utf-8")
        nq_summary, nq_checks = narrative_quality(text)
        story_report = {
            "story_file": str(story_path),
            "summary": nq_summary,
            "checks": nq_checks["checks"],
            "thresholds": nq_checks["thresholds"],
        }

    return per_section, summary, story_report

import re

def humanize_label(x: str) -> str:
    """Strip URI namespaces, split CamelCase/underscores, tidy casing."""
    if not x: return x
    x = re.sub(r'^https?://[^#/]+[/#]', '', x)   # drop namespace
    x = re.sub(r'[_\-]+', ' ', x)                # underscores/dashes → spaces
    x = re.sub(r'(?<=.)([A-Z])', r' \1', x)      # split CamelCase
    x = re.sub(r'\s+', ' ', x).strip()
    return " ".join(w if w.isupper() else w.capitalize() for w in x.split())

_ARROW_RE = re.compile(r'^\s*[-•]?\s*(.+?)\s*(?:→|->)\s*(.+?)\s*(?:→|->)\s*(.+?)\s*$')

def parse_arrow_block(s: str):
    facts = []
    for line in (s or "").splitlines():
        m = _ARROW_RE.match(line)
        if m:
            a, p, b = [t.strip(" \t\"'") for t in m.groups()]
            if a and b:
                facts.append((a, p, b))
    return facts


def strip_quotes(s: str) -> str:
    if not s: return s
    s = s.strip()
    return s[1:-1] if len(s) >= 2 and ((s[0], s[-1]) in {('"','"'), ("'","'")}) else s

def _triple_from_dict(d: dict):
    # many possible fieldings seen "in the wild"
    s = d.get("subject") or d.get("s") or d.get("head") or (d.get("triple") or {}).get("s")
    p = d.get("predicate") or d.get("p") or d.get("rel")  or (d.get("triple") or {}).get("p")
    o = d.get("object")  or d.get("o") or d.get("tail") or (d.get("triple") or {}).get("o")
    if s and o:
        return (s, p or "", o)
    return None

def harvest_triples_anywhere(obj) -> list[tuple[str,str,str]]:
    """Recursively walk obj and collect any dicts that look like triples."""
    out = []
    if isinstance(obj, dict):
        # direct triple dict?
        t = _triple_from_dict(obj)
        if t: out.append(t)
        # pipe-separated?
        val = obj.get("value")
        typ = (obj.get("type") or "").lower()
        if isinstance(val, str):
            if "|" in val and typ in ("fact","triple"):   # "S | P | O"
                parts = [p.strip() for p in val.split("|")]
                if len(parts) == 3:
                    out.append((parts[0], parts[1], parts[2]))
            elif typ == "text":                            # arrow lines in text evidence
                out.extend(parse_arrow_block(val))
        # common containers: "evidence", "triples", "edges", "neighbors", "rows"
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out.extend(harvest_triples_anywhere(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(harvest_triples_anywhere(x))
    return out



def evaluate_many(configs: List[Dict[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    dfs, sums, reports = [], [], []
    for cfg in configs:
        cfg = _force_clean_story(cfg)
        per_sec, summ, story_rep = evaluate_run(
            Path(cfg["answers"]),
            Path(cfg["plan"]),
            cfg.get("name","Run"),
            Path(cfg["story"]) if cfg.get("story") else None
        )
        dfs.append(per_sec); sums.append(summ)
        reports.append({
            "name": cfg.get("name","Run"),
            "answers": cfg["answers"],
            "plan": cfg["plan"],
            "story": cfg.get("story"),
            "per_section": per_sec.to_dict(orient="records"),
            "summary": summ.to_dict(orient="records")[0] if not summ.empty else {},
            "story_quality": story_rep,
        })
    all_sections = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    all_summary = pd.concat(sums, ignore_index=True) if sums else pd.DataFrame()
    return all_sections, all_summary, {"runs": reports}
