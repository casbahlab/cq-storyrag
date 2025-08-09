# evaluation/enhanced_eval.py
# Python 3.9 compatible

import json
import re
from typing import Dict, Any, List, Tuple, Optional

# Optional readability metrics
try:
    from textstat.textstat import textstat
except Exception:
    textstat = None

TRANSITIONS = [
    "building on this",
    "as the set progressed",
    "following this performance",
    "meanwhile",
    "later in the set",
    "in the end",
    "finally",
    "subsequently",
    "moving into the core",
    "to conclude",
]

# --------------------------
# Helpers
# --------------------------
import difflib
import string
from typing import Iterable

_PUNCT = str.maketrans({c: " " for c in string.punctuation})

def _basic_norm(s: str) -> str:
    # lowercase, ascii quotes, drop punctuation → single spaces
    s = (s or "").replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()

def _tokens(s: str) -> list:
    return [t for t in _basic_norm(s).split() if len(t) >= 3]

def _sequence_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _basic_norm(a), _basic_norm(b)).ratio()

def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(len(tb), 1)  # how much of b’s tokens appear in a

# very small helper: yyyy-mm-dd → "Month d, yyyy"
_MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
def _date_variants(s: str) -> Iterable[str]:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if not m:
        return [s]
    y, mm, dd = m.groups()
    try:
        mon = _MONTHS[int(mm)-1]
        d = str(int(dd))
        return [s, f"{mon} {d}, {y}"]
    except Exception:
        return [s]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _values_from_item(item: Dict[str, Any]) -> List[str]:
    """
    Collect human-readable strings from a facts item:
      - row values (rows is a list of dicts)
      - labels values (labels is a dict of {k: v})
    """
    vals: List[str] = []
    for r in (item.get("rows") or []):
        for v in (r or {}).values():
            if v is None:
                continue
            vs = str(v).strip()
            if vs:
                vals.append(vs)
    for v in (item.get("labels") or {}).values():
        if v is None:
            continue
        vs = str(v).strip()
        if vs:
            vals.append(vs)
    return vals

def _mentioned(narr_norm: str, value: str) -> bool:
    """
    Soft presence check:
      1) direct substring on normalized text
      2) fuzzy sequence ratio ≥ 0.82
      3) token overlap ≥ 0.6 (and at least 2 tokens in value)
      4) date variants are also tried
    """
    if not value:
        return False

    # normalize once
    nn = _basic_norm(narr_norm)  # NOTE: narr_norm may already be normalized; safe to re-norm
    candidates = []
    for v in _date_variants(value):
        v = v.strip()
        if not v:
            continue
        candidates.append(v)

    for cand in candidates:
        vn = _basic_norm(cand)
        if len(vn) < 3:
            continue

        # 1) substring (fast path)
        if vn in nn:
            return True

        # 2) fuzzy ratio
        if _sequence_ratio(nn, vn) >= 0.82:
            return True

        # 3) token overlap (value tokens should largely appear in narrative)
        val_toks = _tokens(cand)
        if len(val_toks) >= 2 and _token_overlap(nn, cand) >= 0.6:
            return True

    return False

def _collect_selected_cqs(plan: Dict[str, Any]) -> Dict[str, List[str]]:
    out = {"Entry": [], "Core": [], "Exit": []}
    exec_plan = (plan or {}).get("execution", {})
    for cat in out.keys():
        for it in exec_plan.get(cat, []) or []:
            cid = it.get("cq_id")
            if cid:
                out[cat].append(cid)
    return out

def _collect_answered_cqs(facts: Dict[str, Any]) -> Dict[str, List[str]]:
    out = {"Entry": [], "Core": [], "Exit": []}
    if not isinstance(facts, dict):
        return out
    for cat in out.keys():
        for it in facts.get(cat, []) or []:
            if it.get("rows"):
                cid = it.get("cq_id")
                if cid:
                    out[cat].append(cid)
    return out

def _row_stats(facts: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for cat in ["Entry", "Core", "Exit"]:
        items = facts.get(cat, []) if isinstance(facts, dict) else []
        counts = [len((it or {}).get("rows", []) or []) for it in items]
        total = sum(counts)
        avg = (float(total) / float(len(counts))) if counts else 0.0
        stats[cat] = {"total_rows": total, "avg_rows_per_cq": round(avg, 2)}
    stats["overall"] = {
        "total_rows": sum(stats[c]["total_rows"] for c in ["Entry", "Core", "Exit"]),
        "avg_rows_per_cq": round(
            (stats["Entry"]["avg_rows_per_cq"] + stats["Core"]["avg_rows_per_cq"] + stats["Exit"]["avg_rows_per_cq"]) / 3.0,
            2,
        ),
    }
    return stats

def _item_label_presence_pct(facts: Dict[str, Any]) -> Dict[str, float]:
    """
    Legacy metric retained for continuity:
    % of items that carry a non-empty labels map (not narrative mentions).
    """
    cov: Dict[str, float] = {}
    for cat in ["Entry", "Core", "Exit"]:
        items = facts.get(cat, []) if isinstance(facts, dict) else []
        with_labels = sum(1 for it in items if (it.get("labels") or {}))
        cov[cat] = round(100.0 * float(with_labels) / max(len(items), 1), 1)
    cov["overall"] = round((cov["Entry"] + cov["Core"] + cov["Exit"]) / 3.0, 1)
    return cov

def _length_target_ok(narrative: str, persona: Dict[str, Any]) -> Dict[str, Any]:
    length = ((persona or {}).get("length", "short") or "").lower()
    target = {"short": (0, 900), "medium": (900, 1600), "long": (1600, 999999)}
    lo, hi = target.get(length, (0, 999999))
    chars = len(narrative or "")
    return {
        "length_setting": length,
        "chars": chars,
        "within_target": bool(lo <= chars <= hi),
        "target_range": [lo, hi],
    }

def _flow_cues(narrative: str) -> Dict[str, Any]:
    n = (narrative or "").lower()
    hits = [t for t in TRANSITIONS if t in n]
    return {"transition_cues_found": hits, "count": len(hits)}

# --------------------------
# Fact presence core
# --------------------------

def _evaluate_fact_presence(narrative: str, facts: Dict[str, Any]) -> Dict[str, Any]:
    nn = _basic_norm(narrative)
    per_cat = {"Entry": {"total": 0, "present": 0}, "Core": {"total": 0, "present": 0}, "Exit": {"total": 0, "present": 0}}
    missing, present_examples = [], []

    for cat in ["Entry", "Core", "Exit"]:
        items = facts.get(cat, []) if isinstance(facts, dict) else []
        for it in items:
            per_cat[cat]["total"] += 1
            vals = _values_from_item(it)
            hit_val = next((v for v in vals if _mentioned(nn, v)), None)
            if hit_val:
                per_cat[cat]["present"] += 1
                present_examples.append({
                    "category": cat,
                    "cq_id": it.get("cq_id"),
                    "question": it.get("question"),
                    "matched_value": hit_val
                })
            else:
                missing.append({
                    "category": cat,
                    "cq_id": it.get("cq_id"),
                    "question": it.get("question"),
                    "sample_values": vals[:2]
                })

    totals = {
        "total": sum(per_cat[c]["total"] for c in per_cat),
        "present": sum(per_cat[c]["present"] for c in per_cat),
    }
    totals["presence_pct"] = round(100.0 * float(totals["present"]) / max(totals["total"], 1), 1)

    return {
        "per_category": {
            c: {
                "present": per_cat[c]["present"],
                "total": per_cat[c]["total"],
                "presence_pct": round(100.0 * float(per_cat[c]["present"]) / max(per_cat[c]["total"], 1), 1),
            } for c in ["Entry", "Core", "Exit"]
        },
        "overall": totals,
        "present_examples": present_examples[:20],
        "missing_facts": missing[:50]
    }


# --------------------------
# Main API
# --------------------------

def evaluate_enhanced(
    narrative_json_path: str,
    kg_ttl_path: Optional[str] = None,            # kept for signature compatibility; unused now
    coverage_expectation: float = 0.75            # 0–1, length-driven threshold
) -> Dict[str, Any]:
    """
    Enhanced evaluation (URI-free):
      - Coverage (selected vs answered CQs) + pass/fail vs expectation
      - Rows stats
      - Label presence on items (legacy metric)
      - Utilization (mentions / available values)
      - Fact presence (core grounding) + pass/fail vs expectation
      - Readability (if textstat available)
      - Length target check
      - Flow cue check
    """
    data = json.loads(open(narrative_json_path, "r", encoding="utf-8").read())
    narrative = data.get("narrative", "") or ""
    plan = data.get("plan", {}) or {}
    facts = data.get("facts", {}) or {}
    persona = plan.get("persona", {}) or {}

    # 1) Coverage by CQ
    selected = _collect_selected_cqs(plan)
    answered = _collect_answered_cqs(facts)
    coverage: Dict[str, Any] = {}
    for cat in ["Entry", "Core", "Exit"]:
        s, a = len(selected[cat]), len(answered[cat])
        pct = round(100.0 * float(a) / max(s, 1), 1)
        coverage[cat] = {"selected": s, "answered": a, "answer_rate_pct": pct}
    total_selected = sum(len(selected[c]) for c in selected)
    total_answered = sum(len(answered[c]) for c in answered)
    overall_pct = round(100.0 * float(total_answered) / max(total_selected, 1), 1)
    coverage["overall"] = {
        "selected": total_selected,
        "answered": total_answered,
        "answer_rate_pct": overall_pct,
    }
    coverage_pass = (overall_pct / 100.0) >= float(coverage_expectation)

    # 2) Retrieval stats
    rows_stats = _row_stats(facts)

    # 3) Label presence on items (legacy)
    label_presence_pct = _item_label_presence_pct(facts)

    # 4) Utilization (mentions / values)
    #    Uses all row+label values as candidates and checks containment.
    def _extract_all_values(facts_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        by_cat = {"Entry": [], "Core": [], "Exit": []}
        if not isinstance(facts_dict, dict):
            return by_cat
        for cat in by_cat.keys():
            for item in facts_dict.get(cat, []) or []:
                by_cat[cat].extend(_values_from_item(item))
        return by_cat

    vals_by_cat = _extract_all_values(facts)
    narr_norm = _basic_norm(narrative)
    util_by_cat: Dict[str, Any] = {}
    total_hits, total_vals = 0, 0
    for cat in ["Entry", "Core", "Exit"]:
        vals = vals_by_cat.get(cat, [])
        t = len([v for v in vals if len(_norm(v)) >= 3])
        h = sum(1 for v in vals if _mentioned(narr_norm, v))
        pct = round(100.0 * float(h) / max(t, 1), 1)
        util_by_cat[cat] = {"mentioned": h, "total_values": t, "utilization_pct": pct}
        total_hits += h
        total_vals += t
    util_overall_pct = round(100.0 * float(total_hits) / max(total_vals, 1), 1)
    utilization_pass = (util_overall_pct / 100.0) >= float(coverage_expectation)

    # 5) Fact presence (core grounding) — no URIs, just human-readable strings
    fact_presence = _evaluate_fact_presence(narrative, facts)
    fact_presence_pass = (fact_presence["overall"]["presence_pct"] / 100.0) >= float(coverage_expectation)

    # 6) Readability
    readability = {}
    if textstat:
        try:
            readability = {
                "flesch_reading_ease": float(textstat.flesch_reading_ease(narrative)),
                "smog_index": float(textstat.smog_index(narrative)),
                "automated_readability_index": float(textstat.automated_readability_index(narrative)),
            }
        except Exception:
            readability = {"error": "textstat failed"}

    # 7) Length + flow
    length_report = _length_target_ok(narrative, persona)
    flow_report = _flow_cues(narrative)

    return {
        "coverage": coverage,
        "coverage_expectation": coverage_expectation,
        "coverage_pass": coverage_pass,

        "rows_stats": rows_stats,
        "label_coverage_pct": label_presence_pct,

        "utilization": {
            "by_category": util_by_cat,
            "overall": {
                "mentioned": total_hits,
                "total_values": total_vals,
                "utilization_pct": util_overall_pct,
            },
        },
        "utilization_expectation": coverage_expectation,
        "utilization_pass": utilization_pass,

        "fact_presence": fact_presence,          # <-- core grounding metric
        "fact_presence_expectation": coverage_expectation,
        "fact_presence_pass": fact_presence_pass,

        # URI/unsupported-entity checks removed by design
        "readability": readability,
        "length_report": length_report,
        "flow_report": flow_report,
    }

def summarize_markdown(results: Dict[str, Any]) -> str:
    cov = results.get("coverage", {}).get("overall", {})
    util = results.get("utilization", {}).get("overall", {})
    row = results.get("rows_stats", {}).get("overall", {})
    lab_overall = results.get("label_coverage_pct", {}).get("overall", 0.0)
    length = results.get("length_report", {})
    flow = results.get("flow_report", {})
    fp = results.get("fact_presence", {}).get("overall", {})

    ce = results.get("coverage_expectation", 0.75)
    ue = results.get("utilization_expectation", 0.75)
    fe = results.get("fact_presence_expectation", 0.75)

    cpass = "✅" if results.get("coverage_pass") else "⚠️"
    upass = "✅" if results.get("utilization_pass") else "⚠️"
    fppass = "✅" if results.get("fact_presence_pass") else "⚠️"

    lines: List[str] = []
    lines.append("# Enhanced Evaluation Summary")
    lines.append("")
    lines.append(f"- **Answer rate (CQs):** {cov.get('answer_rate_pct', 0)}% "
                 f"(answered {cov.get('answered', 0)} / selected {cov.get('selected', 0)}) "
                 f"— target {int(ce*100)}% {cpass}")
    lines.append(f"- **Fact presence (core grounding):** {fp.get('presence_pct', 0)}% "
                 f"(present {fp.get('present', 0)} / total {fp.get('total', 0)}) "
                 f"— target {int(fe*100)}% {fppass}")
    lines.append(f"- **Fact utilization (values mentioned):** {util.get('utilization_pct', 0)}% "
                 f"(used {util.get('mentioned', 0)} / {util.get('total_values', 0)}) "
                 f"— target {int(ue*100)}% {upass}")
    lines.append(f"- **Retrieval yield:** {row.get('total_rows', 0)} total rows; "
                 f"avg rows/CQ ≈ {row.get('avg_rows_per_cq', 0)}")
    lines.append(f"- **Items with labels:** {lab_overall}%")
    if length:
        ok = "✅" if length.get("within_target") else "⚠️"
        lines.append(f"- **Length target ({length.get('length_setting', '')}):** {ok} "
                     f"{length.get('chars', 0)} chars "
                     f"(target {length.get('target_range', [0,0])[0]}–{length.get('target_range', [0,0])[1]})")
    lines.append(f"- **Flow cues found:** {flow.get('count', 0)} "
                 f"({', '.join(flow.get('transition_cues_found', [])[:6])})")

    # Add quick list of up to 6 missing facts (human-readable)
    missing = results.get("fact_presence", {}).get("missing_facts", []) or []
    if missing:
        lines.append("\n**Examples of missing facts (up to 6):**")
        for m in missing[:6]:
            q = (m.get("question") or m.get("cq_id") or "fact").strip()
            sv = " | ".join([str(x) for x in (m.get("sample_values") or [])])
            lines.append(f"- {q}  —  {sv}")

    return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Enhanced narrative evaluation (URI-free, fact presence focused).")
    ap.add_argument("narrative_output_json", type=str, help="Path to narrative_output.json")
    ap.add_argument("--coverage-expectation", type=float, default=0.75, help="Target fraction for coverage/utilization/fact presence (0–1)")
    # kg-ttl kept out intentionally (we no longer use URIs/labels from KG for grounding)
    args = ap.parse_args()

    res = evaluate_enhanced(args.narrative_output_json, None, args.coverage_expectation)
    print(json.dumps(res, indent=2, ensure_ascii=False))
