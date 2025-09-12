# trace_pack.py  (replace previous with this)

from __future__ import annotations
import json, re, datetime
from pathlib import Path
from typing import Any, Dict, List

# ---------- sentence + value helpers ----------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(“"\'`])')

def _to_sentences(md_text: str) -> List[str]:
    text = "\n".join(
        ln for ln in md_text.splitlines()
        if not ln.startswith("#") and not ln.startswith("```")
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)           # remove bold markers
    text = re.sub(r"<details>.*?</details>", " ", text, flags=re.S)
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

def _flatten_row_values(rows: List[Dict[str, Any]]) -> List[str]:
    vals: List[str] = []
    for r in rows or []:
        for v in r.values():
            if v is None: continue
            s = str(v).strip().strip('"')
            if s.startswith("<") and s.endswith(">"):  # IRI → skip in text matching
                continue
            if len(s) >= 3:
                vals.append(s)
    seen=set(); out=[]
    for v in vals:
        lo=v.lower()
        if lo not in seen:
            seen.add(lo); out.append(v)
    return out

# ---------- FACTS ----------
def _facts_from_rows(cq_id: str, beat: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts = []
    for r in rows or []:
        facts.append({
            "cq_id": cq_id,
            "beat": beat,
            "vars": list(r.keys()),
            "values": [str(r[k]) for k in r.keys()],
            "row": r
        })
    return facts

def _collect_facts(plan_final: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for it in plan_final.get("items", []):
        facts.extend(_facts_from_rows(it.get("id"), it.get("beat","Unspecified"), it.get("rows") or []))
    return facts

# ---------- narrative_output.json ----------
def build_narrative_output(
    plan_initial: Dict[str, Any],
    plan_retrieved: Dict[str, Any],
    plan_final: Dict[str, Any],
    md_text: str,
    kg_files: List[str],
    prompt: Dict[str, Any] | None
) -> Dict[str, Any]:
    created = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    beats = plan_final.get("beats", [])
    # Detailed plan (final) grouped by beat
    detailed_plan = {b: [] for b in beats}
    for it in plan_final.get("items", []):
        b = it.get("beat", "Unspecified")
        detailed_plan.setdefault(b, []).append({
            "cq_id": it.get("id"),
            "text": it.get("question") or it.get("text"),
            "category": b,
            "beat": b,
            "sparql": it.get("sparql")
        })

    # Evidence per beat
    evidence = {b: [] for b in detailed_plan.keys()}
    for it in plan_final.get("items", []):
        b = it.get("beat", "Unspecified")
        evidence[b].append({
            "cq_id": it.get("id"),
            "question": it.get("question") or it.get("text"),
            "rows": it.get("rows") or [],
            "row_count": it.get("row_count", 0),
            "labels": {},
            "provenance": {"kg": kg_files[0] if kg_files else ""}
        })

    # Facts (flat list, good for eval/search)
    facts: List[Dict[str, Any]] = []
    for it in plan_final.get("items", []):
        facts.extend(_facts_from_rows(it.get("id"), it.get("beat","Unspecified"), it.get("rows") or []))

    total_limit = int(plan_final.get("total_limit", len(plan_final.get("items", []))))
    diagnostics = {
        "length": plan_final.get("length"),
        "budget_total": total_limit,
        "counts": {b: len(detailed_plan.get(b, [])) for b in detailed_plan.keys()},
        "retriever_stats": plan_retrieved.get("retriever_stats"),
        "topup_stats": plan_final.get("topup_stats")
    }

    return {
        "plan": {
            "created_at": created,
            "initial":   plan_initial,   # snapshot before retrieval
            "retrieved": plan_retrieved, # items annotated with kg_ok/rows (pre-topup)
            "final":     plan_final,     # after top-up
            "diagnostics": diagnostics
        },
        "generation": {
            "mode": ("llm" if prompt else "template"),
            "prompt": prompt or {"template": "Q/A renderer with beat sections"}
        },
        "evidence": evidence,
        "facts": facts,
        "narrative": md_text
    }

# ---------- trace.json (facts_trace + unmatched sentences) ----------
def build_facts_trace(md_text: str, plan_final: Dict[str, Any]) -> Dict[str, Any]:
    sents = _to_sentences(md_text)
    used = set()
    facts_trace = []
    for it in plan_final.get("items", []):
        cqid = it.get("id")
        b    = it.get("beat","Unspecified")
        q    = it.get("question") or it.get("text") or ""
        values = _flatten_row_values(it.get("rows") or [])
        matches = []
        for i, sent in enumerate(sents):
            lo = sent.lower()
            hit = [v for v in values if v.lower() in lo]
            if hit:
                matches.append({
                    "sentence_index": i,
                    "sentence": sent if len(sent)<220 else (sent[:217]+"…"),
                    "matched_values": hit[:8]
                })
                used.add(i)
        facts_trace.append({"category": b, "cq_id": cqid, "question": q, "matches": matches})

    unmatched = [{"sentence_index": i, "sentence": s} for i,s in enumerate(sents) if i not in used]
    return {"facts_trace": facts_trace, "unmatched_sentences": unmatched}

# ---------- eval summary (markdown) ----------
def write_eval_summary(md_path: str, plan_final: Dict[str, Any], out_path: str):
    md_text = Path(md_path).read_text(encoding="utf-8")
    trace = build_facts_trace(md_text, plan_final)
    total = len(plan_final.get("items", []))
    answered = sum(1 for it in plan_final.get("items", []) if (it.get("rows") or []))
    answer_rate = (100.0 * answered / total) if total else 0.0

    all_vals = []
    used_vals = set()
    story = " ".join(_to_sentences(md_text)).lower()
    for it in plan_final.get("items", []):
        for v in _flatten_row_values(it.get("rows") or []):
            all_vals.append(v)
            if v.lower() in story: used_vals.add(v.lower())
    util = (100.0 * len(used_vals) / len(all_vals)) if all_vals else 0.0

    rows_total = sum(len(it.get("rows") or []) for it in plan_final.get("items", []))
    rows_avg = rows_total / max(1, answered)

    lines = [
        "# Evaluation Summary",
        f"- Answer rate: {answer_rate:.1f}% ({answered}/{total})",
        f"- Fact utilization: {util:.1f}% (used {len(used_vals)} / {len(all_vals)})",
        f"- Retrieval yield: {rows_total} rows total; avg {rows_avg:.2f} per answered CQ",
        f"- Beats: {', '.join(plan_final.get('beats', []))}"
    ]
    Path(out_path).write_text("\n".join(lines)+"\n", encoding="utf-8")

def save_all(
    plan_initial: Dict[str, Any],
    plan_retrieved: Dict[str, Any],
    plan_final: Dict[str, Any],
    md_text: str,
    out_dir: str,
    kg_files: List[str],
    prompt: Dict[str, Any] | None = None
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) narrative.md
    md_path = out / "narrative.md"
    md_path.write_text(md_text, encoding="utf-8")

    # 2) narrative_output.json (full bundle)
    bundle = build_narrative_output(plan_initial, plan_retrieved, plan_final, md_text, kg_files, prompt)
    (out / "narrative_output.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) trace.json (facts matched to narrative)
    trace = build_facts_trace(md_text, plan_final)
    (out / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) EVAL_SUMMARY.md
    write_eval_summary(str(md_path), plan_final, str(out / "EVAL_SUMMARY.md"))

    # ── NEW: individual files ──────────────────────────────────────────────────
    # a) plan snapshots (verbatim)
    (out / "plan_initial.json").write_text(json.dumps(plan_initial, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "plan_retrieved.json").write_text(json.dumps(plan_retrieved, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "plan_final.json").write_text(json.dumps(plan_final, ensure_ascii=False, indent=2), encoding="utf-8")

    # b) facts.json (flat list of facts extracted from final plan rows)
    facts = _collect_facts(plan_final)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    # ───────────────────────────────────────────────────────────────────────────