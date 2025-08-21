#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate one run (both RAG & narrative) in a single command.

Assumes this run directory layout:
  <run_dir>/
    retriever/plan_with_evidence_{Pattern}.json
    generator/answers_{Pattern}.jsonl
    generator/story_{Pattern}_clean.md (preferred) or story_{Pattern}.md

Outputs to:
  <out_dir>/ (default: <run_dir>/eval/)
    per_section.csv
    summary.csv
    eval_report_all.json
    eval_report_all.md
    narrative_eval_per_beat.csv
    narrative_eval_summary.csv
    narrative_eval_report.html
    _meta.json
"""

from __future__ import annotations
import argparse, json, re, sys, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Make sure your local modules are importable
THIS = Path(__file__).resolve()
ROOT = THIS.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your existing evaluators
# --- import shims so we can load from either eval/ or repo root ---
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
EVAL_DIR = THIS.parent                # composite_rag_pipeline/eval/
REPO_ROOT = EVAL_DIR.parent           # composite_rag_pipeline/

for p in (EVAL_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evaluate_rag      # lives in composite_rag_pipeline/eval/
try:
    import eval_narrative  # yours lives at repo root
except Exception:
    root_parent = REPO_ROOT.parent
    if str(root_parent) not in sys.path:
        sys.path.insert(0, str(root_parent))
    import eval_narrative


ANSWERS_RE = re.compile(r"answers_(?P<pattern>[A-Za-z]+)\.jsonl$", re.I)

def _pick_answers(gen_dir: Path, prefer_pattern: Optional[str]) -> Tuple[Path, str]:
    """Pick answers file; if pattern provided, prefer that."""
    if prefer_pattern:
        p = gen_dir / f"answers_{prefer_pattern}.jsonl"
        if p.exists():
            return p, prefer_pattern
    # else discover
    cands = sorted(gen_dir.glob("answers_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No answers_*.jsonl under {gen_dir}")
    # prefer Graph if multiple, else first
    for c in cands:
        m = ANSWERS_RE.search(c.name)
        if m and m.group("pattern").lower() == "graph":
            return c, m.group("pattern")
    m = ANSWERS_RE.search(cands[0].name)
    if not m:
        raise RuntimeError(f"Could not deduce pattern from {cands[0].name}")
    return cands[0], m.group("pattern")

def _pick_plan(retriever_dir: Path, pattern: str) -> Path:
    exact = retriever_dir / f"plan_with_evidence_{pattern}.json"
    if exact.exists():
        return exact
    cands = sorted(retriever_dir.glob(f"plan_with_evidence_{pattern}*.json"))
    if cands:
        return cands[0]
    raise FileNotFoundError(f"No plan_with_evidence_{pattern}*.json under {retriever_dir}")

def _pick_story(gen_dir: Path, pattern: str) -> Optional[Path]:
    clean = gen_dir / f"story_{pattern}_clean.md"
    raw   = gen_dir / f"story_{pattern}.md"
    if clean.exists(): return clean
    if raw.exists():   return raw
    return None

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows

def _write_md_index(report_dir: Path, cfg: Dict[str,str], persona: Optional[str], length: Optional[str]):
    lines = [
        "# Evaluation (single run)",
        f"- pattern: `{cfg.get('pattern','?')}`",
        f"- persona: `{persona or ''}`",
        f"- length:  `{length or ''}`",
        f"- answers: `{cfg['answers']}`",
        f"- plan:    `{cfg['plan']}`",
    ]
    if cfg.get("story"): lines.append(f"- story:   `{cfg['story']}`")
    (report_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Evaluate one run (RAG + narrative)")
    ap.add_argument("--run-dir", required=True, help="Path to the run folder containing retriever/ and generator/")
    ap.add_argument("--pattern", default=None, help="Pattern name (Graph|KG|Hybrid). Optional; auto-detected if omitted.")
    ap.add_argument("--persona", default=None, help="Persona tag to attach (e.g., Emma, Luca)")
    ap.add_argument("--length",  default=None, help="Length tag to attach (Short|Medium|Long)")
    ap.add_argument("--out-dir", default=None, help="Where to write reports (default: <run-dir>/eval)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    retriever = run_dir / "retriever"
    generator = run_dir / "generator"

    if not retriever.exists() or not generator.exists():
        raise SystemExit(f"[ERR] {run_dir} must contain retriever/ and generator/")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "eval")
    _ensure_dir(out_dir)

    # --- locate artifacts
    answers_path, pattern = _pick_answers(generator, args.pattern)
    plan_path    = _pick_plan(retriever, pattern)
    story_path   = _pick_story(generator, pattern)

    cfg = {"name": f"{pattern}@{run_dir}", "answers": str(answers_path), "plan": str(plan_path), "pattern": pattern}
    if story_path:
        cfg["story"] = str(story_path)

    # --- RAG evaluation (single-config via evaluate_many)
    try:
        per_section_df, summary_df, report_json = evaluate_rag.evaluate_many([cfg])

        # annotate persona/length if provided
        for df in (per_section_df, summary_df):
            if df is not None and not df.empty:
                if args.persona is not None: df["persona"] = args.persona
                if args.length  is not None: df["length"]  = args.length
                if "pattern" not in df.columns:
                    df["pattern"] = pattern

        # write outputs
        per_section_csv = out_dir / "per_section.csv"
        summary_csv     = out_dir / "summary.csv"
        report_json_p   = out_dir / "eval_report_all.json"
        report_md_p     = out_dir / "eval_report_all.md"

        if per_section_df is not None and not per_section_df.empty:
            per_section_df.to_csv(per_section_csv, index=False)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_csv(summary_csv, index=False)
        report_json_p.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

        # minimal MD
        md = [
            "# Evaluation Report (RAG)",
            f"## {cfg['name']}",
            f"- answers: `{cfg['answers']}`",
            f"- plan: `{cfg['plan']}`",
        ]
        if story_path:
            md.append(f"- story: `{cfg['story']}`")
        report_md_p.write_text("\n".join(md) + "\n", encoding="utf-8")

        print(f"[OK] RAG eval → {per_section_csv}, {summary_csv}, {report_json_p}, {report_md_p}")

    except Exception:
        traceback.print_exc()
        raise SystemExit("[ERR] RAG evaluation failed")

    # --- Narrative evaluation (single run)
    try:
        rows = eval_narrative.read_jsonl(answers_path)
        narr_df = eval_narrative.evaluate_rows(rows, persona=args.persona)
        narr_csv, narr_sum_csv, narr_html = eval_narrative.write_reports(narr_df, out_dir)

        # Tag narrative summary with persona/length for easier later aggregation
        try:
            ns = pd.read_csv(narr_sum_csv)
            if args.persona is not None: ns["persona"] = args.persona
            if args.length  is not None: ns["length"]  = args.length
            ns.to_csv(narr_sum_csv, index=False)
        except Exception:
            pass

        print(f"[OK] Narrative eval → {narr_csv}, {narr_sum_csv}, {narr_html}")

    except Exception:
        traceback.print_exc()
        raise SystemExit("[ERR] Narrative evaluation failed")

    # --- meta & index
    meta = {
        "run_dir": str(run_dir),
        "pattern": pattern,
        "persona": args.persona,
        "length":  args.length,
        "answers": str(answers_path),
        "plan":    str(plan_path),
        "story":   str(story_path) if story_path else None,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_md_index(out_dir, cfg, args.persona, args.length)

    print(f"\n[OK] Single-run evaluation complete → {out_dir}\n")


if __name__ == "__main__":
    main()
