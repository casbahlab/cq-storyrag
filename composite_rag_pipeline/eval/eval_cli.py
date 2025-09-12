#!/usr/bin/env python3
"""
CLI for unified RAG evaluation.

Usage:
  python eval/eval_cli.py --root eval/data [--report-dir eval/reports/topic]

It recursively discovers any directory that contains at least:
  - answers_*.jsonl
  - plan_with_evidence_*.json (or *_clean.json)
and optionally:
  - story_*_clean.md (or story_*.md)

It then runs evaluate_rag.evaluate_many on all discovered runs,
and writes eval_report_all.json / .md (+ CSVs) to the report directory.
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from evaluate_rag import evaluate_many
except Exception as e:
    print("ERROR: Could not import evaluate_rag from", HERE)
    print("Make sure evaluate_rag.py is in the same folder as this CLI.")
    print("Underlying error:", e)
    sys.exit(1)

# at top (keep ANS_RE / PLAN_RE as-is)
ANS_RE   = re.compile(r"answers_(?P<pat>[\w\-]+)\.jsonl$", re.I)
PLAN_RE  = re.compile(r"plan_with_evidence_(?P<pat>[\w\-]+)(?:_clean)?\.json(?:l)?$", re.I)

def _pick_story_file(dirpath: Path, pat: str) -> str | None:
    # robust, case-insensitive selection
    candidates = [f for f in dirpath.iterdir() if f.is_file() and f.suffix.lower() == ".md"]
    # prefer *_clean.md for this pattern
    clean = [f for f in candidates if re.fullmatch(fr"story_{pat}_clean\.md", f.name, flags=re.I)]
    if clean:
        return str(clean[0])
    plain = [f for f in candidates if re.fullmatch(fr"story_{pat}\.md", f.name, flags=re.I)]
    if plain:
        return str(plain[0])
    # last resort: any story_*_clean.md in the folder
    any_clean = [f for f in candidates if re.search(r"story_.*_clean\.md$", f.name, flags=re.I)]
    if any_clean:
        return str(any_clean[0])
    return None

def discover_runs(root: Path):
    runs = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        answers, plans = {}, {}

        for f in p.glob("*"):
            if not f.is_file():
                continue
            m = ANS_RE.match(f.name)
            if m:
                answers[m.group("pat")] = f
                continue
            m = PLAN_RE.match(f.name)
            if m:
                plans[m.group("pat")] = f
                continue

        for pat, ans in answers.items():
            pl = plans.get(pat)
            if not pl:
                continue
            cfg = {"name": pat, "answers": str(ans), "plan": str(pl)}
            picked_story = _pick_story_file(p, pat)
            if picked_story:
                cfg["story"] = picked_story
            runs.append(cfg)
    return runs



def main():
    ap = argparse.ArgumentParser(description="Unified RAG evaluator CLI")
    ap.add_argument("--root", required=True, type=Path, help="Root folder to search for runs")
    ap.add_argument("--report-dir", type=Path, default=None, help="Directory to write eval_report_all.{json,md} (+CSVs)")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print("ERROR: root does not exist:", root)
        sys.exit(2)

    configs = discover_runs(root)
    if not configs:
        print("No runs discovered under:", root)
        print("Expected at least answers_*.jsonl + plan_with_evidence_*.json in the same folder.")
        sys.exit(0)

    per_section_df, summary_df, report = evaluate_many(configs)

    outdir = args.report_dir or (root / "_reports" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    (outdir / "eval_report_all.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write Markdown Summary
    lines = ["# Unified RAG Evaluation Report"]
    for run in report["runs"]:
        lines.append(f"## {run['name']}")
        s = run["summary"]
        lines.append(f"- Sections: {s.get('sections',0)}")
        lines.append(f"- Avg support ratio: {round(s.get('avg_support_ratio',0.0),3)}")
        lines.append(f"- Avg coverage: {round(s.get('avg_coverage',0.0),3)}")
        lines.append(f"- Avg redundancy: {round(s.get('avg_redundancy',0.0),3)}")
        lines.append(f"- Avg sentences: {round(s.get('avg_sentences',0.0),3)}")
        lines.append(f"- Avg cohesiveness: {round(s.get('avg_cohesiveness',0.0),3)}")
        if run.get("story_quality"):
            q = run["story_quality"]["summary"]
            lines.append(
                f"- Story Flesch: {q.get('Flesch_RE')} | FK Grade: {q.get('FK_Grade')} | "
                f"Redundancy: {q.get('redundancy_rate_adjacent')} | Temporal OK: {q.get('temporal_consistency')}"
            )
        lines.append("")
    (outdir / "eval_report_all.md").write_text("\n".join(lines), encoding="utf-8")

    # CSVs
    try:
        import pandas as pd
        per_section_df.to_csv(outdir / "per_section.csv", index=False)
        summary_df.to_csv(outdir / "summary.csv", index=False)
    except Exception as e:
        print("Note: could not write CSVs:", e)

    print("Wrote reports to:", outdir)
    print("Runs evaluated:", [c['name'] for c in configs])
    for c in configs:
        print(" -", c['name'])
        print("   answers:", c['answers'])
        print("   plan:   ", c['plan'])
        if 'story' in c:
            print("   story:  ", c['story'])





if __name__ == "__main__":
    main()
