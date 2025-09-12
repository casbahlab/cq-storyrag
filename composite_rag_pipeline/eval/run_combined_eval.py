
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_combined_eval.py

Run both evaluators and produce a unified, dissertation-ready report
(One-Pager) that merges facts-support and narrative metrics (with figures).

Expected layout:
  <RUN_DIR>/<RAG_TYPE>/answers_<RAG_TYPE>.jsonl
  <RUN_DIR>/<RAG_TYPE>/story_<RAG_TYPE>.md

RAG_TYPE is one of: KG | Hybrid | Graph | ALL (ALL tries each present).

Examples:
  python run_combined_eval.py --run-dir data/Emma-.../run-01 --rag-type KG --unified-report
  python run_combined_eval.py --run-dir data/Emma-.../run-01 --rag-type ALL --support-extra "--light-clean --tf-th 0.4" --unified-report
"""

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Matplotlib for figures (no seaborn/colors)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

DEFAULT_SUPPORT_SCRIPT = "support_ctx_reset_refactored.py"
DEFAULT_NARRATIVE_SCRIPT = "narrative_eval.py"

def run_cmd(cmd: List[str]) -> int:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode

def find_script(start_dir: Path, name: str) -> Path:
    here = Path(__file__).resolve().parent
    for c in (here / name, Path.cwd() / name, Path(name)):
        if c.exists():
            return c
    return Path(name)

def collect_support_summary(path: Path) -> Dict[str, Any]:
    if path.with_suffix(".json").exists():
        try:
            return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        return {"summary_csv_rows": rows}
    except Exception as e:
        return {"error": f"could not read summary: {e}"}

def collect_narrative_summary(out_dir: Path) -> Dict[str, Any]:
    out = {}
    summ_csv = out_dir / "narrative_summary.csv"
    if summ_csv.exists():
        try:
            rows = list(csv.DictReader(summ_csv.open(encoding="utf-8")))
            out["narrative_summary_csv"] = rows
        except Exception as e:
            out["narrative_summary_csv_error"] = str(e)
    per_csv = out_dir / "narrative_per_section.csv"
    if per_csv.exists():
        try:
            rows = list(csv.DictReader(per_csv.open(encoding="utf-8")))
            out["narrative_per_section_csv_rows"] = len(rows)
        except Exception as e:
            out["narrative_per_section_csv_error"] = str(e)
    for alt in ["narrative_summary_annotated.csv", "narrative_report.md",
                "narrative_report_annotated.md", "narrative_per_section.csv"]:
        p = out_dir / alt
        if p.exists():
            out[alt] = str(p)
    return out

# ---------- Unified report helpers ----------

def _load_csv(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        return list(csv.DictReader(path.open(encoding="utf-8")))
    except Exception:
        return None

def _find_col(cols: List[str], *candidates: str) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for c in cols:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    return None

def _as_percent_str(v: Any) -> str:
    try:
        if v is None: return "NA"
        x = float(v)
        if math.isnan(x): return "NA"
        return f"{x*100:.1f}%" if x <= 1.0 else f"{x:.1f}%"
    except Exception:
        return str(v)

def _plot_per_beat_support(df_rows: List[Dict[str, Any]], out_dir: Path):
    if not df_rows: return []
    cols = list(df_rows[0].keys())
    section_col = _find_col(cols, "section","beat","title") or "__section__"
    # compute support rate vector
    support_rate_col = _find_col(cols, "support_rate","support%","support_rate_pct","support")
    supp_sent_col = _find_col(cols, "supported_sentences","n_supported","supported")
    total_sent_col = _find_col(cols, "total_sentences","n_sentences","sentences_total","sent_total")

    xs = []
    ys = []
    for i, r in enumerate(df_rows):
        xs.append(r.get(section_col, f"S{i+1}"))
        if support_rate_col and r.get(support_rate_col) not in (None, ""):
            try:
                ys.append(float(r[support_rate_col]))
            except Exception:
                ys.append(0.0)
        elif supp_sent_col and total_sent_col:
            try:
                num = float(r.get(supp_sent_col, 0))
                den = float(r.get(total_sent_col, 0)) or 1.0
                ys.append(num/den)
            except Exception:
                ys.append(0.0)
        else:
            ys.append(0.0)

    plt.figure()
    plt.bar(range(len(xs)), ys)
    plt.xticks(range(len(xs)), xs, rotation=45, ha="right")
    plt.ylabel("Support rate")
    plt.title("Per-beat Support Rate")
    plt.tight_layout()
    p_png = out_dir / "fig_per_beat_support.png"
    p_pdf = out_dir / "fig_per_beat_support.pdf"
    plt.savefig(p_png, dpi=300); plt.savefig(p_pdf); plt.close()
    return [p_png, p_pdf]

def _plot_per_beat_nqi(df_rows: List[Dict[str, Any]], out_dir: Path):
    if not df_rows: return []
    cols = list(df_rows[0].keys())
    section_col = _find_col(cols, "section","beat","title") or "__section__"
    score_col = _find_col(cols, "nqi_lite","nqi-lite","nqi","score")
    if not score_col: return []
    xs = [r.get(section_col, f"S{i+1}") for i, r in enumerate(df_rows)]
    ys = []
    for r in df_rows:
        try:
            ys.append(float(r.get(score_col, 0.0)))
        except Exception:
            ys.append(0.0)
    plt.figure()
    plt.barh(xs, ys)
    plt.xlabel(score_col)
    plt.ylabel("Section")
    plt.title("Narrative Quality Score per Section")
    plt.tight_layout()
    p_png = out_dir / "fig_per_beat_nqi.png"
    p_pdf = out_dir / "fig_per_beat_nqi.pdf"
    plt.savefig(p_png, dpi=300); plt.savefig(p_pdf); plt.close()
    return [p_png, p_pdf]

def _plot_readability_vs_cohesion(df_rows: List[Dict[str, Any]], out_dir: Path):
    if not df_rows: return []
    cols = list(df_rows[0].keys())
    fre_col = _find_col(cols, "flesch_reading_ease","flesch")
    eov_col = _find_col(cols, "entity_overlap_mean","entity_overlap")
    if not (fre_col and eov_col): return []
    xs, ys = [], []
    for r in df_rows:
        try:
            x = float(r.get(fre_col, 0.0))
            y = float(r.get(eov_col, 0.0))
        except Exception:
            continue
        xs.append(x); ys.append(y)
    if not xs: return []
    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("Flesch Reading Ease")
    plt.ylabel("Entity Overlap (mean)")
    plt.title("Readability vs Cohesion per Section")
    plt.tight_layout()
    p_png = out_dir / "fig_readability_vs_cohesion.png"
    p_pdf = out_dir / "fig_readability_vs_cohesion.pdf"
    plt.savefig(p_png, dpi=300); plt.savefig(p_pdf); plt.close()
    return [p_png, p_pdf]

def generate_unified_report(eval_root: Path) -> Path:
    """Create Evaluation_OnePager.md and figures inside eval_root (per RAG)."""
    # Expected files
    support_overview_fp = eval_root / "support_overview.csv"
    support_summary_fp  = eval_root / "support_summary.csv"
    coverage_summary_fp = eval_root / "coverage_summary.csv"
    narr_dir            = eval_root / "narrative"
    narr_summary_fp     = narr_dir / "narrative_summary.csv"
    narr_per_section_fp = narr_dir / "narrative_per_section.csv"

    # Load
    so_rows = _load_csv(support_overview_fp) or []
    ss_rows = _load_csv(support_summary_fp) or []
    ns_rows = _load_csv(narr_summary_fp) or []
    nps_rows= _load_csv(narr_per_section_fp) or []

    # Headline KPIs
    headline = {}
    if so_rows:
        row = so_rows[0]
        cols = list(row.keys())
        support_rate_col  = _find_col(cols, "support_rate","support%","support")
        coverage_rate_col = _find_col(cols, "coverage_rate","coverage%","coverage")
        total_sent_col    = _find_col(cols, "total_sentences","n_sentences","sentences_total","sent_total")
        supp_sent_col     = _find_col(cols, "supported_sentences","n_supported","supported")
        ev_total_col      = _find_col(cols, "evidence_total","n_evidence","evidence_items","evidence_count")
        ev_used_col       = _find_col(cols, "evidence_used","evidence_covered","covered_evidence","evidence_used_count")
        headline.update({
            "support_rate":  row.get(support_rate_col),
            "coverage_rate": row.get(coverage_rate_col),
            "total_sentences": row.get(total_sent_col),
            "supported_sentences": row.get(supp_sent_col),
            "evidence_total": row.get(ev_total_col),
            "evidence_used": row.get(ev_used_col),
        })
    if ns_rows:
        row = ns_rows[0]
        cols = list(row.keys())
        def pick(*names):
            c = _find_col(cols, *names)
            return row.get(c) if c else None
        headline.update({
            "flesch_reading_ease": pick("flesch_reading_ease","flesch"),
            "fk_grade":            pick("flesch_kincaid_grade","fk_grade"),
            "entity_overlap_mean": pick("entity_overlap_mean","entity_overlap"),
            "content_overlap_mean":pick("content_overlap_mean","content_overlap"),
            "transition_rate":     pick("transition_rate","transitions"),
            "trigram_rep_rate":    pick("trigram_repetition_rate","trigram_rep","trigram"),
            "near_dupe_rate":      pick("near_duplicate_rate","near_dupe_rate","near_dupes"),
            "topic_variance":      pick("topic_variance","topic_var"),
        })

    # Figures
    figs = []
    figs += _plot_per_beat_support(ss_rows, eval_root)
    figs += _plot_per_beat_nqi(nps_rows, eval_root)
    figs += _plot_readability_vs_cohesion(nps_rows, eval_root)

    # Markdown
    kpis = []
    if "support_rate" in headline or "coverage_rate" in headline:
        kpis.append(f"- **Support rate:** {_as_percent_str(headline.get('support_rate'))} ({headline.get('supported_sentences')}/{headline.get('total_sentences')})")
        kpis.append(f"- **Coverage:** {_as_percent_str(headline.get('coverage_rate'))} ({headline.get('evidence_used')}/{headline.get('evidence_total')})")
    if headline.get("flesch_reading_ease") is not None or headline.get("fk_grade") is not None:
        kpis.append(f"- **Flesch Reading Ease:** {headline.get('flesch_reading_ease')}  |  **FK Grade:** {headline.get('fk_grade')}")
    if headline.get("entity_overlap_mean") is not None:
        kpis.append(f"- **Cohesion:** entity overlap {headline.get('entity_overlap_mean')}, content overlap {headline.get('content_overlap_mean')}")
    if headline.get("transition_rate") is not None:
        kpis.append(f"- **Transitions rate:** {headline.get('transition_rate')}")
    if headline.get("trigram_rep_rate") is not None or headline.get("near_dupe_rate") is not None:
        kpis.append(f"- **Redundancy:** trigram rep {headline.get('trigram_rep_rate')}, near-dup {headline.get('near_dupe_rate')}")
    if headline.get("topic_variance") is not None:
        kpis.append(f"- **Topic variance:** {headline.get('topic_variance')}")

    report_md = eval_root / "Evaluation_OnePager.md"
    report_md.write_text(
        "# Evaluation One-Pager\n\n"
        "## Headline KPIs\n" +
        (("\n".join(kpis)) if kpis else "- (No KPI rows found; check CSV contents.)") +
        "\n\n## Figures\n" +
        f"- Per-beat Support Rate: `{'fig_per_beat_support.png' if any('fig_per_beat_support' in f.name for f in figs) else 'n/a'}`\n" +
        f"- Narrative Quality per Section: `{'fig_per_beat_nqi.png' if any('fig_per_beat_nqi' in f.name for f in figs) else 'n/a'}`\n" +
        f"- Readability vs Cohesion: `{'fig_readability_vs_cohesion.png' if any('fig_readability_vs_cohesion' in f.name for f in figs) else 'n/a'}`\n\n" +
        "All figure files are saved alongside this report in this folder.\n\n"
        "## Provenance\n"
        f"- Support files: `{support_overview_fp.name}`, `{support_summary_fp.name}`, `{coverage_summary_fp.name}`\n"
        f"- Narrative files: `{narr_summary_fp.name}`, `{narr_per_section_fp.name}`\n",
        encoding="utf-8"
    )
    print("[OK] Wrote unified report →", report_md)
    return report_md

# ---------- Runner ----------

def evaluate_one(run_dir: Path,
                 rag_type: str,
                 out_root: Optional[Path],
                 support_script: Path,
                 narrative_script: Path,
                 beats: str = "auto",
                 equal_k: int = 4,
                 with_coherence: bool = True,
                 annotate: bool = True,
                 nqi_lite: bool = True,
                 support_extra: str = "",
                 unified_report: bool = False) -> Dict[str, Any]:

    input_dir = run_dir / rag_type
    answers = input_dir / f"answers_{rag_type}.jsonl"
    story   = input_dir / f"story_{rag_type}.md"

    if not answers.exists():
        raise FileNotFoundError(f"Missing answers file: {answers}")
    if not story.exists():
        raise FileNotFoundError(f"Missing story file: {story}")

    out_dir = (out_root / rag_type) if out_root else (input_dir / "eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Facts support
    sup_out_csv = out_dir / "support_summary.csv"
    sup_out_summary = out_dir / "support_overview.csv"
    cov_out_csv = out_dir / "coverage_evidence.csv"
    cov_out_summary = out_dir / "coverage_summary.csv"

    sup_cmd = [
        sys.executable, str(support_script),
        "--answers", str(answers),
        "--out-csv", str(sup_out_csv),
        "--out-summary", str(sup_out_summary),
        "--emit-coverage",
        "--cov-out-csv", str(cov_out_csv),
        "--cov-out-summary", str(cov_out_summary),
    ]
    if support_extra:
        sup_cmd += shlex.split(support_extra)
    rc1 = run_cmd(sup_cmd)
    if rc1 != 0:
        print("[WARN] support evaluator returned non-zero exit code:", rc1)

    # 2) Narrative eval
    narr_out_dir = out_dir / "narrative"
    narr_out_dir.mkdir(parents=True, exist_ok=True)
    narr_cmd = [
        sys.executable, str(narrative_script),
        "-i", str(story),
        "-o", str(narr_out_dir),
        "--beats", beats,
        "--neardupe-th", "0.90"
    ]
    if beats == "equal":
        narr_cmd += ["--equal-k", str(equal_k)]
    if with_coherence:
        narr_cmd.append("--with-coherence")
    if annotate:
        narr_cmd.append("--annotate")
    if nqi_lite:
        narr_cmd.append("--nqi-lite")
    rc2 = run_cmd(narr_cmd)
    if rc2 != 0:
        print("[WARN] narrative evaluator returned non-zero exit code:", rc2)

    # 3) Unified report
    report_path = None
    if unified_report:
        report_path = generate_unified_report(out_dir)

    # 4) Aggregate meta JSON
    meta = {
        "rag_type": rag_type,
        "input_dir": str(input_dir.resolve()),
        "answers_file": str(answers.resolve()),
        "story_file": str(story.resolve()),
        "support": {
            "out_csv": str(sup_out_csv),
            "out_summary": str(sup_out_summary),
            "coverage_csv": str(cov_out_csv),
            "coverage_summary": str(cov_out_summary),
            "support_preview": collect_support_summary(sup_out_summary),
            "coverage_preview": collect_support_summary(cov_out_summary),
        },
        "narrative": collect_narrative_summary(narr_out_dir),
        "unified_report": str(report_path) if report_path else None,
        "out_dir": str(out_dir.resolve())
    }
    meta_path = out_dir / "combined_eval_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[OK] Wrote combined meta:", meta_path)
    return meta

def main():
    ap = argparse.ArgumentParser(description="Run support + narrative evaluation for a run folder and RAG type(s), and generate a unified report.")
    ap.add_argument("--run-dir", required=True, help="e.g., data/Emma-.../run-01")
    ap.add_argument("--rag-type", required=True, help="KG | Hybrid | Graph | ALL")
    ap.add_argument("--out-root", default=None, help="Output root (default: <run-dir>/<rag_type>/eval_out)")
    ap.add_argument("--support-script", default=None, help="Path to support_ctx_reset_refactored.py")
    ap.add_argument("--narrative-script", default=None, help="Path to narrative_eval.py")
    ap.add_argument("--support-extra", default="", help="Extra flags to pass to support script (quoted string)")
    ap.add_argument("--beats", default="auto", choices=["auto","none","equal"], help="Beat splitting mode (narrative_eval)")
    ap.add_argument("--equal-k", type=int, default=4, help="If beats=equal, number of chunks")
    ap.add_argument("--no-coherence", action="store_true", help="Disable coherence (TF-IDF) in narrative eval")
    ap.add_argument("--no-annotate", action="store_true", help="Disable annotated outputs")
    ap.add_argument("--no-nqi-lite", action="store_true", help="Disable NQI_LITE")
    ap.add_argument("--unified-report", action="store_true", help="Generate a unified one-pager report and figures")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rag_types = ["KG","Hybrid","Graph"] if args.rag_type.upper() == "ALL" else [args.rag_type]
    out_root = Path(args.out_root) if args.out_root else None

    support_script = Path(args.support_script) if args.support_script else find_script(run_dir, DEFAULT_SUPPORT_SCRIPT)
    narrative_script = Path(args.narrative_script) if args.narrative_script else find_script(run_dir, DEFAULT_NARRATIVE_SCRIPT)

    results = []
    for rt in rag_types:
        try:
            meta = evaluate_one(
                run_dir=run_dir,
                rag_type=rt,
                out_root=out_root,
                support_script=support_script,
                narrative_script=narrative_script,
                beats=args.beats,
                equal_k=args.equal_k,
                with_coherence=not args.no_coherence,
                annotate=not args.no_annotate,
                nqi_lite=not args.no_nqi_lite,
                support_extra=args.support_extra,
                unified_report=args.unified_report,
            )
            results.append(meta)
        except FileNotFoundError as e:
            print("[SKIP]", e)

    # If ALL, write aggregate
    if len(results) > 1:
        top = run_dir / ("combined_eval_ALL.json" if args.rag_type.upper()=="ALL" else f"combined_eval_{args.rag_type}.json")
        top.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("[OK] Wrote aggregate:", top)

if __name__ == "__main__":
    main()
