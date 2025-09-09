#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
aggregate_eval_runs.py (meta-aware, metrics include)

- Discovers run-* folders under one or more experiment directories
- Optionally calls run_combined_eval.py for each run (unless --skip-run-if-json-exists)
- Loads one of:
    <run>/combined_eval_{RAG}.json
    <run>/combined_eval_ALL.json
    <run>/combined_eval_meta.json  (fallback)
- Flattens numeric metrics (ints/floats/bools and numeric strings)
- Derives overall support metrics from per-beat rows when available:
    support_overall.sentences_total, supported_total, support_pct_overall
- Writes:
    runs_flat.csv, summary_by_metric.csv, aggregated_summary.json, README.txt
- Optionally prints selected metric averages via --metrics-include
"""

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def find_run_dirs(exp_dir: Path, pattern: str = "run-") -> List[Path]:
    runs = []
    if not exp_dir.exists():
        return runs
    for p in sorted(exp_dir.iterdir()):
        if p.is_dir() and p.name.startswith(pattern):
            runs.append(p)
    return runs

def run_eval_for_run(
    eval_script: Path,
    run_dir: Path,
    rag_type: str,
    support_extra: Optional[str],
    python_bin: str,
    timeout_s: Optional[int] = None,
    read_stdout_json_last_line: bool = False,
) -> Tuple[int, str, Optional[str]]:
    cmd = [
        python_bin,
        str(eval_script),
        "--run-dir", str(run_dir),
        "--rag-type", rag_type,
    ]
    if support_extra:
        cmd += ["--support-extra", support_extra]
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, "", f"Timeout after {timeout_s}s for {run_dir}"
    except Exception as e:
        return 1, "", f"Error running evaluator for {run_dir}: {e}"

def load_eval_json_from_run(run_dir: Path, rag_type: str) -> Optional[Any]:
    cand = [
        run_dir / f"{rag_type}" / "eval_out" / f"combined_eval_{rag_type}.json",
        run_dir / f"{rag_type}" / "eval_out" / "combined_eval_ALL.json",
        run_dir / f"{rag_type}" / "eval_out" / "combined_eval_meta.json",
    ]
    for path in cand:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] Failed to read {path}: {e}")
    return None

def flatten_numeric(prefix: str, obj: Any, out: Dict[str, float]):
    def try_coerce_num(x):
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        if isinstance(x, str):
            sx = x.strip()
            try:
                val = float(sx)
                if math.isnan(val) or math.isinf(val):
                    return None
                return float(val)
            except Exception:
                return None
        return None

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten_numeric(key, v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            flatten_numeric(key, v, out)
    else:
        num = try_coerce_num(obj)
        if num is not None and prefix:
            out[prefix] = num
        return

def to_rows_from_meta(
    meta_obj: Any,
    exp_dir: Path,
    run_dir: Path,
    rag_type_requested: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def merge_meta_convenience(m: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(m)
        try:
            ns = m.get("narrative", {}).get("narrative_summary_csv", [])
            if isinstance(ns, list) and ns:
                first = ns[0]
                if isinstance(first, dict):
                    merged.setdefault("narrative_summary", {})
                    merged["narrative_summary"] = first
        except Exception:
            pass
        try:
            rows_list = m.get("support", {}).get("summary_preview", {}).get("summary_csv_rows", [])
            if isinstance(rows_list, list) and rows_list:
                tot_sent = 0.0
                tot_supp = 0.0
                for row in rows_list:
                    if not isinstance(row, dict):
                        continue
                    s = row.get("sentences", 0)
                    u = row.get("supported_sentences", 0)
                    try:
                        s = float(s)
                    except Exception:
                        s = 0.0
                    try:
                        u = float(u)
                    except Exception:
                        u = 0.0
                    tot_sent += s
                    tot_supp += u
                if tot_sent > 0:
                    merged.setdefault("support_overall", {})
                    merged["support_overall"] = {
                        "sentences_total": tot_sent,
                        "supported_total": tot_supp,
                        "support_pct_overall": (tot_supp / tot_sent) * 100.0,
                    }
        except Exception:
            pass
        return merged

    def one_row(meta: Dict[str, Any], rag_type_for_row: str):
        meta_enriched = merge_meta_convenience(meta)
        flat: Dict[str, float] = {}
        flatten_numeric("", meta_enriched, flat)
        row: Dict[str, Any] = {
            "exp_dir": str(exp_dir),
            "run_dir": str(run_dir),
            "rag_type": rag_type_for_row,
        }
        if isinstance(meta_enriched, dict):
            for k in ("id", "run_id", "uuid", "timestamp"):
                if k in meta_enriched:
                    row[k] = meta_enriched[k]
        for k, v in flat.items():
            row[k] = v
        rows.append(row)

    if isinstance(meta_obj, list):
        for elem in meta_obj:
            rt = rag_type_requested
            if isinstance(elem, dict):
                rt = str(elem.get("rag_type", rag_type_requested))
            one_row(elem if isinstance(elem, dict) else {"value": elem}, rt)
    elif isinstance(meta_obj, dict):
        rt = str(meta_obj.get("rag_type", rag_type_requested))
        one_row(meta_obj, rt)
    else:
        one_row({"value": meta_obj}, rag_type_requested)

    return rows

def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    header_keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header_keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def summarize_metrics(rows: List[Dict[str, Any]], group_key: str = "rag_type") -> List[Dict[str, Any]]:
    if not rows:
        return []
    # discover numeric columns
    numeric_keys = sorted({
        k for row in rows for k, v in row.items() if isinstance(v, (int, float))
    })
    # group
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        g = str(r.get(group_key, "UNKNOWN"))
        groups.setdefault(g, []).append(r)

    def stats_for(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        values_f = [float(v) for v in values if isinstance(v, (int, float))]
        if not values_f:
            return {}
        values_f.sort()
        n = len(values_f)
        mean = sum(values_f) / n
        stdev = statistics.pstdev(values_f) if n > 1 else 0.0
        def q(p):
            idx = (n - 1) * p
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return values_f[lo] * (1 - frac) + values_f[hi] * frac
        return {
            "count": float(n),
            "mean": mean,
            "std": stdev,
            "min": values_f[0],
            "p25": q(0.25),
            "p50": q(0.50),
            "p75": q(0.75),
            "max": values_f[-1],
        }

    out_rows: List[Dict[str, Any]] = []
    for g, g_rows in groups.items():
        for k in numeric_keys:
            vals = [r[k] for r in g_rows if k in r and isinstance(r[k], (int, float))]
            if not vals:
                continue
            st = stats_for(vals)
            out_rows.append({
                "group": g,
                "metric": k,
                **st,
            })
    return out_rows

def main():
    ap = argparse.ArgumentParser(description="Aggregate evaluation across many runs (meta-aware).")
    ap.add_argument("--exp-dirs", nargs="+", required=True,
                    help="One or more experiment directories containing run-* subfolders.")
    ap.add_argument("--rag-type", default="Hybrid",
                    help="RAG type to evaluate (e.g., Hybrid | KG | Graph | ALL).")
    ap.add_argument("--support-extra", default=None,
                    help="Extra args string to pass to evaluator's --support-extra.")
    ap.add_argument("--eval-script", default="run_combined_eval.py",
                    help="Path to run_combined_eval.py")
    ap.add_argument("--python-bin", default=sys.executable,
                    help="Python executable to use.")
    ap.add_argument("--timeout-s", type=int, default=None,
                    help="Per-run timeout in seconds (optional).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write aggregation outputs (default: <first_exp>/_aggregate).")
    ap.add_argument("--read-stdout-json-last-line", action="store_true",
                    help="Parse evaluator's last stdout line as JSON if files are absent.")
    ap.add_argument("--skip-run-if-json-exists", action="store_true",
                    help="Skip calling evaluator if combined_eval JSON already exists.")
    ap.add_argument("--metrics-include", default=None,
                    help="Comma-separated list of metric names to print averages for.")
    args = ap.parse_args()

    eval_script = Path(args.eval_script)
    if not eval_script.exists():
        print(f"[WARN] eval script not found: {eval_script} (will still work with --skip-run-if-json-exists)")

    exp_dirs = [Path(p) for p in args.exp_dirs]
    for d in exp_dirs:
        if not d.exists():
            print(f"[WARN] exp dir not found: {d}")

    out_dir = Path(args.out_dir) if args.out_dir else (exp_dirs[0] / "_aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    run_count = 0
    errors: List[str] = []
    for exp_dir in exp_dirs:
        runs = find_run_dirs(exp_dir)
        if not runs:
            print(f"[WARN] No run-* subfolders in {exp_dir}")
            continue

        for run_dir in runs:
            run_count += 1
            meta_obj = load_eval_json_from_run(run_dir, args.rag_type)
            if args.skip_run_if_json_exists and meta_obj is not None:
                print(f"[SKIP] Found existing combined_eval JSON in {run_dir}")
                rows = to_rows_from_meta(meta_obj, exp_dir, run_dir, args.rag_type)
                all_rows.extend(rows)
                continue

            # Try to run evaluator if present
            if eval_script.exists():
                rc, out, err = run_eval_for_run(
                    eval_script=eval_script,
                    run_dir=run_dir,
                    rag_type=args.rag_type,
                    support_extra=args.support_extra,
                    python_bin=args.python_bin,
                    timeout_s=args.timeout_s,
                    read_stdout_json_last_line=args.read_stdout_json_last_line,
                )
                if rc != 0:
                    msg = f"[ERR] Evaluator failed for {run_dir} (rc={rc}): {err.strip()}"
                    print(msg, file=sys.stderr)
                    errors.append(msg)

            meta_obj = load_eval_json_from_run(run_dir, args.rag_type)
            if meta_obj is None and args.read_stdout_json_last_line:
                lines = [ln for ln in out.strip().splitlines() if ln.strip()]
                if lines:
                    last = lines[-1].strip()
                    try:
                        meta_obj = json.loads(last)
                    except Exception as e:
                        print(f".[WARN] Could not parse last stdout line as JSON for {run_dir}: {e}")

            if meta_obj is None:
                print(f"[WARN] No combined_eval JSON found for {run_dir}")
                continue

            rows = to_rows_from_meta(meta_obj, exp_dir, run_dir, args.rag_type)
            all_rows.extend(rows)

    # Write per-run flat CSV
    runs_flat_csv = out_dir / "runs_flat.csv"
    write_csv(runs_flat_csv, all_rows)
    print(f"[OK] Wrote {runs_flat_csv} with {len(all_rows)} row(s) from {run_count} run(s).")

    # Write summary CSV (per metric per rag_type)
    summary_rows = summarize_metrics(all_rows, group_key="rag_type")
    summary_csv = out_dir / "summary_by_metric.csv"
    write_csv(summary_csv, summary_rows)
    print(f"[OK] Wrote {summary_csv} with {len(summary_rows)} metric summaries.")

    # Optional: print concise averages for selected metrics
    if args.metrics_include:
        wanted = [m.strip() for m in args.metrics_include.split(",") if m.strip()]
        lines = []
        lines.append("Selected metric averages (grouped by rag_type)")
        lines.append("metric, group, count, mean, std, min, p25, p50, p75, max")
        for row in summary_rows:
            if row.get("metric") in wanted:
                lines.append("{metric}, {group}, {count:.0f}, {mean:.6g}, {std:.6g}, {min:.6g}, {p25:.6g}, {p50:.6g}, {p75:.6g}, {max:.6g}".format(**row))
        avg_txt = out_dir / "averages.txt"
        avg_txt.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"[OK] Wrote {avg_txt}")

    # Write JSON bundle
    bundle = {
        "args": vars(args),
        "rows": all_rows,
        "summary": summary_rows,
        "errors": errors,
    }
    bundle_json = out_dir / "aggregated_summary.json"
    bundle_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {bundle_json}")

    # README
    readme = out_dir / "README.txt"
    readme.write_text(
        "Aggregate Evaluation (meta-aware)\n"
        "===============================\n\n"
        f"Command: {' '.join(shlex.quote(a) for a in sys.argv)}\n"
        f"Experiment dirs: {', '.join(str(d) for d in exp_dirs)}\n"
        f"RAG type: {args.rag_type}\n"
        f"Runs discovered: {run_count}\n"
        f"Flat rows: {len(all_rows)}\n"
        f"Errors: {len(errors)}\n"
        "\nOutputs:\n"
        f"- {runs_flat_csv}\n- {summary_csv}\n- {bundle_json}\n",
        encoding="utf-8"
    )
    print(f"[OK] Wrote {readme}")
    print("[DONE]")

if __name__ == "__main__":
    main()
