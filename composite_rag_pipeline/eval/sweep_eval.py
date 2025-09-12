#!/usr/bin/env python3
# sweep_eval.py — grid search over support_ctx_reset.py settings
from __future__ import annotations
import argparse, subprocess, sys, re, os, json, math, time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from datetime import datetime

SUPPORT_RX = re.compile(
    r"support:\s*(\d+)\s*/\s*(\d+)\s*\(([\d\.]+)%\).*?\(canon=(on|off),\s*bm25=([a-zA-Z]+)\)",
    re.I,
)

def _parse_keyval_list(items: List[str]) -> Dict[str, str]:
    """Parse ['KG=/path/..', 'Graph=/path/..'] -> dict."""
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Expected NAME=VALUE, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def _parse_float_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_str_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _safe_float(x: Any, default=float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def run_one(cmd: List[str]) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Execute support_ctx_reset.py once.
    Returns: (ok, support_pct, extras_dict)
    extras contains: sentences_supported, sentences_total, canon (bool), bm25_mode (str),
                     duration_sec (float), stdout_path (optional)
    """
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return False, float("nan"), {"error": f"spawn failed: {e}", "duration_sec": time.monotonic() - t0}

    dt = time.monotonic() - t0
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    ok = (proc.returncode == 0)

    # Parse the support line
    sup_pct = float("nan")
    sup = tot = None
    canon_flag = None
    bm25_mode = None

    # last matching line wins
    for m in SUPPORT_RX.finditer(out):
        sup = int(m.group(1))
        tot = int(m.group(2))
        sup_pct = float(m.group(3))
        canon_flag = (m.group(4).lower() == "on")
        bm25_mode = m.group(5).lower()

    extras: Dict[str, Any] = {
        "sentences_supported": sup if sup is not None else np.nan,
        "sentences_total": tot if tot is not None else np.nan,
        "canon": bool(canon_flag) if canon_flag is not None else np.nan,
        "bm25_mode": bm25_mode if bm25_mode is not None else "",
        "duration_sec": dt,
        "stdout": out,
        "returncode": proc.returncode,
    }

    # OK only if returncode==0 and we parsed a percentage
    ok = ok and not math.isnan(sup_pct)
    return ok, sup_pct, extras

def main():
    ap = argparse.ArgumentParser(description="Parameter sweep for support_ctx_reset.py")
    ap.add_argument("--dataset", action="append", required=True,
                    help="Repeatable. NAME=/path/to/answers.jsonl (e.g., KG=... Hybrid=... Graph=...)")
    ap.add_argument("--tf-grid", required=True, help="Comma list of TF thresholds, e.g. 0.31,0.33,0.35")
    ap.add_argument("--cj-grid", required=True, help="Comma list of char-3 thresholds, e.g. 0.26,0.27,0.28")
    ap.add_argument("--cleaning", default="light",
                    help="Comma list from {raw,light}. 'light' = --light-clean, 'raw' = no cleaning flag.")
    ap.add_argument("--canon", choices=["on","off","both"], default="on",
                    help="Canonicalization: on/off/both.")
    ap.add_argument("--bm25-modes", default="off",
                    help="Comma list from {off,filter}.")
    ap.add_argument("--bm25-k1", type=float, default=1.2)
    ap.add_argument("--bm25-b", default="0.25,0.50",
                    help="Comma list of BM25 b values.")
    ap.add_argument("--bm25-topk", default="40,60",
                    help="Comma list of BM25 topk values.")
    ap.add_argument("--alias-by-dataset", action="append",
                    help="Repeatable. NAME=alias.json (applied only when canon=on).")
    ap.add_argument("--outdir", default=None,
                    help="Base output directory for sweep results (default: context/sweeps/YYYYMMDD)")
    args = ap.parse_args()

    datasets = _parse_keyval_list(args.dataset)
    alias_map = _parse_keyval_list(args.alias_by_dataset or [])

    tf_grid = _parse_float_csv(args.tf_grid)
    cj_grid = _parse_float_csv(args.cj_grid)
    cleaning_opts = _parse_str_csv(args.cleaning)
    bm25_modes = _parse_str_csv(args.bm25_modes)
    bm25_b_vals = [_safe_float(x) for x in _parse_str_csv(args.bm25_b)]
    bm25_topk_vals = [int(float(x)) for x in _parse_str_csv(args.bm25_topk)]

    canon_opts = ["on", "off"] if args.canon == "both" else [args.canon]

    date_tag = datetime.now().strftime("%Y%m%d")
    base_out = Path(args.outdir or f"context/sweeps/{date_tag}")
    base_out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for ds_name, answers_path in datasets.items():
        answers_path = str(answers_path)
        for tf in tf_grid:
            for cj in cj_grid:
                for clean_mode in cleaning_opts:
                    for canon_mode in canon_opts:
                        canon_on = (canon_mode == "on")
                        for bm25 in bm25_modes:
                            if bm25 not in {"off","filter"}:
                                continue
                            b_vals = bm25_b_vals if bm25 == "filter" else [np.nan]
                            topk_vals = bm25_topk_vals if bm25 == "filter" else [np.nan]
                            for b in b_vals:
                                for topk in topk_vals:
                                    # Build output directory & file names (just for organization)
                                    slug = [
                                        f"{ds_name}",
                                        f"tf{tf:.2f}",
                                        f"cj{cj:.2f}",
                                        clean_mode,
                                        f"canon{'On' if canon_on else 'Off'}",
                                        f"bm25{bm25}",
                                    ]
                                    if bm25 == "filter":
                                        slug += [f"b{b:.2f}", f"top{int(topk)}"]
                                    run_dir = base_out / "_".join(slug)
                                    run_dir.mkdir(parents=True, exist_ok=True)
                                    out_csv = run_dir / "support_sentences.csv"
                                    out_summary = run_dir / "support_summary.csv"

                                    cmd = [
                                        sys.executable, "support_ctx_reset.py",
                                        "--answers", answers_path,
                                        "--out-csv", str(out_csv),
                                        "--out-summary", str(out_summary),
                                        "--tf-th", f"{tf:.2f}",
                                        "--cj-th", f"{cj:.2f}",
                                    ]
                                    # cleaning flag
                                    if clean_mode == "light":
                                        cmd += ["--light-clean"]
                                    elif clean_mode == "raw":
                                        pass
                                    else:
                                        print(f"[WARN] unknown cleaning={clean_mode}, skipping")
                                        continue
                                    # canon flag
                                    if not canon_on:
                                        cmd += ["--no-canon"]
                                    else:
                                        # alias by dataset (only when canon ON)
                                        alias_file = alias_map.get(ds_name)
                                        if alias_file:
                                            cmd += ["--alias-file", alias_file]

                                    # bm25 flags
                                    if bm25 == "filter":
                                        cmd += [
                                            "--bm25-mode", "filter",
                                            "--bm25-k1", str(args.bm25_k1),
                                            "--bm25-b",  f"{float(b):.2f}",
                                            "--bm25-topk", str(int(topk)),
                                        ]
                                    else:
                                        cmd += ["--bm25-mode", "off"]

                                    ok, pct, extras = run_one(cmd)
                                    row = {
                                        "dataset": ds_name,
                                        "answers_path": answers_path,
                                        "tf_th": tf,
                                        "cj_th": cj,
                                        "cleaning": clean_mode,
                                        "canon": bool(canon_on),
                                        "bm25_mode": bm25,
                                        "bm25_k1": args.bm25_k1,
                                        "bm25_b": float(b) if not math.isnan(b) else np.nan,
                                        "bm25_topk": int(topk) if not math.isnan(topk) else np.nan,
                                        "support_pct": pct,
                                        "ok": bool(ok),
                                        "duration_sec": _safe_float(extras.get("duration_sec")),
                                        "sentences_supported": extras.get("sentences_supported"),
                                        "sentences_total": extras.get("sentences_total"),
                                    }
                                    rows.append(row)

                                    # log line
                                    t_part = f" (t={row['duration_sec']:.2f}s)" if not math.isnan(row['duration_sec']) else ""
                                    print(
                                        f"[SWEEP] {ds_name} tf={tf:.2f} cj={cj:.2f} "
                                        f"clean={clean_mode} canon={'on' if canon_on else 'off'}, "
                                        f"bm25={bm25} k1={args.bm25_k1} "
                                        f"b={row['bm25_b'] if not math.isnan(row['bm25_b']) else '-'} "
                                        f"topk={row['bm25_topk'] if not math.isnan(row['bm25_topk']) else '-'} "
                                        f"→ {row['support_pct']:.1f}% {'OK' if ok else 'ERR'}{t_part}"
                                    )

    # Save results
    df = pd.DataFrame(rows)
    results_csv = base_out / "sweep_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"\n[SWEEP] wrote {results_csv}")

    # Best per dataset
    # Be tolerant if duration_sec is NaN: sort by support_pct desc only in that case
    best_rows: List[pd.DataFrame] = []
    for ds, g in df.groupby("dataset"):
        sort_cols = ["support_pct"]
        ascending = [False]
        if "duration_sec" in g.columns:
            sort_cols.append("duration_sec")
            ascending.append(True)
        best_rows.append(g.sort_values(sort_cols, ascending=ascending).head(1))
    best = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    best_csv = base_out / "best_by_dataset.csv"
    best.to_csv(best_csv, index=False)
    print(f"[SWEEP] wrote {best_csv}")

    # Simple markdown report
    rpt = ["# Sweep Report", "", f"Date: {datetime.now().isoformat(timespec='seconds')}  ", ""]
    for _, r in best.iterrows():
        rpt.append(
            f"- **{r['dataset']}** → {r['support_pct']:.1f}% "
            f"(tf={r['tf_th']:.2f}, cj={r['cj_th']:.2f}, clean={r['cleaning']}, "
            f"canon={'on' if r['canon'] else 'off'}, bm25={r['bm25_mode']}, "
            f"k1={r['bm25_k1']}, b={r['bm25_b'] if not np.isnan(r['bm25_b']) else '-'}, "
            f"topk={int(r['bm25_topk']) if not np.isnan(r['bm25_topk']) else '-'})"
        )
    report_md = base_out / "REPORT.md"
    report_md.write_text("\n".join(rpt) + "\n", encoding="utf-8")
    print(f"[SWEEP] wrote {report_md}")

if __name__ == "__main__":
    main()
