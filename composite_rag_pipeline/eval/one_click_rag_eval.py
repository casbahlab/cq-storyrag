#!/usr/bin/env python3
"""
one_click_core4.py
One-click: run both aggregation steps in one go:
1) aggregate_eval_runs.py
2) aggregate_core4_from_meta.py

Usage (example):
  python one_click_core4.py \
    --exp-dirs data/Luca-Long-20250909-133910 \
    --rag-type KG \
    --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
    --out-csv data/Luca-Long-20250909-133910/aggregated/core4_KG.csv

By default, this script looks for the two underlying scripts in the same directory.
Override with --scripts-dir if needed.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _quote(cmd):
    return " ".join(shlex.quote(c) for c in cmd)


def run_cmd(cmd):
    print(">>", _quote(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run both aggregation steps sequentially.")
    parser.add_argument(
        "--exp-dirs",
        nargs="+",
        required=True,
        help="One or more experiment directories passed to both scripts.",
    )
    parser.add_argument(
        "--rag-type",
        required=True,
        help="RAG type argument to pass to both scripts (e.g., KG, Hybrid, Graph).",
    )
    parser.add_argument(
        "--support-extra",
        default="",
        help="String of extra flags to pass through to aggregate_eval_runs.py via --support-extra.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path for aggregate_core4_from_meta.py. If empty and a single exp dir is provided, a default is derived.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--scripts-dir",
        default=None,
        help="Directory where aggregate_eval_runs.py and aggregate_core4_from_meta.py live. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="Skip the first step (aggregate_eval_runs.py).",
    )
    parser.add_argument(
        "--only-first",
        action="store_true",
        help="Run only the first step and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )

    args = parser.parse_args()
    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else Path(__file__).resolve().parent

    eval_script = scripts_dir / "aggregate_eval_runs.py"
    core4_script = scripts_dir / "aggregate_core4_from_meta.py"

    if not eval_script.exists():
        print(f"[ERROR] Could not find {eval_script}", file=sys.stderr)
        sys.exit(2)
    if not core4_script.exists():
        print(f"[ERROR] Could not find {core4_script}", file=sys.stderr)
        sys.exit(2)

    # Build common exp-dirs args
    exp_args = []
    for d in args.exp_dirs:
        exp_args += ["--exp-dirs", d]

    # Step 1 command
    cmd1 = [args.python, str(eval_script)] + exp_args + ["--rag-type", args.rag_type]
    if args.support_extra.strip():
        # Pass as a single string value to --support-extra
        cmd1 += ["--support-extra", args.support_extra]

    # Derive default out_csv if not provided and a single exp dir is given
    out_csv = args.out_csv
    if not out_csv:
        if len(args.exp_dirs) == 1:
            exp_dir = Path(args.exp_dirs[0])
            out_csv = str(exp_dir / "aggregated" / f"core4_{args.rag_type}.csv")
        else:
            out_csv = "aggregated_core4.csv"

    # Step 2 command
    cmd2 = [args.python, str(core4_script)] + exp_args + ["--rag-type", args.rag_type, "--out-csv", out_csv]

    print("=== Step 1/2: aggregate_eval_runs ===")
    if args.only_first:
        print("Only-first mode: will run Step 1 and exit.")
    if args.dry_run:
        print(_quote(cmd1))
    else:
        if not args.skip_first:
            run_cmd(cmd1)
        else:
            print("(Skipped Step 1)")

    if args.only_first:
        print("Done (only Step 1).")
        return

    print("\n=== Step 2/2: aggregate_core4_from_meta ===")
    if args.dry_run:
        print(_quote(cmd2))
    else:
        run_cmd(cmd2)

    print("\nAll done.")
    print(f"Expected output CSV: {out_csv}")


if __name__ == "__main__":
    main()
