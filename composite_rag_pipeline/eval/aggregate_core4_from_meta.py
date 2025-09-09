
#!/usr/bin/env python3
import argparse, json, math, csv, sys
from pathlib import Path
from typing import Any, Dict, List

def coerce_float(x) -> float:
    try:
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        return float(str(x).strip())
    except Exception:
        return math.nan

def find_meta_files(exp_dirs: List[Path]) -> List[Path]:
    metas = []
    for root in exp_dirs:
        for p in root.rglob("combined_eval_meta.json"):
            metas.append(p)
    metas = sorted(set(metas), key=lambda p: str(p))
    return metas

def aggregate_core4(exp_dirs: List[Path], rag_type: str) -> Dict[str, Any]:
    files = find_meta_files(exp_dirs)
    support_vals: List[float] = []
    coverage_vals: List[float] = []
    flesch_vals: List[float] = []
    cohesion_vals: List[float] = []
    local_cohesion_vals: List[float] = []

    runs_seen = 0
    beats_support = 0
    beats_coverage = 0

    for f in files:
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        rt = str(meta.get("rag_type", "")).strip() or None
        if rag_type and rt and rt != rag_type:
            continue

        try:
            s_rows = meta.get("support", {}).get("support_preview", {}).get("summary_csv_rows", [])
            for r in s_rows:
                v = coerce_float(r.get("support_pct"))
                if not math.isnan(v):
                    support_vals.append(v)
                    beats_support += 1
        except Exception:
            pass

        try:
            c_rows = meta.get("support", {}).get("coverage_preview", {}).get("summary_csv_rows", [])
            for r in c_rows:
                v = coerce_float(r.get("coverage_pct"))
                if not math.isnan(v):
                    coverage_vals.append(v)
                    beats_coverage += 1
        except Exception:
            pass

        try:
            ns = meta.get("narrative", {}).get("narrative_summary_csv", [])
            if isinstance(ns, list) and ns:
                first = ns[0]
                fr = coerce_float(first.get("flesch_reading_ease"))
                cg = coerce_float(first.get("beat_to_global_sim_mean"))
                adj_sim = coerce_float(first.get("adj_sim_mean"))


                if not math.isnan(fr):
                    flesch_vals.append(fr)
                if not math.isnan(cg):
                    cohesion_vals.append(cg)
                if not math.isnan(adj_sim):
                    local_cohesion_vals.append(adj_sim)
        except Exception:
            pass

        runs_seen += 1

    def mean(vals: List[float]) -> float:
        vals2 = [v for v in vals if not math.isnan(v)]
        return sum(vals2)/len(vals2) if vals2 else math.nan

    return {
        "rag_type": rag_type or "ALL",
        "runs_count": runs_seen,
        "beats_support_count": beats_support,
        "beats_coverage_count": beats_coverage,
        "support_pct_mean": mean(support_vals),
        "coverage_pct_mean": mean(coverage_vals),
        "flesch_reading_ease_mean": mean(flesch_vals),
        "global_cohesion_mean": mean(cohesion_vals),
        "local_cohesion_mean": mean(local_cohesion_vals),
    }

def main():
    ap = argparse.ArgumentParser(description="Aggregate core-4 metrics from combined_eval_meta.json across runs.")
    ap.add_argument("--exp-dirs", nargs="+", required=True, help="Experiment dir(s) containing run-* subfolders.")
    ap.add_argument("--rag-type", default="", help="Filter by rag type (e.g., KG, Hybrid). Leave empty for all.")
    ap.add_argument("--out-csv", default="", help="Optional: write a one-row CSV here.")
    args = ap.parse_args()

    exp_dirs = [Path(p) for p in args.exp_dirs]
    for d in exp_dirs:
        if not d.exists():
            print(f"[WARN] Not found: {d}", file=sys.stderr)

    out = aggregate_core4(exp_dirs, args.rag_type.strip())

    def fmt(x):
        if isinstance(x, float) and not math.isnan(x):
            return f"{x:.4f}"
        return str(x)

    print("rag_type,runs,beats_support,beats_coverage,support_pct_mean,coverage_pct_mean,flesch_reading_ease_mean,global_cohesion_mean,local_cohesion_mean")
    print(",".join([
        out["rag_type"],
        str(out["runs_count"]),
        str(out["beats_support_count"]),
        str(out["beats_coverage_count"]),
        fmt(out["support_pct_mean"]),
        fmt(out["coverage_pct_mean"]),
        fmt(out["flesch_reading_ease_mean"]),
        fmt(out["global_cohesion_mean"]),
        fmt(out["local_cohesion_mean"]),
    ]))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rag_type","runs","beats_support","beats_coverage",
                        "support_pct_mean","coverage_pct_mean",
                        "flesch_reading_ease_mean","global_cohesion_mean", "local_cohesion_mean"])
            w.writerow([
                out["rag_type"],
                out["runs_count"],
                out["beats_support_count"],
                out["beats_coverage_count"],
                out["support_pct_mean"],
                out["coverage_pct_mean"],
                out["flesch_reading_ease_mean"],
                out["global_cohesion_mean"],
                out["local_cohesion_mean"],
            ])
        print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
