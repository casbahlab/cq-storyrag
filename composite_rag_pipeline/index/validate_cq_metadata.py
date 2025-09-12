#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def validate(meta_path: Path) -> dict:
    report = {
        "file": str(meta_path),
        "ok": True,
        "errors": [],
        "counts": {"total": 0, "missing_question": 0, "missing_beats": 0, "missing_mode": 0, "missing_sparqlsource": 0}
    }
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"Failed to read/parse metadata JSON: {e}")
        return report

    order = meta.get("order") or []
    m     = meta.get("metadata") or {}
    report["counts"]["total"] = len(order)

    for sid in order:
        rec = m.get(sid) or {}
        q = (rec.get("question") or "").strip()
        beats = rec.get("beats")
        mode = rec.get("retrieval_mode")
        sps = (rec.get("sparql_source") or "").strip()

        if not q:
            report["ok"] = False
            report["counts"]["missing_question"] += 1
            report["errors"].append(f"{sid}: missing question")
        if beats is None or (isinstance(beats, list) and len(beats) == 0):
            report["ok"] = False
            report["counts"]["missing_beats"] += 1
            report["errors"].append(f"{sid}: beats missing or empty")
        if mode is None or (isinstance(mode, list) and len(mode) == 0):
            report["ok"] = False
            report["counts"]["missing_mode"] += 1
            report["errors"].append(f"{sid}: retrieval_mode missing or empty")
        if not sps:
            report["ok"] = False
            report["counts"]["missing_sparqlsource"] += 1
            report["errors"].append(f"{sid}: sparql_source missing")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to cq_metadata.json")
    args = ap.parse_args()
    p = Path(args.meta)
    rep = validate(p)
    out = p.with_name("validation_report.json")
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'OK' if rep['ok'] else 'ISSUES FOUND'} - report: {out}")

if __name__ == "__main__":
    main()
