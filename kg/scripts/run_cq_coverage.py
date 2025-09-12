#!/usr/bin/env python3
"""
Run multi-query SPARQL coverage from a single .rq template file that contains many queries.
- Splits on lines that start with "#CQ-ID:" (your convention)
- Ensures each query has PREFIX lines (adds defaults if missing)
- Executes each query with rdflib
- Writes one CSV per CQ and a summary CSV

Usage:
  python scripts/run_cq_coverage_from_template.py \
    --kg kg/liveaid_instances_master.ttl \
    --input cqs/cqs_queries_template_filled_in.rq \
    --out cqs/out
"""
import os, re, csv, argparse
from pathlib import Path
from rdflib import Graph

DEFAULT_PREFIXES = """PREFIX ex: <http://wembrewind.live/ex#>
PREFIX schema: <http://schema.org/>
PREFIX mm: <https://w3id.org/polifonia/ontology/music-meta/>
"""

HDR_RE = re.compile(r"^\s*#CQ-ID:(?P<cq>[A-Za-z0-9\-\_]+)\s*(?P<title>.*)$")

def split_queries(text: str):
    """Yield (cq_id, title, query_text) for each query section."""
    blocks = []
    current = {"cq": None, "title": "", "lines": []}
    for line in text.splitlines():
        m = HDR_RE.match(line)
        if m:
            if current["cq"] and current["lines"]:
                blocks.append((current["cq"], current["title"].strip(), "\n".join(current["lines"]).strip()))
            current = {"cq": m.group("cq"), "title": m.group("title") or "", "lines": []}
        else:
            current["lines"].append(line)
    if current["cq"] and current["lines"]:
        blocks.append((current["cq"], current["title"].strip(), "\n".join(current["lines"]).strip()))
    return blocks

def ensure_prefixes(q: str):
    """If the query already has any PREFIX line, leave it. Otherwise, prepend defaults."""
    if re.search(r"^\s*PREFIX\s+\w+:", q, flags=re.IGNORECASE | re.MULTILINE):
        return q
    return DEFAULT_PREFIXES + "\n" + q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True, help="Path to TTL with the graph (e.g., kg/liveaid_instances_master.ttl)")
    ap.add_argument("--input", required=True, help="Path to the filled .rq file with many queries")
    ap.add_argument("--out", required=True, help="Output folder for CSVs")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    g = Graph().parse(args.kg, format="turtle")
    text = Path(args.input).read_text(encoding="utf-8")
    blocks = split_queries(text)

    summary_rows = []
    for cq_id, title, q in blocks:
        q_fixed = ensure_prefixes(q)
        # quick sanity check: warn if placeholders still present
        placeholders = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", q_fixed)
        if placeholders:
            print(f"[WARN] {cq_id} still has placeholders: {set(placeholders)}")

        out_csv = Path(args.out) / f"{cq_id}.csv"
        try:
            res = g.query(q_fixed)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(res.vars)
                for row in res:
                    w.writerow([str(row.get(var)) for var in res.vars])
            count = len(res)
            ok = True
            err = ""
        except Exception as e:
            count = 0
            ok = False
            err = str(e)
            out_csv.write_text("", encoding="utf-8")

        print(f"[{cq_id}] rows={count}{'' if ok else '  ERROR: '+err}")
        summary_rows.append([cq_id, title, count, "OK" if ok else f"ERROR: {err}"])

    # Write summary
    summary_csv = Path(args.out) / "coverage_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["CQ-ID", "Title", "Rows", "Status"])
        w.writerows(summary_rows)

    print(f"\nSummary → {summary_csv}")

if __name__ == "__main__":
    main()
