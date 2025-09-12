#!/usr/bin/env python3
"""Run SHACL validation using pySHACL.

Install: pip install pyshacl rdflib
Usage:
  python kg/scripts/run_shacl.py --data kg/liveaid_instances_master.ttl \
                                 --shapes kg/schema/91_shacl_data_quality.ttl \
                                 --report kg/validation/shacl_report.txt
"""
import argparse
from pathlib import Path
from rdflib import Graph
from pyshacl import validate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data TTL")
    ap.add_argument("--shapes", required=True, nargs="+", help="One or more SHACL shape files")
    ap.add_argument("--report", required=True, help="Path to write text report")
    args = ap.parse_args()

    data_g = Graph()
    data_g.parse(args.data, format="turtle")

    shapes_g = Graph()
    for s in args.shapes:
        shapes_g.parse(s, format="turtle")

    conforms, results_graph, results_text = validate(
        data_g, shacl_graph=shapes_g, inference="rdfs", debug=False
    )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(results_text, encoding="utf-8")
    print(results_text)
    if not conforms:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
