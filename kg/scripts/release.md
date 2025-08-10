# Live Aid Knowledge Graph – Quick Run Guide

## Prerequisites
- Python 3.9+
- All TTL instance files, mappings, and schema files in the `kg/` folder
- Required ontology files:
  - `schemaorg.ttl`
  - `musicmeta.owl`
  - `liveaid_schema.ttl`

## One-Click Release
To merge all TTL files, validate schema usage, and run Competency Question coverage in one step:

```bash
# 1) Merge modules → master
python kg/scripts/merge_all_ttls.py

# 2) Run strict vocab validation (fails if unknown terms exist)
python kg/scripts/release_one_click.py --label "Post-schema-lock baseline"

# (Optional) If you just want CQ coverage against the working master:
python kg/scripts/run_cq_coverage.py \
  --kg kg/liveaid_instances_master.ttl \
  --input kg/cqs/cqs_queries_template_filled_in.rq \
  --out kg/coverage_summary.csv
