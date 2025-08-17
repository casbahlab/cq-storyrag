python3 retriever_local_rdflib.py \
  --mode KG \
  --plan ../planner/plan_KG.json \
  --meta ../index/KG/cq_metadata.json \
  --rdf ../data/liveaid_instances_master.ttl \
  --bindings params.json \
  --require_sparql \
  --per_item_sample 5 \
  --timeout_s 10 \
  --log_dir run_trace/logs \
  --errors_jsonl run_trace/retriever.jsonl \
  --evidence_out evidence_KG.jsonl \
  --content_max_chars 240 \
  --out plan_with_evidence_KG.json




python3 retriever_local_rdflib.py --mode Hybrid --plan ../planner/plan_Hybrid.json --meta ../index/Hybrid/cq_metadata.json --rdf ../data/liveaid_instances_master.ttl --bindings params.json --require_sparql --per_item_sample 5 --timeout_s 10 \--hy_enrich_labels --hy_enrich_neighbors 8 --hy_enrich_incoming 4 --enrich_urls --fetch_url_content --url_timeout_s 5 --max_urls_per_item 5 --log_dir run_trace/logs --errors_jsonl run_trace/retriever.jsonl --evidence_out evidence_Hybrid.jsonl --content_max_chars 240 --out plan_with_evidence_Hybrid.json