python3 generator_dual.py --plan ../planner/plan_Hybrid.json --plan_with_evidence ../retriever/plan_with_evidence_Hybrid.json --hy_meta ../index/Hybrid/cq_metadata.json --llm_provider gemini --llm_model gemini-2.5-flash --use_url_content --max_url_snippets 2 --snippet_chars 500 --out answers_Hybrid.jsonl --story_out story_Hybrid.md --include_citations



python3 generator_dual.py --plan ../planner/plan_KG.json --plan_with_evidence ../retriever/plan_with_evidence_KG.json --hy_meta ../index/KG/cq_metadata.json --llm_provider gemini --llm_model gemini-2.5-flash --use_url_content --max_url_snippets 2 --snippet_chars 500 --out answers_KG.jsonl --story_out story_KG.md --include_citations



python3 generator_dual.py \
  --plan ../planner/plan_Hybrid.json \
  --plan_with_evidence ../retriever/plan_with_evidence_Hybrid.json \
  --hy_meta ../index/Hybrid/cq_metadata.json \
  --params params.json \
  --llm_provider ollama --llm_model llama3.1-128k \
  --use_url_content --max_url_snippets 2 --snippet_chars 500 \
  --out answers_Hybrid.jsonl \
  --story_out story_Hybrid.md --include_citations


python3 generator_dual.py \
  --plan ../planner/plan_KG.json \
  --plan_with_evidence ../retriever/plan_with_evidence_KG.json \
  --kg_meta ../index/KG/cq_metadata.json \
  --params params.json \
  --llm_provider ollama --llm_model llama3.1-128k \
  --out answers_KG.jsonl \
  --story_out story_KG.md --include_citations
