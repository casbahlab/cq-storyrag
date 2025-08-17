# KG
 python planner_creative.py \
  --index_dir ../index/KG --meta ../index/KG/cq_metadata.json \
  --persona Emma --length Medium --mode KG \
  --items_per_beat 2 --num_beats 5 \
  --llm_provider ollama --llm_model llama3.1-128k:latest \
  --llm_base_url http://localhost:11434 \
  --temperature 0.2 > plan_KG.json


# Hybrid

 python planner_creative.py \
  --index_dir ../index/Hybrid --meta ../index/Hybrid/cq_metadata.json \
  --persona Emma --length Medium --mode Hybrid \
  --items_per_beat 2 --num_beats 5 \
  --llm_provider ollama --llm_model llama3.1-128k:latest \
  --llm_base_url http://localhost:11434 \
  --temperature 0.2 > plan_Hybrid.json



# Both Random

python3 planner_dual_random.py \
  --kg_meta ../index/KG/cq_metadata.json \
  --hy_meta ../index/Hybrid/cq_metadata.json \
  --narrative_plans ../data/narrative_plans.json \
  --persona Emma --length Medium \
  --items_per_beat 2 --seed 42 \
  --match_strategy intersect \
  --out_kg plan_KG.json --out_hybrid plan_Hybrid.json


python3 index_sanity_check.py ../index/KG
python3 index_sanity_check.py ../index/Hybrid