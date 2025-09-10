python3 pipeline_programmatic.py \
  --kg_meta index/KG/cq_metadata.json \
  --hy_meta index/Hybrid/cq_metadata.json \
  --narrative_plans data/narrative_plans.json \
  --rdf data/liveaid_instances_master.ttl \
  --persona Emma --length Medium --items_per_beat 2 --seed 42 \
  --use_external_planner \
  --planner_path planner/planner_dual_random.py \
  --planner_match_strategy intersect \
  --retriever_params_json '{"event":"ex:LiveAid1985","musicgroup":"ex:Queen","singleartist":"ex:Madonna","bandmember":"ex:BrianMay","venue":"ex:WembleyStadium","venue2":"ex:JFKStadium"}' \
  --generator_params_json '{"Event":"Live Aid 1985","MusicGroup":"Queen","SingleArtist":"Madonna","BandMember":"Brian May","Venue":"Wembley Stadium","Venue2":"JFK Stadium"}' \
  --llm_provider gemini \
  --llm_model gemini-2.5-flash \
  --run_root runs --persist_params




python3 pipeline_programmatic.py \
  --kg_meta index/KG/cq_metadata.json \
  --hy_meta index/Hybrid/cq_metadata.json \
  --narrative_plans data/narrative_plans.json \
  --rdf data/liveaid_instances_master.ttl \
  --persona Emma --length Medium --items_per_beat 2 --seed 42 \
  --use_external_planner \
  --planner_path planner/planner_dual_random.py \
  --planner_match_strategy intersect \
  --retriever_params_json '{"event":"ex:LiveAid1985","musicgroup":"ex:Queen","singleartist":"ex:Madonna","bandmember":"ex:BrianMay","venue":"ex:WembleyStadium","venue2":"ex:JFKStadium"}' \
  --generator_params_json '{"Event":"Live Aid 1985","MusicGroup":"Queen","SingleArtist":"Madonna","BandMember":"Brian May","Venue":"Wembley Stadium","Venue2":"JFK Stadium"}'
  --run_root runs --persist_params



python eval/eval_single_run.py \
  --run-dir runs/Emma-Medium-20250820-223705/KG/run-01 \   
  --persona Emma \
  --length Medium

python run_narrative_grid.py --personas Emma --lengths Medium --patterns KG Hybrid Graph  --runs 1


python run_narrative_grid.py --personas Luca --lengths Long --patterns KG Hybrid Graph  --runs 1
