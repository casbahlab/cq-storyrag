python3 support_ctx_pipeline.py eval \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/lexical/support_from_ctx_KG_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/lexical/support_from_ctx_KG_summary.csv \
  --mode lexical

python3 support_ctx_pipeline.py eval \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/nli/support_from_ctx_KG_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/nli/support_from_ctx_KG_summary.csv \
  --mode nli --nli-model roberta-large-mnli


python3 support_ctx_pipeline.py eval \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/nli/support_from_ctx_KG_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/nli/support_from_ctx_KG_summary.csv \
  --mode nli --nli-model roberta-large-mnli --topk 50 --ent-th 0.62 --con-th 0.35


python3 support_ctx_pipeline.py eval \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/nli/support_from_ctx_KG_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/nli/support_from_ctx_KG_summary.csv \
  --mode nli --nli-model roberta-large-mnli \
  --ent-th 0.62 --con-th 0.35 \
  --max-repairs-per-beat 2

CLEAN_CONTEXT=1 NLI_GATE_TFIDF=0.20 NLI_GATE_CHAR3=0.18 \
python3 support_ctx_pipeline.py eval \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/nli_clean/support_from_ctx_KG_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/nli_clean/support_from_ctx_KG_summary.csv \
  --mode nli --nli-model roberta-large-mnli \
  --clean-context --topk 50 --ent-th 0.62 --con-th 0.35


CLEAN_CONTEXT=1 NLI_GATE_TFIDF=0.18 NLI_GATE_CHAR3=0.16 \
python3 support_ctx_pipeline.py pipeline \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --workdir context/Emma-Medium-20250825-223330/pipeline_nli_clean_fix \
  --mode nli --nli-model roberta-large-mnli \
  --clean-context --topk 80 --ent-th 0.62 --con-th 0.35 \
  --max-repairs-per-beat 2


CLEAN_CONTEXT=1 NLI_GATE_TFIDF=0.18 NLI_GATE_CHAR3=0.16 \
python3 support_ctx_pipeline.py eval \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/nli_merge/support_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/nli_merge/support_summary.csv \
  --mode nli --nli-model roberta-large-mnli \
  --topk 100 --ent-th 0.62 --con-th 0.35


python3 support_ctx_pipeline.py fix \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-fixed-answers context/Emma-Medium-20250825-223330/fix/answers_KG_fixed.jsonl \
  --out-fixed-story context/Emma-Medium-20250825-223330/fix/story_KG_fixed.md \
  --out-patches-csv context/Emma-Medium-20250825-223330/fix/patches_KG.csv \
  --max-repairs-per-beat 2







/Users/sowjanyab/code/dissertation/comp70225-wembrewind/composite_rag_pipeline/eval/data/Emma-Medium-20250826-093041/run-01/Hybrid


python3 support_ctx_reset.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv context/Emma-Medium-20250825-223330/kg_det/support_sentences.csv \
  --out-summary context/Emma-Medium-20250825-223330/kg_det/support_summary.csv \
  --tf-th 0.31 --cj-th 0.26 --emit-near --near-low 0.26 --near-high 0.31 --emit-coverage
ans_path : data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl
[DET] wrote context/Emma-Medium-20250825-223330/kg_det/near_misses.csv (near-miss window [0.26, 0.31))
[DET] wrote context/Emma-Medium-20250825-223330/kg_det/support_sentences.csv
[DET] wrote context/Emma-Medium-20250825-223330/kg_det/support_summary.csv
[DET] wrote context/Emma-Medium-20250825-223330/kg_det/coverage_evidence.csv
[DET] wrote context/Emma-Medium-20250825-223330/kg_det/coverage_summary.csv
[DET] support: 34/56 (60.7%)  coverage: 29/32 (90.6%)  (canon=on)


(wemb) sowjanyab@Sowjanyas-MacBook-Pro eval % python3 support_ctx_reset.py \
  --answers data/Emma-Medium-20250826-093041/run-01/Hybrid/answers_Hybrid.jsonl \
  --out-csv context/Emma-Medium-20250826-093041/hybrid_det/support_sentences.csv \
  --out-summary context/Emma-Medium-20250826-093041/hybrid_det/support_summary.csv \
  --tf-th 0.31 --cj-th 0.26 --emit-near --near-low 0.26 --near-high 0.31 --emit-coverage
ans_path : data/Emma-Medium-20250826-093041/run-01/Hybrid/answers_Hybrid.jsonl
[DET] wrote context/Emma-Medium-20250826-093041/hybrid_det/near_misses.csv (near-miss window [0.26, 0.31))
[DET] wrote context/Emma-Medium-20250826-093041/hybrid_det/support_sentences.csv
[DET] wrote context/Emma-Medium-20250826-093041/hybrid_det/support_summary.csv
[DET] wrote context/Emma-Medium-20250826-093041/hybrid_det/coverage_evidence.csv
[DET] wrote context/Emma-Medium-20250826-093041/hybrid_det/coverage_summary.csv
[DET] support: 37/62 (59.7%)  coverage: 40/42 (95.2%)  (canon=on)


python3 support_ctx_reset.py \
  --answers data/Emma-Medium-20250826-201608/run-01/Graph/answers_Graph.jsonl \
  --out-csv context/Emma-Medium-20250826-201608/graph_det/support_sentences.csv \
  --out-summary context/Emma-Medium-20250826-201608/graph_det/support_summary.csv \
  --tf-th 0.31 --cj-th 0.26 --emit-near --near-low 0.26 --near-high 0.31 --emit-coverage

ans_path : data/Emma-Medium-20250826-201608/run-01/Graph/answers_Graph.jsonl
[DET] wrote context/Emma-Medium-20250826-201608/graph_det/near_misses.csv (near-miss window [0.26, 0.31))
[DET] wrote context/Emma-Medium-20250826-201608/graph_det/support_sentences.csv
[DET] wrote context/Emma-Medium-20250826-201608/graph_det/support_summary.csv
[DET] wrote context/Emma-Medium-20250826-201608/graph_det/coverage_evidence.csv
[DET] wrote context/Emma-Medium-20250826-201608/graph_det/coverage_summary.csv
[DET] support: 7/32 (21.9%)  coverage: 2/100 (2.0%)  (canon=on)
(wemb) sowjanyab@Sowjanyas-MacBook-Pro eval % 


python3 sweep_eval.py \                                                                                                                
  --support-script composite_rag_pipeline/eval/support_ctx_reset.py \       
  --workdir context/sweeps/20250826 \                                                   
  --dataset KG=data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \         
  --dataset Hybrid=data/Emma-Medium-20250826-093041/run-01/Hybrid/answers_Hybrid.jsonl \
  --dataset Graph=data/Emma-Medium-20250826-201608/run-01/Graph/answers_Graph.jsonl \
  --tf-grid 0.31,0.32,0.33,0.34,0.35 \
  --cj-grid 0.26,0.27,0.28 \
  --cleaning raw,light \   
  --canon both \ 
  --bm25-modes off,filter \
  --bm25-k1 1.2 \    
  --bm25-b 0.25,0.50 \               
  --bm25-topk 40,60 \
  --alias-by-dataset Graph=alias.json
  

python3 sweep_eval.py \
  --dataset KG=data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --dataset Hybrid=data/Emma-Medium-20250826-093041/run-01/Hybrid/answers_Hybrid.jsonl \
  --dataset Graph=data/Emma-Medium-20250826-201608/run-01/Graph/answers_Graph.jsonl \
  --tf-grid 0.31,0.32,0.33,0.34,0.35 \
  --cj-grid 0.26,0.27,0.28 \
  --cleaning raw,light \
  --canon both \
  --bm25-modes off,filter \
  --bm25-k1 1.2 \
  --bm25-b 0.25,0.50 \
  --bm25-topk 40,60 \
  --alias-by-dataset Graph=alias.json



python3 narrative_eval.py \
  -i data/Emma-Medium-20250825-223330/run-01/KG/story_KG.md \
  -o out_dir \
  --beats auto \
  --neardupe-th 0.85 \
  --domain-stopwords live aid concert performance wembley philadelphia \
  --with-coherence





# All three on
python support_ctx_reset_refactored.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl --out-csv out/support.csv --out-summary out/summary.csv \
  --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-k 8 --topic-th 0.30

# Only topic modelling (no lexical signals)
python support_ctx_reset_refactored.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl --out-csv out/support.csv --out-summary out/summary.csv \
  --use-tfidf 0 --use-char3 0 --use-topic 1

# Lexical only (your current default)
python support_ctx_reset_refactored.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl --out-csv out/support.csv --out-summary out/summary.csv \
  --use-tfidf 1 --use-char3 1 --use-topic 0

# Cosine only 
python support_ctx_reset_refactored.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl --out-csv out/support.csv --out-summary out/summary.csv \
  --use-tfidf 1 --use-char3 0 --use-topic 0


# Topic modeling only 
python support_ctx_reset_refactored.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl --out-csv out/support.csv --out-summary out/summary.csv \
  --use-tfidf 0 --use-char3 0 --use-topic 1



python support_ctx_reset_refactored.py \
  --answers data/Emma-Medium-20250825-223330/run-01/KG/answers_KG.jsonl \
  --out-csv out/support.csv --out-summary out/summary.csv \
  --use-tfidf 1 --use-char3 1 --use-topic 1 \
  --fusion rrf --rrf-k 60 \
  --tf-th 0.33 --cj-th 0.27 --topic-th 0.30 \
  --topic-k 8




 python narrative_eval.py \
  -i data/Emma-Medium-20250901-215830/run-01/KG/story_KG.md \
  -o narr_eval_out \
  --beats auto \
  --with-coherence \
  --annotate \
  --nqi-lite \
  --neardupe-th 0.85 \
  --domain-stopwords live aid concert wembley philadelphia


python run_combined_eval.py \
  --run-dir data/Emma-Medium-20250902-223253/run-01 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"




python aggregate_eval_runs.py \
  --exp-dirs data/Emma-Medium-20250904-154423 \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"





python aggregate_eval_runs.py \
  --exp-dirs data/Emma-Medium-20250908-235849 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"


python aggregate_core4_from_meta.py \
  --exp-dirs data/Emma-Medium-20250908-235849 \
  --rag-type KG \
  --out-csv data/Emma-Medium-20250908-235849/aggregated/core4_KG.csv



Emma-Medium-20250909-130026

python aggregate_eval_runs.py \
  --exp-dirs data/Emma-Medium-20250909-130026 \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"



python aggregate_core4_from_meta.py \
  --exp-dirs data/Emma-Medium-20250909-130026 \
  --rag-type Hybrid \
  --out-csv data/Emma-Medium-20250909-130026/aggregated/core4_Hybrid.csv

Emma-Medium-20250909-131038

python aggregate_eval_runs.py \
  --exp-dirs data/Emma-Medium-20250909-131038 \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"


python aggregate_core4_from_meta.py \
  --exp-dirs data/Emma-Medium-20250909-131038 \
  --rag-type Graph \
  --out-csv data/Emma-Medium-20250909-131038/aggregated/core4_Graph.csv



Emma-Long-20250909-135054

python aggregate_eval_runs.py \
  --exp-dirs data/Emma-Long-20250909-135054 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"


python aggregate_core4_from_meta.py \
  --exp-dirs data/Emma-Long-20250909-135054 \
  --rag-type KG \
  --out-csv data/Emma-Long-20250909-135054/aggregated/core4_KG.csv



Luca-Long-20250909-133910


python aggregate_eval_runs.py \
  --exp-dirs data/Luca-Long-20250909-133910 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage"

python aggregate_core4_from_meta.py \
  --exp-dirs data/Luca-Long-20250909-133910 \
  --rag-type KG \
  --out-csv data/Luca-Long-20250909-133910/aggregated/core4_KG.csv


python one_click_rag_eval.py \
  --exp-dirs data/Luca-Long-20250909-133910 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Long-20250909-133910/aggregated/core4_KG.csv



Emma-Medium-20250911-154543 --- KG - 20 

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Medium-20250911-154543 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Medium-20250911-154543/aggregated/aggregated_KG.csv


Emma-Medium-20250911-164455 --- Graph - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Medium-20250911-164455 \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Medium-20250911-164455/aggregated/aggregated_Graph.csv


Emma-Medium-20250911-180309 --- Hybrid - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Medium-20250911-180309 \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Medium-20250911-180309/aggregated/aggregated_Hybrid.csv


Emma-Small-20250911-183109 --- Hybrid - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Small-20250911-183109 \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Small-20250911-183109/aggregated/aggregated_Hybrid.csv

Emma-Small-20250911-202907 --- KG - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Small-20250911-202907 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Small-20250911-202907/aggregated/aggregated_KG.csv


Emma-Small-20250911-212937 --- Graph - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Small-20250911-212937 \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Small-20250911-212937/aggregated/aggregated_Graph.csv


Emma-Long-20250911-220017 --- Graph - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Long-20250911-220017 \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Long-20250911-220017/aggregated/aggregated_Graph.csv

Emma-Long-20250911-222226 --- KG - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Long-20250911-222226 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Long-20250911-222226/aggregated/aggregated_KG.csv


Emma-Long-20250911-232806 --- Hybrid - 20

python one_click_rag_eval.py \
  --exp-dirs data/Emma-Long-20250911-232806 \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Emma-Long-20250911-232806/aggregated/aggregated_Hybrid.csv


Luca-Small-20250912-073410 --- KG - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Small-20250912-073410 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Small-20250912-073410/aggregated/aggregated_KG.csv

Luca-Small-20250912-073410 --- Hybrid - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Small-20250912-073410_Hybrid \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Small-20250912-073410_Hybrid/aggregated/aggregated_Hybrid.csv

Luca-Small-20250912-073410 --- Graph - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Small-20250912-073410_Graph \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Small-20250912-073410_Graph/aggregated/aggregated_Graph.csv

Luca-Medium-20250912-073410 --- KG - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Medium-20250912-073410 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Medium-20250912-073410/aggregated/aggregated_KG.csv


Luca-Medium-20250912-073410 --- Hybrid - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Medium-20250912-073410_Hybrid \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Medium-20250912-073410_Hybrid/aggregated/aggregated_Hybrid.csv

Luca-Medium-20250912-073410 --- Graph - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Medium-20250912-073410_Graph \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Medium-20250912-073410_Graph/aggregated/aggregated_Graph.csv

Luca-Long-20250912-073410 --- KG - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Long-20250912-073410 \
  --rag-type KG \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Long-20250912-073410/aggregated/aggregated_KG.csv

Luca-Long-20250912-073410_Hybrid --- Hybrid - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Long-20250912-073410_Hybrid \
  --rag-type Hybrid \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Long-20250912-073410_Hybrid/aggregated/aggregated_Hybrid.csv

Luca-Long-20250912-073410_Graph --- Graph - 5

python one_click_rag_eval.py \
  --exp-dirs data/Luca-Long-20250912-073410_Graph \
  --rag-type Graph \
  --support-extra "--light-clean --tf-th 0.40 --cj-th 0.30 --use-tfidf 1 --use-char3 1 --use-topic 1 --topic-th 0.30 --fusion rrf --rrf-k 60 --decision vote --vote-k 2 --emit-coverage" \
  --out-csv data/Luca-Long-20250912-073410_Graph/aggregated/aggregated_Graph.csv




