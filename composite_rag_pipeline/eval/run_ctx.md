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
