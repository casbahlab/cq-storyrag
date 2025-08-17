python3 build_cq_index_v2.py \
  --cq_path ../data/WembleyRewindCQs_with_beats_trimmed.json \
  --retrieval_mode Both \
  --out_root . \
  --sparql_root ../data/sparql_templates \
  --build_faiss \
  --embedder ollama \
  --ollama_model nomic-embed-text \
  --auto_ollama --ollama_pull \
  --validate

# Validate the CQ metadata

python3 validate_cq_metadata.py --meta KG/cq_metadata.json

python3 validate_cq_metadata.py --meta Hybrid/cq_metadata.json