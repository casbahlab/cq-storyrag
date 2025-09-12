python ttl_url_index.py extract --data ../data --mask "*.ttl"

python ttl_url_index.py crawl --max 200


python ttl_url_index.py extract-objects \
  --data ../data --mask "*.ttl" \
  --out-urls urls_from_objects.txt \
  --out-context-csv urls_context.csv

python ttl_url_index.py extract-objects \
  --data ../data --mask "*.ttl" \
  --out-urls urls_from_objects.txt \
  --out-context-csv urls_context.csv \
  --exclude-domain wembrewind.live \
  --exclude-prefix http://wembrewind.live


python extract_schema_url.py \
  --data ../data --mask "*.ttl" \
  --out-urls schema_url_values.txt


python pull_content_bundle.py \
  --in schema_url_values.txt \
  --out content_bundle.jsonl


python prepare_content_index.py \
  --in schema_url_values.txt \
  --out-dir output \
  --summarizer extractive \
  --target-words 120


python prepare_content_index.py \
  --in schema_url_values.txt \
  --out-dir output \
  --summarizer gemini \
  --target-words 120




