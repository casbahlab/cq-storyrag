import csv
import json
from pathlib import Path

input_csv = Path("WembleyRewindCQs_with_beats_trimmed.csv")
output_json = Path("WembleyRewindCQs_with_beats_trimmed.json")

rows = []
with input_csv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Remove columns with None or empty string
        clean_row = {k: v for k, v in row.items() if v and v.strip()}
        rows.append(clean_row)

with output_json.open("w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"JSON saved: {output_json} ({len(rows)} records)")
