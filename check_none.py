import json
from collections import Counter

rows = [json.loads(line) for line in open("ttft_results.jsonl")]
none_rows = [r for r in rows if r["ttft_seconds"] is None and r["error"] is None]

print(f"Total rows: {len(rows)}")
print(f"Rows with TTFT=None and no error: {len(none_rows)}")
print(Counter((r["model"], r["prompt_size_label"]) for r in none_rows))