import json
rows = [json.loads(line) for line in open("ttft_results_rerun.jsonl")]
none_rows = [r for r in rows if r["ttft_seconds"] is None and r["error"] is None]
print(f"Total rerun rows: {len(rows)}")
print(f"Still None: {len(none_rows)}")