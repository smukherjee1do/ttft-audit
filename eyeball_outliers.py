import json
import pandas as pd

rows = [json.loads(line) for line in open("ttft_final.jsonl")]
df = pd.DataFrame(rows)

cells_to_check = [
    ("deepseek-3.2", "short"),
    ("deepseek-3.2", "deepseek_10k"),
    ("gemma-4-31b-it", "long"),
]

for model, size in cells_to_check:
    subset = df[(df["model"] == model) & (df["prompt_size_label"] == size)]
    values = sorted(subset["ttft_seconds"].tolist())
    print(f"\n{model} / {size} — {len(values)} values, sorted:")
    for v in values:
        print(f"  {v:.3f}")
    print(f"  mean:   {subset['ttft_seconds'].mean():.3f}")
    print(f"  median: {subset['ttft_seconds'].median():.3f}")
    print(f"  min:    {min(values):.3f}")
    print(f"  max:    {max(values):.3f}")

# import json
# rows = [json.loads(line) for line in open("ttft_final.jsonl")]
# outlier = [r for r in rows if r["model"] == "deepseek-3.2" and r["prompt_size_label"] == "deepseek_10k" and r["ttft_seconds"] and r["ttft_seconds"] > 100]
# print(outlier)

import json
rows = [json.loads(line) for line in open("ttft_final.jsonl")]
rows_sorted = sorted(rows, key=lambda r: r["timestamp"])
target_time = "2026-09-29T07:34:50"
nearby = [r for r in rows_sorted if abs(
    __import__("datetime").datetime.fromisoformat(r["timestamp"]) -
    __import__("datetime").datetime.fromisoformat(target_time + "+00:00")
).total_seconds() < 300]
for r in nearby:
    print(r["timestamp"], r["model"], r["prompt_size_label"], r["ttft_seconds"])

import json
rows = [json.loads(line) for line in open("ttft_final.jsonl")]
deepseek_all = [r for r in rows if r["model"] == "deepseek-3.2" and r["ttft_seconds"]]
extreme = sorted([r for r in deepseek_all if r["ttft_seconds"] > 10], key=lambda r: r["ttft_seconds"], reverse=True)
print(f"DeepSeek V3.2 total calls: {len(deepseek_all)}")
print(f"Calls with TTFT > 10s: {len(extreme)}")
for r in extreme:
    print(r["timestamp"], r["prompt_size_label"], r["ttft_seconds"])

other_models = [r for r in rows if r["model"] != "deepseek-3.2" and r["ttft_seconds"]]
other_extreme = [r for r in other_models if r["ttft_seconds"] > 10]
print(f"\nAll other models combined: {len(other_models)} calls, {len(other_extreme)} with TTFT > 10s")