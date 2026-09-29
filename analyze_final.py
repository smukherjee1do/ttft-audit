import json
import pandas as pd

rows = [json.loads(line) for line in open("ttft_final.jsonl")]
df = pd.DataFrame(rows)

# Main summary: median and 90th-percentile TTFT per model/size
summary = df[df["error"].isna()].groupby(["model", "prompt_size_label"])["ttft_seconds"].agg(
    median="median",
    p90=lambda x: x.quantile(0.9),
    count="count"
).reset_index()

print(summary.to_string())
summary.to_csv("ttft_summary.csv", index=False)

# The DeepSeek V3.2 side-comparison against DO's published 0.96s claim
deepseek_10k = df[(df["model"] == "deepseek-3.2") & (df["prompt_size_label"] == "deepseek_10k")]
print("\nDeepSeek V3.2 @ ~10K tokens:")
print("  Median TTFT:", deepseek_10k["ttft_seconds"].median())
print("  90th percentile TTFT:", deepseek_10k["ttft_seconds"].quantile(0.9))
print("  DO's published claim: 0.96s")

# How TTFT scales with prompt size, per model
pivot = summary.pivot(index="model", columns="prompt_size_label", values="median")
print("\nMedian TTFT by model and prompt size:")
print(pivot)