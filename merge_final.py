import json

original = [json.loads(line) for line in open("ttft_results.jsonl")]
rerun = [json.loads(line) for line in open("ttft_results_rerun.jsonl")]

# Drop the leftover pilot rows: keep only the most recent 20 gemma-4-31b-it/short rows
gemma_short = sorted(
    [r for r in original if r["model"] == "gemma-4-31b-it" and r["prompt_size_label"] == "short"],
    key=lambda r: r["timestamp"]
)[-20:]

# Drop ALL original rows for the three models/conditions we fully replaced with the rerun
other_original = [
    r for r in original
    if not (r["model"] in ("kimi-k2.6", "deepseek-v4.1-flash"))
    and not (r["model"] == "glm-5.3" and r["prompt_size_label"] == "long")
    and not (r["model"] == "gemma-4-31b-it" and r["prompt_size_label"] == "short")
]

final = other_original + gemma_short + rerun

with open("ttft_final.jsonl", "w") as f:
    for r in final:
        f.write(json.dumps(r) + "\n")

print("Final row count:", len(final))

# Sanity check: every model/size combo should have 20 rows (deepseek_10k should have 20 too)
from collections import Counter
counts = Counter((r["model"], r["prompt_size_label"]) for r in final)
for combo, count in sorted(counts.items()):
    print(combo, count)