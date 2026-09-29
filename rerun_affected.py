import os
import time
import json
import uuid
import random
from datetime import datetime, timezone
from openai import OpenAI

client = OpenAI(
    base_url="https://inference.do-ai.run/v1",
    api_key=os.environ["MODEL_ACCESS_KEY"],
)

with open("prompts.json") as f:
    PROMPTS = json.load(f)


def measure_ttft(model, prompt_text, size_label):
    session_marker = str(uuid.uuid4())
    full_prompt = f"[session:{session_marker}]\n{prompt_text}"

    start_time = time.perf_counter()
    first_token_time = None
    full_response_chunks = []
    error = None

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True,
            max_tokens=50,
        )
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content_piece = getattr(delta, "content", None) or getattr(delta, "reasoning_content", None)
                if content_piece:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_response_chunks.append(content_piece)
    except Exception as e:
        error = str(e)

    end_time = time.perf_counter()
    ttft = (first_token_time - start_time) if first_token_time else None
    total_time = end_time - start_time

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_size_label": size_label,
        "session_marker": session_marker,
        "ttft_seconds": ttft,
        "total_time_seconds": total_time,
        "error": error,
    }


conditions_to_rerun = []
for size in ["short", "medium", "long"]:
    for rep in range(20):
        conditions_to_rerun.append(("kimi-k2.6", size))
        conditions_to_rerun.append(("deepseek-v4.1-flash", size))
for rep in range(20):
    conditions_to_rerun.append(("glm-5.3", "long"))

random.shuffle(conditions_to_rerun)

for idx, (model, size) in enumerate(conditions_to_rerun):
    result = measure_ttft(model, PROMPTS[size], size)
    with open("ttft_results_rerun.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[{idx+1}/{len(conditions_to_rerun)}] {model}/{size}: TTFT={result['ttft_seconds']}, error={result['error']}")
    time.sleep(1)