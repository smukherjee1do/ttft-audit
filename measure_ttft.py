import os
import time
import json
import uuid
from datetime import datetime, timezone
from openai import OpenAI

client = OpenAI(
       base_url="https://inference.do-ai.run/v1",
       api_key=os.environ["MODEL_ACCESS_KEY"]
   )

with open("prompts.json") as f:
    PROMPTS = json.load(f)  # {"short": "...", "medium": "...", "long": "...", "deepseek_10k": "..."}

MODELS = [
       "gemma-4-31b-it",
       "glm-5.3",
       "kimi-k2.6",
       "deepseek-3.2",
       "deepseek-v4.1-flash",
       "llama-4-maverick",
   ]
PROMPT_SIZES = ["short", "medium", "long"]
REPETITIONS = 20
LOG_FILE = "ttft_results.jsonl"

def measure_ttft(model, prompt_text, size_label):
       # Prepend a fresh UUID so this exact prompt has never been sent before,
       # guaranteeing a cold, uncached call. Without this line, your TTFT
       # numbers on repeat prompts will quietly include cache hits and be meaningless.
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
               max_tokens=50
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
def log_result(result):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


# if __name__ == "__main__":
#     for i in range(5):
#         result = measure_ttft("gemma-4-31b-it", PROMPTS["short"], "short")
#         log_result(result)
#         print(result)
#         time.sleep(1)  # brief pause between calls       

import random

if __name__ == "__main__":
    conditions = []
    for model in MODELS:
        for size in PROMPT_SIZES:
            for rep in range(REPETITIONS):
                conditions.append((model, size))

    # Add the DeepSeek V3.2 special 10K-token condition, 20 reps, separately
    for rep in range(REPETITIONS):
        conditions.append(("deepseek-3.2", "deepseek_10k"))

    random.shuffle(conditions)

    for idx, (model, size) in enumerate(conditions):
        prompt_text = PROMPTS[size]
        result = measure_ttft(model, prompt_text, size)
        log_result(result)
        print(f"[{idx+1}/{len(conditions)}] {model} / {size}: "
                f"TTFT={result['ttft_seconds']}, error={result['error']}")
        time.sleep(1)  # be polite to the API and avoid rate limits