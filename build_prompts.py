"""
build_prompts.py

Run this once, from inside your ttft-audit project folder, after placing
frankenstein_medium_excerpt.txt and frankenstein_long_excerpt.txt in the
same folder (both provided alongside this script).

What it does:
1. Defines your short prompt directly.
2. Builds your medium and long prompts from the Frankenstein excerpts + a question.
3. Calls the live API to get the EXACT token count for each prompt, using the
   model's own tokenizer (via the usage field) rather than estimating.
4. Automatically trims the long-prompt text, sentence by sentence, until the
   DeepSeek V3.2 special condition lands as close as possible to exactly
   10,000 input tokens -- this is the one prompt that needs to be precise,
   since it's your side-comparison against DO's published 0.96s claim.
5. Writes everything to prompts.json, ready for measure_ttft.py to load.

Requires: MODEL_ACCESS_KEY set as an environment variable (same as the rest
of the guide), and the two excerpt .txt files in this folder.
"""

import os
import json
import re
from openai import OpenAI

client = OpenAI(
    base_url="https://inference.do-ai.run/v1",
    api_key=os.environ["MODEL_ACCESS_KEY"],
)


def count_tokens(text, model):
    """Get the exact prompt token count from the model's own tokenizer."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        max_tokens=1,
    )
    return resp.usage.prompt_tokens


def split_sentences(text):
    """Very simple sentence splitter -- good enough for trimming prose."""
    return re.split(r"(?<=[.!?])\s+", text)


# --- 1. Short prompt ---
SHORT_PROMPT = "What is the capital of France, and why is it significant historically?"

# --- 2. Medium prompt ---
with open("frankenstein_medium_excerpt.txt") as f:
    medium_excerpt = f.read().strip()
MEDIUM_PROMPT = (
    f"{medium_excerpt}\n\n"
    "Summarize the key points of the passage above in two sentences."
)

# --- 3. Long prompt (general, ~5,000-10,000 tokens, used for the 5 non-DeepSeek models) ---
with open("frankenstein_long_excerpt.txt") as f:
    long_excerpt_full = f.read().strip()
LONG_PROMPT = (
    f"{long_excerpt_full}\n\n"
    "Based on the passage above, describe the narrator's emotional state "
    "and what events led to it."
)

# --- 4. DeepSeek V3.2 special condition: trim to exactly 10,000 tokens ---
DEEPSEEK_MODEL = "deepseek-3.2"
sentences = split_sentences(long_excerpt_full)
TARGET_TOKENS = 10000
QUESTION_SUFFIX = (
    "\n\nBased on the passage above, describe the narrator's emotional state "
    "and what events led to it."
)

print("Trimming long excerpt to hit exactly 10,000 tokens for DeepSeek V3.2...")
low, high = 1, len(sentences)
best_text, best_count, best_diff = None, None, float("inf")

# Binary search over how many sentences to include, converging on the
# sentence count whose token total is closest to 10,000.
while low <= high:
    mid = (low + high) // 2
    candidate_text = " ".join(sentences[:mid]) + QUESTION_SUFFIX
    tok_count = count_tokens(candidate_text, DEEPSEEK_MODEL)
    diff = abs(tok_count - TARGET_TOKENS)

    print(f"  sentences={mid:4d}  tokens={tok_count:6d}  diff={diff}")

    if diff < best_diff:
        best_text, best_count, best_diff = candidate_text, tok_count, diff

    if tok_count < TARGET_TOKENS:
        low = mid + 1
    elif tok_count > TARGET_TOKENS:
        high = mid - 1
    else:
        break  # exact match, stop early

print(f"\nBest match: {best_count} tokens (target was {TARGET_TOKENS}).")
print("Report this exact number in the article rather than rounding it.")

DEEPSEEK_10K_PROMPT = best_text

# --- 5. Verify actual token counts for everything, and print a summary ---
print("\nFinal prompt token counts (measured against deepseek-3.2's tokenizer):")
print("  short: ", count_tokens(SHORT_PROMPT, DEEPSEEK_MODEL))
print("  medium:", count_tokens(MEDIUM_PROMPT, DEEPSEEK_MODEL))
print("  long:  ", count_tokens(LONG_PROMPT, DEEPSEEK_MODEL))
print("  deepseek_10k:", best_count)
print(
    "\nNote: short/medium/long counts above are measured on DeepSeek V3.2's "
    "tokenizer as a reference point only -- different models tokenize text "
    "slightly differently, so the same prompt may count as a marginally "
    "different number of tokens on Gemma 4, GLM-5.3, Kimi K2.6, etc. That's "
    "expected and fine for the short/medium/long conditions, since only the "
    "DeepSeek 10K condition needs to be exact."
)

# --- 6. Save everything to prompts.json ---
prompts = {
    "short": SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "long": LONG_PROMPT,
    "deepseek_10k": DEEPSEEK_10K_PROMPT,
}

with open("prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print("\nSaved prompts.json. You're ready for Step 5 in the execution guide.")
