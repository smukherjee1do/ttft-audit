# DO Serverless Inference — TTFT Audit

Firsthand, measured time-to-first-token (TTFT) results across six models on DigitalOcean Serverless Inference, collected September 2026. This repo holds the raw data, scripts, and prompts behind the accompanying article: [article link — add once published].

## What this measures

TTFT — the time between sending a request and receiving the first streamed token — across three prompt sizes (short, medium, long) for six models, plus a special condition for DeepSeek V3.2 at exactly 10,000 input tokens, matched against DigitalOcean's own published claim of sub-1-second TTFT for that model at that prompt size.

This is not a comparison against other inference providers. It's a check of DigitalOcean's own numbers against directly measured behavior on its Serverless Inference product.

## Models tested

- `gemma-4-31b-it` (Google) — small/fast baseline
- `glm-5.3` (Z.ai) — reasoning always enabled
- `kimi-k2.6` (Moonshot AI) — largest model tested (1T params)
- `deepseek-3.2` (DeepSeek) — the model DO's published TTFT claim is about
- `deepseek-v4.1-flash` (DeepSeek) — cross-referenced against an earlier prompt-caching audit on the same model
- `llama-4-maverick` (Meta)

## Methodology

- 20 repetitions per model/prompt-size combination (short ~20-50 tokens, medium ~500 tokens, long ~5,000-10,000 tokens), plus 20 repetitions of DeepSeek V3.2 at exactly 10,000 input tokens (verified via the model's own tokenizer, not estimated) — 380 calls total.
- Every prompt is prefixed with a fresh random UUID before sending, so every call is a guaranteed cache-cold request. Without this, prompt caching on repeated prompts would produce artificially fast TTFT readings.
- TTFT is measured from request start to the first non-empty streamed content chunk, checking both `content` and `reasoning_content` fields — several models (Kimi K2.6, DeepSeek V4.1 Flash, and GLM-5.3 on long prompts) stream reasoning tokens before answer tokens, and an earlier version of this script that only checked `content` silently missed TTFT for those calls. That bug and its fix are visible in the commit history.
- Responses were capped at 50 output tokens (`max_tokens=50`) since TTFT doesn't depend on response length, keeping the full run fast without affecting the measurement.
- Long-prompt source text is drawn from *Frankenstein* (Project Gutenberg, public domain).

## Files

- `measure_ttft.py` — main measurement script
- `rerun_affected.py` — targeted re-run script used to fix the reasoning-content bug for three affected models
- `build_prompts.py` — builds the prompt set and trims the DeepSeek V3.2 prompt to exactly 10,000 tokens
- `prompts.json` — the exact prompts used
- `ttft_final.jsonl` — the complete, corrected raw dataset (380 rows)
- `ttft_summary.csv` — aggregated median/90th-percentile TTFT per model and prompt size
- `frankenstein_medium_excerpt.txt`, `frankenstein_long_excerpt.txt` — source text for the medium/long prompts

## Known limitations

- Single DigitalOcean account, Tier 1 rate limits, tested over roughly a 10-minute window on one day in September 2026.
- Two extreme tail-latency events (78s and 187s TTFT) were observed on DeepSeek V3.2 specifically, out of 80 total calls to that model; zero comparable events occurred across 300 calls to the other five models. This is a real observation, not a measured general failure rate — treat it as evidence of tail-latency risk, not a known percentage.
- DigitalOcean's own documentation states serverless inference capacity is shared and latency is not guaranteed; these results should be read in that context, and may not reproduce identically on a different day, account, or tier.

## Reproducing this

1. Set `MODEL_ACCESS_KEY` as an environment variable with a DigitalOcean model access key.
2. `pip install openai pandas`
3. Run `build_prompts.py`, then `measure_ttft.py`, in that order.