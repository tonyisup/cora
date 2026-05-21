# M1 Speaking-Policy Model Benchmark — Pilot

Run date: 2026-05-21
Results file: `results/m1_model_benchmark_pilot.json`

## Setup

We benchmarked 4 models on 4 representative speaking-policy prompts:
- speak on silence with a queued utterance
- yield on collision
- clarify after silence / no reply
- wait while the user is speaking

Each model was scored for:
- action accuracy vs expected label
- parse rate (whether a valid `ACTION:` line was extracted)
- average latency per prompt

## Pilot Results

| Model | Accuracy | Parse rate | Avg latency | Notes |
|---|---:|---:|---:|---|
| `openrouter/liquid-lfm-2.5-1.2b-instruct:free` | 75% | 100% | 0.720s | Fastest usable model in this pilot; missed the collision/yield case |
| `hermes/openai-codex:gpt-5.4-mini` | 25% | 100% | 5.693s | Parsed fine, but often returned the wrong turn-taking action |
| `openrouter/owl-alpha` | 25% | 100% | 11.187s | Correct on only 1/4 prompts; slowest usable model tested |
| `openrouter/openai-gpt-oss-safeguard-20b` | 0% | 0% | 0.637s | Fast, but returned no usable `content` for this prompt format |

## Observations

1. `gpt-5.4-mini` through Hermes/openai-codex is reachable, but on this prompt set it is not a good policy comparator yet.
2. `owl-alpha` is also not a fit for the current prompt as written; it tends to confuse `wait` vs `yield` and `clarify` vs `speak`.
3. The tiny `liquid/lfm-2.5-1.2b-instruct:free` model was surprisingly competitive on the easier prompts, but still failed the collision case.
4. `gpt-oss-safeguard-20b` appears unsuitable for this benchmark because it returns reasoning-only output with no regular assistant content.

## Takeaway

This pilot suggests the benchmark loop is useful, but the prompt itself still matters a lot. The next step is to:
- keep collecting model results under the same fixed prompt set
- add few-shot examples for collision / clarify / yield
- then compare whether accuracy improves without hidden prompt changes

For now, the deterministic policy remains the strongest and most reliable baseline.
