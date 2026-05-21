# M1 Scenario-Level Benchmark — Pilot

Run date: 2026-05-21
Results file: `results/m1_scenario_benchmark_pilot.json`

## What this benchmark measures

This pilot uses six representative speaking-policy scenarios that mirror the simulator's key decision points:

- speak on silence with a queued utterance
- wait while the user is speaking
- yield on interruption
- yield on collision
- clarify after prolonged silence / no reply
- repair from the repair state

Each model is scored on:
- action accuracy vs the expected label
- parse rate (did we get a usable `ACTION:` line?)
- latency per scenario
- raw response text saved in JSON

## Models in this reduced pilot

- `hermes/current-config` (Hermes CLI using the current OpenAI Codex-backed config)
- `openrouter/liquid-lfm-2.5-1.2b-instruct:free`
- `openrouter/openai-gpt-oss-safeguard-20b`

## Results

| Model | Accuracy | Parse rate | Avg latency | Notes |
|---|---:|---:|---:|---|
| `hermes/current-config` | 83.3% | 100% | 4.037s | Best of the reduced pilot; only missed the clarify-after-silence case |
| `openrouter/liquid-lfm-2.5-1.2b-instruct:free` | 16.7% | 100% | 0.541s | Fast, but mostly chose `wait` incorrectly |
| `openrouter/openai-gpt-oss-safeguard-20b` | 16.7% | 16.7% | 0.826s | Fast, but usually returned reasoning-only output and no parseable action |

## Notable per-scenario findings

### Hermes/current-config (`gpt-5.4-mini` through openai-codex)
- Correct on:
  - speak_on_silence
  - wait_while_user_speaks
  - yield_on_interrupt
  - yield_on_collision
  - repair_in_repair_state
- Missed:
  - clarify_after_silence → returned `wait` instead of `clarify`

This is materially better than the earlier prompt-level spot checks and suggests the model can follow the policy when the scenario framing is sufficiently explicit, but it still has a weak spot around the clarify decision.

### Liquid 1.2B instruct
- Parsed cleanly and was fast.
- Mostly collapsed to `wait` across the board.
- Not a useful policy comparator yet, though it might be useful as a cheap baseline or prompt canary.

### GPT-OSS safeguard
- Very fast.
- Poor fit for this harness because it often returned reasoning-only output with no parseable `content`.
- Not suitable as-is for action-level benchmarking.

## Takeaway

The scenario-level benchmark is more useful than the earlier prompt-only probe because it tests the actual policy states we care about.

The current read:
- `gpt-5.4-mini` via Hermes/openai-codex is the only model in this pilot that looks remotely competitive.
- It is still not clearly better than the deterministic policy, but it is close enough to justify more scenario data.
- The model still seems to struggle most with the clarify-after-silence decision.

## Next steps

1. Add more candidate models as they become available.
2. Keep the same six scenarios so comparisons remain apples-to-apples.
3. Add a few-shot variant of the benchmark prompt if we want to see whether the clarify case can be fixed without changing the model.
4. Consider storing multiple runs per model to measure variance, not just one seed.
