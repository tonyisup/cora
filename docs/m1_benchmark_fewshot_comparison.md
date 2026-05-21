# M1 Scenario-Level Benchmark — Few-Shot Comparison

Baseline run:  `results/m1_scenario_benchmark_15_models.json`
Few-shot run:  `results/m1_scenario_benchmark_15_models_fewshot_2.json`

## Setup

6 scenarios × 15 models, with a 4-example few-shot prefix added to every prompt:
- Example 1: speak_on_silence (correct: `speak`)
- Example 2: yield_on_collision (correct: `yield`)
- Example 3: clarify_after_silence (correct: `clarify`)
- Example 4: repair_in_repair_state (correct: `repair`)

## Results

| Model | Baseline acc | Few-shot acc | Baseline clarify | Few-shot clarify | Latency (few-shot) |
|---|---:|---:|---:|---:|---:|
| anthropic/claude-3.5-haiku | 66.7% | **100.0%** | wait | **clarify** | 1.486s |
| google/gemini-3.1-flash-lite | 66.7% | 66.7% | clarify | wait | 0.808s |
| hermes/openai-codex:gpt-5.4-mini | 83.3% | 83.3% | wait | wait | 6.486s |
| hermes/openai-codex:gpt-5.4 | 83.3% | 83.3% | wait | wait | 7.758s |
| hermes/openai-codex:gpt-5.5 | 83.3% | 83.3% | wait | wait | 10.140s |
| ibm-granite/granite-4.1-8b | 66.7% | 66.7% | wait | speak | 0.477s |
| meta-llama/llama-3.2-3b-instruct | 0.0% | 50.0% | None | speak | 0.640s |
| openrouter/owl-alpha | 66.7% | 66.7% | None | None | 4.539s |
| openrouter/liquid-lfm-2.5-1.2b-instruct:free | 16.7% | 16.7% | yield | speak | 0.569s |
| nvidia/nemotron-nano-9b-v2 | 0.0% | 0.0% | wait | None | 1.507s |
| openrouter/openai-gpt-oss-safeguard-20b | 16.7% | 16.7% | None | None | 0.730s |
| deepseek/deepseek-r1-distill-llama-70b | 0.0% | 0.0% | None | None | 3.613s |
| google/gemini-4-26b-a4b-instruct | 0.0% | 0.0% | None | None | 0.079s |
| moonshotai/kimi-k2.5-flash | 0.0% | 0.0% | None | None | 0.054s |
| minimax/minimax-m2.5-flash | 0.0% | 0.0% | None | None | 0.055s |

## Key Findings

1. **Claude 3.5 Haiku is the only model that nailed all 6 scenarios after few-shot.**
   - It went from 66.7% to 100%.
   - It is the only model where clarify was correct in both the focused and full few-shot runs.
   - Latency is reasonable: ~1.5s avg.

2. **Few-shot did not fix clarify for the Codex models.**
   - gpt-5.4-mini, gpt-5.4, and gpt-5.5 all stayed at 83.3%.
   - They all answered `wait` on the clarify case regardless of few-shot.

3. **Few-shot did not fix Gemini either.**
   - Gemini got clarify wrong in both directions: `clarify` at baseline, `wait` in the full few-shot run.
   - Accuracy stayed flat at 66.7%.

4. **Some weaker models became more parseable but not more accurate.**
   - Llama 3.2 3B went from 0% to 50%, but its clarify answer is `speak`, not `clarify`.

5. **The zero-parse-rate models are structurally wrong for this harness.**
   - Kimi, Gemma, DeepSeek-R1-distill, safeguard model: they mostly return `None` for parsed_action.
   - They are not just inaccurate; they don't emit the action format at all.

## Takeaway

The few-shot prefix helps the right kind of model (Claude Haiku) but does not automatically fix the Codex models or Gemini on the clarify decision.

This suggests:
- clarify is not just a prompting gap; it may also be a model capability / alignment tendency
- if a model keeps answering `wait` instead of `clarify`, it may be because it defaults to "silence = keep listening"
- a hand-written `has_queued_utterance = False + silence > threshold → clarify` rule may still be the most reliable path

For the project, the actionable signal is:
- keep the deterministic policy as the trust baseline
- use Claude as the best current teacher candidate for distillation or contrastive data
- treat Gemini and the Codex models as useful but not yet policy-reliable without hard rules protecting the clarify case
