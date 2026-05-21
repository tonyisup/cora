# M1 Speaking Policy — Phase 4 Step 2 Results: LLM vs Deterministic Comparison

## Setup

Connected local OMLX API (`http://127.0.0.1:8666/v1`, model: `gemma-4-e4b-it-4bit`) as the LLM callable for `PromptSpeakingPolicy`. Ran 5 trials with seed=42, identical scenario distribution as the deterministic baseline.

## Results (5 trials, seed=42)

| Metric | Prompt (LLM) | Deterministic | Δ |
|--------|-------------|--------------|---|
| Interruption rate | 0.0% | 0.0% | — |
| Turn continuation | 15.1% | 76.2% | **-61.1%** |
| Repair success | 0.0% | 0.0% | — |
| Clarification efficiency | 0.0% | 40.0% | **-40.0%** |
| Collision rate | 91.0% | 0.0% | **+91.0%** |
| Response gap | 514ms | 372ms | +142ms |

## Key Findings

### 1. LLM causes constant collisions (91% vs 0%)

The LLM policy causes collisions in nearly every trial. Looking at the turn records:
- Trial 0: 8 collisions (LLM) vs 0 (deterministic)
- Trial 1: 6 collisions (LLM) vs 0 (deterministic)
- Trial 2: 7 collisions (LLM) vs 0 (deterministic)

The LLM is emitting SPEAK actions when the user is also speaking, causing simultaneous speech.

### 2. LLM turn-taking is broken (15% vs 76% continuation)

The LLM repeatedly speaks at the wrong time:
- Speaks while user is speaking (collision)
- Repeats the same utterance multiple times with nested quotes
- Doesn't wait for silence before speaking

Example from Trial 0:
```
Turn 0: agent | speak | "Hello! How can I help you today?"
Turn 1: agent | speak | ""Hello! How can I help you today?""
Turn 2: agent | speak | """Hello! How can I help you today?"""
Turn 3: user  | user_speech_start | Yes, it keeps making a weird noise.
Turn 4: agent | speak | "I see. Can you tell me a bit more about the noise?"
```

The agent speaks 3 times (with escalating quote nesting) before the user responds, then speaks again immediately when the user starts talking.

### 3. Root cause: LLM doesn't understand the action format

Testing the LLM directly with a collision scenario:
- **Input**: `attribution: both`, `state: speak`
- **Expected**: `ACTION: yield`
- **LLM returned**: `ACTION: wait` with explanation "agent must wait for the user to take their turn"

The LLM doesn't understand that `attribution: both` means simultaneous speech requiring an immediate yield. It treats it as a normal wait situation.

### 4. Prompt engineering needed

The current prompt is too verbose and the LLM doesn't follow the rules. Issues:
- The LLM ignores the "yield immediately on collision" rule
- The LLM generates natural language instead of structured actions
- The LLM doesn't understand the timing implications of `attribution: both`

## Recommendations

### Short-term: Improve the prompt
1. Make the collision rule more explicit: "If attribution is 'both', you MUST yield immediately"
2. Add few-shot examples showing correct behavior for each scenario type
3. Reduce prompt length to minimize token generation time
4. Add explicit "DO NOT speak while user is speaking" constraint

### Medium-term: Consider a different approach
The LLM is currently worse than the deterministic policy on every metric. Options:
1. **Fine-tune the prompt** with more explicit rules and examples
2. **Use the LLM as a critic** rather than the primary policy — let the deterministic policy act, and use the LLM to evaluate and suggest improvements
3. **Distill the deterministic policy** into a smaller model that runs faster
4. **Accept the deterministic policy as the baseline** and focus on Milestone 2 (distillation)

### Long-term: The deterministic policy is the product

The deterministic state machine with tuned thresholds (200ms silence, immediate yield on collision) performs well:
- 76% turn continuation (target > 70%)
- 81% repair success on exercised scenarios
- 100% clarification success on exercised scenarios
- 0% collision rate

The LLM prompt policy needs significant prompt engineering to match this performance. The deterministic policy should be considered the current best policy, not a baseline to beat.

## Files

- `results/m4_prompt_llm_5trials.json` — LLM prompt policy results (5 trials)
- `results/m4_deterministic_5trials.json` — Deterministic policy results (5 trials)
- `results/m4_harder_scenarios.json` — Deterministic policy results (50 trials)
