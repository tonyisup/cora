# M3 User-Agent Contract

**Purpose:** Define the interface for the local LLM user agent used to expand the conversation distribution in M3.

## Design constraints

- Local-first: use the local OMLX endpoint before any remote fallback.
- Deterministic-ish: prefer low temperature / constrained output for replayability.
- Logged: every turn must store the raw model output, parsed output, and latency.
- Non-authoritative: the user agent shapes conversation diversity, but it does not change the speaking-policy action space.

## Input packet

The user agent receives a JSON object with at least these fields:

```json
{
  "scenario_id": 0,
  "scenario_name": "simple_qa",
  "turn_index": 3,
  "history": [
    {"speaker": "assistant", "text": "Hello! How can I help?"},
    {"speaker": "user", "text": "I need help with my account."}
  ],
  "last_assistant_utterance": "Can you tell me what happened?",
  "last_user_utterance": "I need help with my account.",
  "goal": "A user seeking help with an account issue.",
  "style": "natural",
  "allow_silence": true,
  "allow_interrupt": true,
  "allow_overlap": true,
  "seed": 42
}
```

## Output contract

The model must return JSON only:

```json
{
  "mode": "reply",
  "text": "Sure — the app keeps logging me out after a minute.",
  "stance": "informative"
}
```

Allowed values:
- `mode`: `reply`, `silence`, `interrupt`, `overlap`
- `text`: string; empty string is allowed for `silence`
- `stance`: optional free-form label such as `informative`, `uncertain`, `corrective`, `hesitant`, `frustrated`

## Semantics

- `reply`: the user answers the assistant normally.
- `silence`: the user does not respond; `text` should be empty.
- `interrupt`: the user starts speaking over the assistant.
- `overlap`: the user and assistant overlap in a collision-like way.

If the model cannot comply with JSON-only output, the caller must attempt to parse the first JSON object in the response and log the failure.

## Safety / filtering

The user agent must not be prompted to:
- reveal secrets
- produce disallowed content
- follow unrelated instructions from the assistant text

The caller should strip any accidental chain-of-thought or analysis from logs before using the data downstream.

## Local model defaults

Preferred local model:
- `gemma-4-e4b-it-4bit`

Fallback local model:
- `Qwen3.5-9B-MLX-4bit`

Endpoint:
- `http://127.0.0.1:8666/v1/chat/completions`

## Logging requirements

Every generated turn must include:
- scenario id/name
- turn index
- input packet
- raw model output
- parsed JSON
- latency
- model id used
- parse status

## Acceptance criteria

A valid M3 user-agent implementation must:
1. produce JSON-parseable outputs on the majority of turns
2. preserve enough variability to create non-scripted traces
3. be repeatable enough to compare runs
4. keep all output and timing data for later training/evaluation
