# Pivot: Learns to Speak

This doc captures the idea behind the new project pivot:

Not "generate speech from text" but "build an intelligence that learns to speak."

## Core Thesis

Corollary is a self-listening loop. The pivot extends that into a self-speaking loop.

The interesting system is not a TTS engine with better audio quality. It is a policy learner that discovers:

- when to speak
- what to say
- how long to wait
- how to repair misunderstandings
- how to modulate prosody, emphasis, and pacing
- how to adapt to conversational context and social feedback

Speech is the behavior. Audio rendering is just the transport.

## What Changes

Corollary today is centered on closed-loop perception:

- input is captured
- self/world attribution happens
- the graph and memory update
- output is synthesized
- the loop is measured

The pivot changes the objective from "speak clearly" to "learn speech as an interactive control policy."

That means the model should treat vocalization as a decision under uncertainty, not a default response after text generation.

## Reusable Pieces From Corollary

These look directly reusable:

- `packages/sag/` for semantic grounding and concept tracking
- `packages/thalamus/` for gating and attribution
- `packages/reasoning/` for provider-agnostic policy decisions
- `packages/hippocampus/` for memory and consolidation
- `packages/m2_policy/` and later policy work as the template for conservative action selection
- `scripts/` harness patterns for offline eval, live loops, and acceptance tests

The existing audio stack is still useful, but in the pivot it becomes instrumentation rather than the main product.

## New Capabilities Needed

The new system needs a learning loop for speech behavior itself:

- turn-taking policy
- response timing
- repair strategy after failed understanding
- prosody selection
- conversational style adaptation
- self-evaluation from feedback

If Corollary asks "Did I hear myself correctly?", the pivot asks "Did I speak well enough to keep the conversation moving?"

## Suggested Architecture

### 1. Perception Layer

Collect:

- user speech
- agent speech
- timing gaps
- interruptions
- corrections
- explicit feedback
- conversational outcome signals

### 2. Speech Policy Layer

Learns a policy over conversational actions:

- speak now
- wait
- ask a clarification
- restate
- repair
- soften
- emphasize
- stay silent

### 3. Expression Layer

Converts the policy decision into an utterance plan:

- text content
- prosody plan
- speaking rate
- pause placement
- emphasis

### 4. Execution Layer

Turns the utterance plan into audio or text output.

This can still use TTS or a voice model, but that is downstream of the policy.

### 5. Feedback Layer

Uses downstream signals to update the policy:

- user responds quickly or slowly
- user interrupts
- user corrects the agent
- conversation continues smoothly or stalls
- the agent has to repeat itself

## Learning Target

The cleanest initial target is not "general speech."

Start with a narrower, measurable skill:

- turn-taking
- repair
- pacing
- interruption handling

That keeps the first loop trainable and lets the project accumulate evidence instead of vibes.

## Recommended First Milestone

Build a "speaking policy" demo that can choose among a small action set:

- `speak`
- `wait`
- `clarify`
- `repair`
- `yield`

Success means the policy improves conversation flow under repeated trials.

## Evaluation Ideas

Possible metrics:

- interruption latency
- repair success rate
- clarification efficiency
- turn continuation rate
- user correction rate
- average response appropriateness rating

The key question is whether the system becomes better at participating in conversation, not just generating audio.

## Project Framing

Working name options:

- Cora
- Cora Speak
- Learned Speech Loop
- Self-Speaking Agent

My recommendation is to treat this as a sibling project to Corollary, not a rename.

Corollary remains the self-listening loop.
This project becomes the self-speaking loop.

## Open Questions

- What is the smallest speech behavior we can learn first?
- What feedback signal is strong enough to train on?
- Do we optimize for conversation success, not speaker imitation?
- Is text the intermediate representation, or do we learn a richer utterance plan?
- What parts should stay deterministic for safety and testability?

