# Pivot: Learns to Speak

This doc captures the idea behind the Cora project:

Not "generate speech from text" but "build an intelligence that learns to speak."

## Core Thesis

Cora is a self-listening loop that extends into a self-speaking loop.

The interesting system is not a TTS engine with better audio quality. It is a policy learner that discovers:

- when to speak
- what to say
- how long to wait
- how to repair misunderstandings
- how to modulate prosody, emphasis, and pacing
- how to adapt to conversational context and social feedback

Speech is the behavior. Audio rendering is just the transport.

## Two Tracks

This pivot should be treated as two related but distinct tracks:

1. Practical speaking policy
   - learns turn-taking, timing, repair, and conversational control over a controllable TTS or voice output layer
   - this is the first productizable milestone
2. Sound-first motor learning
   - learns acoustic imitation and sound control directly
   - this is the deeper research path inspired by infant babbling and oscillator-based generation

Track 1 solves the near-term conversation problem.
Track 2 explores the more ambitious sound-learning problem without blocking progress on Track 1.

## Learning Sequence

The recommended sequence is:

1. Offline simulation first
   - replay recorded dialogues
   - generate simulated user responses
   - keep runs repeatable and measurable
2. Deterministic or prompt-based baseline first
   - use a small explicit action set
   - make the baseline easy to inspect and debug
3. Distill into a smaller learned policy
   - once the behavior is stable, compress the policy
4. RL only after the environment is shaped
   - use RL or PPO later, not on day one
   - only after rewards, simulator behavior, and action space are stable

## What Changes

Cora today is centered on closed-loop perception:

- input is captured
- self/world attribution happens
- the graph and memory update
- output is synthesized
- the loop is measured

The pivot changes the objective from "speak clearly" to "learn speech as an interactive control policy."

That means the model should treat vocalization as a decision under uncertainty, not a default response after text generation.

## Reusable Pieces From Cora

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

If Cora asks "Did I hear myself correctly?", the pivot asks "Did I speak well enough to keep the conversation moving?"

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

For Track 1, those downstream signals are conversational and behavioral.
For Track 2, those signals can be acoustic similarity and imitation success.

## LLM Role

The LLM should be treated as a teacher and long-term semantic store, not the initial online policy.

Good uses:

- semantic memory
- trajectory labeling
- critique and explanation
- high-level planning

Not the first use:

- direct low-level policy control before the action space is stable

## Learning Target

The cleanest initial target is not "general speech."

Start with a narrower, measurable skill:

- turn-taking
- repair
- pacing
- interruption handling

That keeps the first loop trainable and lets the project accumulate evidence instead of vibes.

For the first milestone, the policy should operate over a controllable speech layer rather than raw waveform generation.
That lets us measure turn-taking and repair before we take on full acoustic synthesis.

The first implementation should therefore be a deterministic or prompt-based controller over a small action set, with the learned policy coming later through distillation.

## Recommended First Milestone

Build a "speaking policy" demo that can choose among a small action set:

- `speak`
- `wait`
- `clarify`
- `repair`
- `yield`

Success means the policy improves conversation flow under repeated trials.

## Sound-First Development Path

The first reinforcement signal should be purely acoustic.

That means:

- input is a target sound
- output is a produced sound
- reward is acoustic similarity between target and output
- no text label is required at first

This makes the earliest problem a motor-learning loop, not a symbol-grounding loop.

Suggested progression:

1. Sound shape imitation
   - pitch contour
   - duration
   - loudness envelope
   - onset timing
2. Repeatability
   - the same sound should converge across trials
3. Weak terminal confirmation
   - "correct / incorrect" as a secondary reinforcement signal
4. Text association
   - connect sound patterns to words only after the sounds are stable
5. Meaning and context
   - attach semantics once the acoustic side is no longer brittle

The main risk to avoid is rewarding text too early. If that happens, the system can learn symbol association without learning to actually produce the sound.

This is the Track 2 research lane, not the first product lane.
It is the right place to explore babbling, imitation, oscillators, and neural acoustic control.

## SAG Expansion

The SAG should eventually store more than word abstractions.

It should also hold acoustic abstractions:

- sound prototypes
- sound families
- phoneme-like clusters
- exemplar recordings
- confidence over the mapping between sound and concept

Word nodes can then connect to both:

- the collection of words that define them
- the collection of sounds that express them

That gives the system a bidirectional bridge between sound and meaning instead of treating speech as a separate downstream renderer.

## Oscillator + Neural Controller

The transcript suggests a useful pattern:

- a rhythmic driver keeps the system active
- a recurrent substrate holds transient state
- a trainable readout shapes that state into a target output

That is a good fit for the sound-first phase.

Possible division of labor:

- driver / oscillator bank
  - provides stable rhythmic input
  - keeps the acoustic system moving
- recurrent acoustic substrate
  - holds short-term sound state
  - supports temporal pattern generation
- neural controller / readout
  - maps target goals to actual sound production
  - learns from similarity feedback
- SAG
  - stores abstractions, labels, and exemplar links
  - connects sound patterns to concepts later

This is why a neural network is worth checking out:

- not as a replacement for SAG
- but as the trainable mechanism that can shape sound over time

In other words, SAG can be the memory and grounding layer, while the neural part learns the continuous control policy that makes sound production work.

For the practical speaking-policy track, the neural controller can stay behind a controllable TTS layer at first.
That gives us a simpler policy-learning problem while preserving a path toward deeper acoustic learning later.

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

This project is named **Cora**.

The self-speaking loop (this pivot) extends the self-listening loop (the original Corollary system) into a conversational agent that learns speech as interactive control.

Earlier working name options included: Cora Speak, Learned Speech Loop, Self-Speaking Agent.

## Open Questions

## Still Open

These are the remaining design questions that need a concrete spec:

- The exact first action set for Track 1
- The design of the offline simulator and simulated user responses
- The precise reward function for Track 1
- The boundary between SAG, policy model, and prompt baseline
- The first acoustic primitives to formalize for Track 2
