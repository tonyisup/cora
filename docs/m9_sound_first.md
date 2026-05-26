# M9 — Sound-First Motor Learning (Track 2)

> **Status:** Scoped. M8 complete. M9 is the second track of the self-speaking pivot.

## Goal

Build a motor-learning loop that discovers how to produce sound — not by learning text-to-speech, but by learning acoustic imitation and sound control directly. The system learns to produce sounds that match targets, then associates those sounds with words and meaning.

This is the deeper research path from the pivot doc: "Not 'generate speech from text' but 'build an intelligence that learns to speak.'"

## Core Thesis

The interesting system is not a TTS engine with better audio quality. It is a policy learner that discovers:
- what sounds it can produce
- how to control pitch, duration, loudness, and timing
- how to imitate target sounds
- how to associate sounds with meaning (only after the sounds are stable)

Speech is the behavior. Audio rendering is just the transport.

## Architecture

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    M9 Sound-First Loop                       │
  │                                                              │
  │  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
  │  │  Target      │───▶│  Neural      │───▶│  Acoustic      │  │
  │  │  Sound       │    │  Controller  │    │  Output        │  │
  │  │  (reference) │    │  (readout)   │    │  (produced)    │  │
  │  └─────────────┘    └──────┬───────┘    └───────┬────────┘  │
  │                            │                     │           │
  │                     ┌──────▼───────┐    ┌───────▼────────┐  │
  │                     │  Recurrent   │    │  Similarity    │  │
  │                     │  Substrate   │    │  Reward        │  │
  │                     │  (oscillator │    │  (acoustic     │  │
  │                     │   bank)      │    │   distance)    │  │
  │                     └──────────────┘    └────────────────┘  │
  │                                                              │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │  SAG (acoustic expansion)                             │   │
  │  │  Sound prototypes, phoneme clusters, exemplar links   │   │
  │  └──────────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────┘
```

### Division of Labor

| Component | Role |
|-----------|------|
| **Oscillator bank** | Provides stable rhythmic input; keeps the acoustic system moving |
| **Recurrent substrate** | Holds short-term sound state; supports temporal pattern generation |
| **Neural controller / readout** | Maps target goals to actual sound production; learns from similarity feedback |
| **SAG (expanded)** | Stores acoustic abstractions (prototypes, clusters, exemplars); connects sound to meaning later |

## Sound-First Development Path

Per the pivot doc, the progression is:

1. **Sound shape imitation** — pitch contour, duration, loudness envelope, onset timing
2. **Repeatability** — same sound converges across trials
3. **Weak terminal confirmation** — "correct / incorrect" as secondary reinforcement signal
4. **Text association** — connect sound patterns to words only after sounds are stable
5. **Meaning and context** — attach semantics once the acoustic side is no longer brittle

**Key risk to avoid:** rewarding text too early. If that happens, the system learns symbol association without learning to actually produce the sound.

## M9a — Sound Imitation Primitive

**Goal:** Build the simplest possible motor-learning loop. A target sound is presented, the system produces a sound, and the reward is acoustic similarity.

### Components

1. **Target sound generator** — produces reference audio (simple tones, vowels, syllables)
2. **Acoustic producer** — the system's output (starts as a simple oscillator + neural readout)
3. **Similarity reward** — MFCC/chroma/mel distance between target and produced sound
4. **SAG acoustic node** — stores the target sound as a prototype with confidence

### Primitives

| Primitive | Description | Reward Signal |
|-----------|-------------|---------------|
| `pitch_contour` | Match the pitch trajectory of the target | DTW distance on F0 |
| `duration` | Match the length of the target | Absolute time difference |
| `loudness` | Match the energy envelope | MSE on RMS envelope |
| `onset_timing` | Match the attack time | Time to 50% energy |

### Acceptance Criteria

1. System produces non-silence when given a target (not just random noise)
2. Similarity score improves over 10+ trials on the same target (learning signal exists)
3. Different targets produce measurably different outputs (not just one sound)
4. SAG stores at least 3 distinct acoustic prototypes with confidence scores

### Non-Goals (M9a)

- No text or word association
- No TTS or voice model
- No conversational context
- No prosody or emphasis control
- Simple targets only (single vowels, pure tones, simple syllables)

## M9b — Repeatability and Convergence

**Goal:** The system reliably produces the same sound when given the same target, across multiple trials.

### Components

1. **Trial runner** — presents target, runs producer, scores, repeats
2. **Convergence detector** — measures variance across trials for the same target
3. **SAG prototype refinement** — updates stored prototype with running average of successful productions

### Acceptance Criteria

1. Coefficient of variation < 20% across 5 trials for the same target
2. SAG prototype confidence increases with successful trials
3. System can produce at least 5 distinct sounds with above convergence

## M9c — Weak Terminal Confirmation

**Goal:** Add a secondary reinforcement signal — a human or automated "correct / incorrect" judgment — to guide learning beyond pure acoustic similarity.

### Components

1. **Terminal judge** — binary correct/incorrect signal (can be automated via threshold)
2. **Combined reward** — weighted combination of acoustic similarity + terminal confirmation
3. **SAG label attachment** — associate sound prototypes with simple labels

### Acceptance Criteria

1. Terminal confirmation improves convergence speed vs similarity-only
2. System can learn to produce a sound that passes the terminal judge > 70% of trials
3. SAG nodes have both acoustic data and label associations

## M9d — Text Association

**Goal:** Connect stable sound patterns to words. Only attempted after M9c demonstrates reliable sound production.

### Components

1. **Word-sound pairing** — present word + target sound together
2. **Bidirectional SAG edges** — word node <-> sound prototype node
3. **Production from word** — given a word, produce the associated sound

### Acceptance Criteria

1. Given a word, system produces the associated sound (not just any sound)
2. SAG has bidirectional edges: word -> sound prototype -> word
3. At least 10 word-sound pairs stored and retrievable

## SAG Acoustic Expansion

The current SAG stores word abstractions (Node with `core`, `context`, `grounding`). M9 requires:

```python
class AcousticNode(Node):
    """A sound prototype — entity, phoneme-like cluster, or exemplar."""
    prototype_audio: Optional[np.ndarray]  # Reference waveform or mel
    mfcc_centroid: Optional[np.ndarray]    # MFCC feature centroid
    pitch_mean_hz: Optional[float]
    duration_mean_s: Optional[float]
    production_count: int = 0               # How many times produced
    avg_similarity: float = 0.0             # Running avg similarity score
    label: Optional[str] = None             # Associated word/label (M9c+)
```

New edge types:
- `SOUND_OF` — acoustic node represents the sound of a word node
- `SIMILAR_TO` — acoustic nodes are perceptually similar
- `PRECEDES` — temporal ordering in a sequence

## Evaluation Metrics

| Metric | Target | How measured |
|--------|--------|-------------|
| Similarity improvement | >20% over 10 trials | MFCC distance trend |
| Repeatability | CV < 20% across 5 trials | Variance of acoustic features |
| Distinctness | 5+ separable sounds | Clustering of MFCC centroids |
| Convergence speed | < 20 trials to threshold | Trials to reach similarity > 0.7 |
| Label association | > 70% correct production | Word -> sound retrieval accuracy |

## Execution Order

1. **M9a** — Sound imitation primitive (oscillator + neural readout + similarity reward)
2. **M9b** — Repeatability and convergence (trial runner + SAG prototype refinement)
3. **M9c** — Weak terminal confirmation (combined reward + label attachment)
4. **M9d** — Text association (word-sound pairing + bidirectional SAG edges)

Each phase gates the next. Don't rush to text association before sound production is stable.

## Files Summary

New files:
- `packages/m9_sound/` — M9 sound-first motor learning package
  - `oscillator.py` — Oscillator bank for rhythmic input
  - `substrate.py` — Recurrent acoustic substrate
  - `controller.py` — Neural controller / readout
  - `producer.py` — Acoustic output generation
  - `similarity.py` — Acoustic similarity metrics (MFCC, chroma, mel)
- `packages/sag/acoustic_models.py` — AcousticNode, new edge types
- `scripts/m9_imitation_loop.py` — M9a trial runner
- `scripts/m9_convergence_test.py` — M9b repeatability evaluation
- `scripts/m9_terminal_judge.py` — M9c terminal confirmation
- `scripts/m9_text_association.py` — M9d word-sound pairing
- `docs/m9_acceptance_report.md`

Existing files modified:
- `packages/sag/models.py` — Add AcousticNode, new edge types
- `packages/sag/graph.py` — Handle acoustic node indexing
- `packages/broca/tts.py` — May need acoustic output path (not just TTS)

## Storage & Memory Assessment

### Current State

| Component | Storage | Persistence | M9 Readiness |
|-----------|---------|-------------|--------------|
| `Hippocampus` | In-memory list + JSONL append-only | Partial (traces persist, graph doesn't) | Usable for trial logs |
| `SAG` | Purely in-memory dicts | None | Needs acoustic node type + persistence |
| `OfflineConsolidator` | Background process, writes SAG nodes | `persist_path` exists but unclear if wired | Not needed for M9a/b |
| `faiss-cpu` | Installed, unused | N/A | Candidate for vector search in M9c+ |
| `chromadb` | Installed, unused | N/A | Candidate for prototype storage in M9c+ |
| `data/` | Empty (gitignored) | N/A | Checkpoint space for M9 |

### M9a/b Data Volume (small — no persistence needed yet)

- Per trial: 1 target sound (~1s) + 1 produced sound (~1s) + features (~1KB)
- Per prototype: MFCC centroid (13 floats), pitch (1 float), duration (1 float), count (int), avg_similarity (float) — ~100 bytes
- 100 trials × 5 prototypes = ~500KB total

This fits comfortably in memory with JSONL checkpoint. No need for a database yet.

### M9c+ Data Volume (persistent storage needed)

- Label associations (word → prototype) must survive restarts
- Production history (per-prototype similarity time series) grows without bound
- Text association requires durable bidirectional SAG edges

**Decision:** Defer persistence to M9c. Use in-memory + JSONL checkpoint for M9a/b. When text association requires durable storage, add:
- `SemanticAbstractionGraph.persist()` / `.load()` — serialize nodes + edges to JSON
- `faiss-cpu` index for MFCC centroid similarity search (nearest prototype lookup)
- `chromadb` alternative if metadata-rich queries are needed

### Storage-Related Open Risks

- **SAG storage:** Storing raw audio in SAG nodes doesn't scale. Need compressed representations (MFCC centroids, embeddings).
- **Vector search:** M9b+ needs nearest-prototype lookup. Current SAG has token-based indexing only. Need vector index (faiss) for acoustic similarity queries.
- **Production history growth:** Per-prototype similarity time series grows with each trial. Need bounded storage (circular buffer or summary stats only).

## Open Risks (General)

- **Oscillator design:** Simple sine waves may not produce rich enough sound space. May need FM synthesis or wavetable.
- **Similarity metric:** MFCC distance may not capture perceptual similarity well. May need learned similarity (e.g., using the ECAPA embeddings from M4).
- **Credit assignment:** The recurrent substrate makes it hard to attribute which timestep contributed to the reward. May need temporal credit assignment (e.g., REINFORCE with baseline).
- **Text association timing:** Pivoting to text too early could collapse the sound diversity. Need a clear gate criterion.

## Relationship to M8

M8 (Track 1) built the speaking policy — deciding *when* to speak, *what* action to take. M9 (Track 2) builds the sound production — *how* to produce the actual audio. They compose:

```
M8 policy decides "speak" -> M9 produces the actual sound -> TTS renders it
```

M8's `SpeakingPolicy.decide()` outputs an action. In a future integration, the `speak` action would trigger M9's `producer` to generate the acoustic output, which then feeds into the streaming runtime's TTS pipeline.

For now, M9 is standalone — it doesn't depend on M8's policy decisions, and M8 doesn't depend on M9's sound production. They're independent tracks that compose later.
