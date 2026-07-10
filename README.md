# Cora-Infans

Cora-Infans is an audio-centric research system for testing local auditory
learning, bounded event memory, active attention, and analysis by synthesis.

The governing research contract is [VISION.md](VISION.md). It distinguishes
implemented mechanisms from measured and validated capabilities.

External corpus use follows [docs/DATA_POLICY.md](docs/DATA_POLICY.md): Cora is
a personal, non-commercial research project, and third-party audio stays outside
the repository.

## Current evidence status

| Stage | Status | Honest interpretation |
|---|---|---|
| Reproducible substrate | Implemented | Synthetic/live contracts, deterministic reports, and tests exist |
| Real-audio evaluation | First replay passed | WAV and prior-art captured NumPy PCM are measured directly with dev/held-out provenance; developmental-speech breadth is still open |
| Parentese-relevant saliency | Cross-fieldsite gate failed | The Beijing result did not transport to Krakow or Mbendjele; song activated the pitch gate on 76–92% of frames |
| Local predictive hierarchy | Safe but not superior | The redesigned local expert mixture passed fresh-fieldsite regret limits but won 0/8 clips; the phase prior remains the only positive predictor evidence |
| Workspace binding | Synthetic ablation passed | Matching self-audio identity plus phase improves synthetic self-echo handling without suppressing a legitimate overlap; real-audio validation remains open |
| Analysis by synthesis | Clean synthetic gate passed | Five vowels round-trip across multiple F0 values and 16/48 kHz; recorded-speaker validation remains open |
| Efficiency | Host boundary measured on hardware | Core microbenchmarks and native XVF3800 capture-call-to-workspace latency are reported; device-onset timestamps, live CPU/memory, and energy remain excluded |

## Hardware boundary

The XVF3800 board has four physical microphones and performs on-board DSP. The
current macOS/CoreAudio path exposes two fused USB channels. Cora therefore does
not claim raw four-channel host-side learning or beamforming.

Firmware speech-classified source direction can drive beam steering when it is
finite. Two-channel energy imbalance is otherwise only a coarse left/right cue.

## Install and test

```bash
python3 -m pip install -e .
python3 -m pytest tests -q
```

The installed entry point is `cora-infans`; `python3 -m cora_infans` is the
portable fallback.

## Research commands

Describe implemented and planned surfaces:

```bash
python3 -m cora_infans --describe
```

Run deterministic pipeline scenarios:

```bash
python3 -m cora_infans --simulate-frame
python3 -m cora_infans --simulate-sequence --family gap
```

Evaluate real uncompressed WAV clips (integer PCM or IEEE float):

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
  --frame-ms 250 \
  --saliency-threshold 0.02 \
  --score-mode peak_rms
```

The label CSV maps annotations to existing WAV files. `split` is required for
development/held-out comparison:

```csv
clip_id,start_s,end_s,label,split
parentese_dev_01,0.20,1.10,parentese,dev
parentese_held_01,0.15,0.95,parentese,held_out
```

Annotations are used only as ground truth. WAV samples—not labels—produce the
evaluated observations.

Tune transparent saliency baselines on `dev` and evaluate once on `held_out`:

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
  --compare-baselines \
  --positive-label infant_directed_speech
```

This compares peak RMS, spatial contrast, pitch, pitch confidence, spectral
centroid, and spectral flux. Every threshold is selected on development clips
and frozen for held-out evaluation.

### External developmental-speech corpus

The first corpus is the Music Lab's [Human vocalization corpus](https://zenodo.org/records/5525161)
(DOI `10.5281/zenodo.5525161`). It contains infant-directed and adult-directed
speech/song from 21 societies and is licensed CC BY-NC-SA 4.0. The edited audio
archive is approximately 5.8 GB, so Cora does not download it automatically.

After intentionally downloading and extracting the edited archive, build a
participant-grouped manifest:

```bash
python3 -m cora_infans --prepare-ids-manifest \
  --root /path/to/IDS-corpus-edited \
  --out /path/to/IDS-corpus-edited/labels.csv
```

The importer maps filename condition `B` to `infant_directed_speech`, condition
`D` to `adult_directed_speech`, and assigns all clips from one participant to
the same development or held-out split. Song conditions are excluded unless
`--include-song` is requested.

The first 12-clip participant-disjoint pilot is recorded in
`docs/evidence/2026-07-10-ids-pilot.json`. It is deliberately not a promotion:
the best feature's greater recall came with an unacceptable false-positive
increase. Pilot participants are excluded from confirmatory reuse.

The criterion was then frozen and evaluated on 12 fresh Beijing clips. Pitch
confidence passed narrowly: held-out F1 `0.575` versus RMS `0.510`, with FPR
`0.594` versus `0.547`. The auditable result is in
`docs/evidence/2026-07-10-ids-confirmation.json`. It supports a one-fieldsite,
participant-disjoint result—not a robust universal parentese detector.

That limitation is now measured rather than hypothetical. A preregistered test
froze the Beijing thresholds and added paired song controls for unseen Krakow
and Mbendjele BaYaka speakers. Both fieldsites failed. See
`docs/evidence/2026-07-10-cross-fieldsite-result.json`.

Evaluate prequential prediction on held-out acoustic observations:

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
    --evaluate-prediction dominant_frequency_hz
```

Evaluate the bounded online learner's safety separately from superiority:

```bash
python3 -m cora_infans --eval-corpus \
  --root ./corpus \
  --label-csv ./corpus/labels.csv \
  --evaluate-online-safety peak_rms,pitch_confidence,spectral_centroid_hz,spectral_flux
```

The safety report compares every clip with persistence and EMA, reporting mean
and worst RMSE regret separately from strict win rate. Passing safety does not
count as predictive advantage.

Replay retained Corollary real-capture arrays without copying them:

```bash
python3 -m cora_infans --prepare-corollary-manifest \
  --root /path/to/corollary/capture-family \
  --out /tmp/cora-corollary.csv

python3 -m cora_infans --eval-corpus \
  --root /path/to/corollary/capture-family \
  --label-csv /tmp/cora-corollary.csv \
  --input-mode recorded_npy \
  --compare-baselines \
  --positive-label mixed_voice
```

The importer uses seeds 0/1 for development and seed 2 for held-out evaluation
by default. This evidence is real captured audio but belongs to a repeated,
aligned self/world experiment, not an infant-directed-speech corpus.

Measure the clean synthetic vowel confusion matrix:

```bash
python3 -m cora_infans --evaluate-vowel-loop
```

Synthesize and inspect one vowel:

```bash
python3 -m cora_infans --synthesize-vowel a --out /tmp/a.wav --analyze
```

Benchmark only the in-process event pipeline:

```bash
python3 -m cora_infans --benchmark-core --iterations 1000 --frame-ms 25
```

The benchmark report lists all excluded latency sources and does not measure
energy.

Run the workspace strategy ablation:

```bash
python3 -m cora_infans --evaluate-workspace-ablation
```

The ablation compares flat, fixed-window, and theta-gamma handling on aligned
self echo and legitimate world/motor overlap. Scheduling suppression requires
verified self-audio identity; temporal coincidence by itself is insufficient.

## Live hardware commands

```bash
python3 -m cora_infans --capture-live --device :0 --duration-ms 250
python3 -m cora_infans --listen-live --duration-ms 2000 --with-attention
python3 -m cora_infans --listen-live --duration-ms 1000 --frame-ms 25 --capture-backend native
python3 -m cora_infans --snapshot-firmware
python3 -m cora_infans --tune-raw-linear
```

Every listener report includes host-monotonic capture-call, pipeline-decision,
deadline-miss, and real-time-factor metrics. The boundary is the start of
`capture_frame()` to the workspace decision—not ADC sample onset or an
audio-backend timestamp.

When the native backend provides a nominal input-latency value, the report also
adds it to the host call measurements as an explicitly estimated
ADC-to-workspace distribution. It is not presented as a per-buffer hardware
timestamp.

The CLI listener prefers a native PortAudio/CoreAudio stream through
`sounddevice`, using the same substrate validated by the Corollary prior-art
project. `--capture-backend ffmpeg` is an explicit real-audio fallback. Auto
fallback reasons are included in reports; neither path substitutes synthetic
audio. Reports separate cold-start latency from steady state and expose both
strict and jitter-tolerant deadline misses.

The bounded speaking and self-subtraction paths require explicit real-I/O
authorization. They must not be treated as evidence of conversational or
developmental language learning.

## Important limitations

- Licensed developmental-speech audio has been evaluated from external
  temporary subsets. A same-fieldsite gate passed narrowly but failed on both
  unseen fieldsites; the CC BY-NC-SA audio is not checked in.
- The current pitch-confidence threshold confuses song with IDS and cannot be
  called a cross-cultural parentese detector.
- The current saliency gate is a loudness baseline, not a parentese detector.
- The online local predictor is now conservatively bounded and passed its fresh
  regret gate, but it still has zero real-clip wins. Only the onset-phase
  template has a held-out advantage.
- Theta/gamma scheduling is an event-timing model, not measured neural dynamics.
- The vowel table uses adult reference targets and clean synthetic success does
  not imply recorded infant/adult speech performance.
- No end-to-end latency, CPU utilization, or energy comparison has passed the
  efficiency stage gate.

## Main modules

- `io/wav_frames.py` — real uncompressed WAV decoding and measured features
- `io/recorded_frames.py` — retained real-capture arrays and prior-art manifests
- `io/offline_eval.py` — confusion metrics and dev/held-out baseline comparison
- `predictive_learning.py` — bounded local hierarchy and baseline evaluation
- `io/motor_plan.py` — source-filter synthesis, LPC perception, and vowel gate
- `benchmark.py` — explicit-boundary event-pipeline microbenchmark
- `pipeline.py` — saliency, predictive state, and workspace admission
- `theta_gamma.py` — deterministic bounded temporal scheduler
- `io/live_mic.py` — real XVF3800/CoreAudio adapter
- `io/active_attention.py` — pure beam-steering governor
- `io/hardware_tuner.py` — verified USB DSP controls

Historical experiment logs and retractions are retained under `docs/`. The
current claim boundary is defined by `VISION.md` and this README.
