# M10 Data Layout for Persistent Consonant Learning

Status: proposed

This document sketches the minimal on-disk layout for M10 so we can persist captures, segment labels, reviews, and model artifacts without introducing a database too early.

## Guiding rule

Keep the filesystem as the source of truth first:
- raw WAVs stay immutable
- manifests are append-only
- corrections are new records, not in-place edits
- checkpoints are versioned by name

Move to SQLite only if JSONL and folders become painful to query or edit.

## Root layout

Use a dedicated M10 subtree:

- `data/m10/raw/` — original recordings
- `data/m10/segments/` — cropped consonant / vowel segments
- `data/m10/manifests/` — JSONL metadata indexes
- `data/m10/reviews/` — human corrections and review notes
- `data/m10/checkpoints/` — model artifacts and eval outputs
- `data/m10/runs/` — per-run demo / smoke / eval artifacts

## Per-capture layout

Raw sessions should be grouped by speaker and session:

- `data/m10/raw/<speaker>/<session_id>/<timestamp>_<prompt>.wav`
- `data/m10/raw/<speaker>/<session_id>/session.json`

Suggested session JSON fields:
- `session_id`
- `speaker`
- `prompt`
- `created_at`
- `input_device`
- `output_device`
- `device_profile`
- `device_gain`
- `sample_rate`
- `notes`

## Per-segment layout

Segments are the unit used for training and evaluation:

- `data/m10/segments/<speaker>/<session_id>/<capture_id>_<label>_<idx>.wav`

Suggested labels for the first consonant slice:
- stops: `p`, `t`, `k`
- fricatives: `s`, `f`, `sh`
- nasals: `m`, `n`
- non-speech / separator: `_`

## Persistent manifests

Keep three JSONL indexes:

- `data/m10/manifests/captures.jsonl`
- `data/m10/manifests/segments.jsonl`
- `data/m10/manifests/reviews.jsonl`

### Capture manifest row

Suggested fields:
- `capture_id`
- `speaker`
- `session_id`
- `prompt`
- `target_token`
- `wav_path`
- `sample_rate`
- `duration_s`
- `device_profile`
- `device_gain`
- `created_at`
- `model_version`
- `notes`

### Segment manifest row

Suggested fields:
- `segment_id`
- `capture_id`
- `speaker`
- `token`
- `label`
- `start_frame`
- `end_frame`
- `wav_path`
- `confidence`
- `source`  # auto / manual / corrected
- `created_at`
- `segmenter_version`

### Review manifest row

Suggested fields:
- `segment_id`
- `old_label`
- `new_label`
- `reviewer`
- `reason`
- `created_at`

## Run artifacts

Every demo or evaluation run should land in its own directory:

- `data/m10/runs/<run_id>/report.json`
- `data/m10/runs/<run_id>/rendered.wav`
- `data/m10/runs/<run_id>/captured.wav`
- `data/m10/runs/<run_id>/console.log` if needed

This makes it easy to compare runs and keep the listening artifacts together.

## Checkpoints

Version model outputs explicitly:

- `data/m10/checkpoints/m10_consonant_v1.json`
- `data/m10/checkpoints/m10_consonant_v1_eval.json`

If the training recipe changes materially, use a new version suffix instead of overwriting the old file.

## Suggested workflow

1. Record a consonant-bearing phrase.
2. Save the raw WAV and session JSON.
3. Auto-segment the capture.
4. Append segment rows to `segments.jsonl`.
5. Review obvious mistakes and append correction rows.
6. Train from the manifests.
7. Save a versioned checkpoint and run report.

## When to move to SQLite

Upgrade to SQLite if you need:
- cross-session queries by speaker / token / confidence
- label editing over many sessions
- fast search over a growing corpus
- a more structured review workflow

Until then, JSONL is enough and keeps the system simple.
