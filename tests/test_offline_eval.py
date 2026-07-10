"""Tests for the offline corpus + evaluation harness (no hardware)."""

import csv

from cora_infans.io.offline_eval import (
    Annotation,
    CorpusClip,
    CorpusLoader,
    FrameObservation,
    OfflineEvalHarness,
    compare_saliency_baselines,
)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def test_corpus_loader_reads_label_csv(tmp_path):
    wav = tmp_path / "clip01.wav"
    wav.write_bytes(b"RIFF")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            ["clip_id", "start_s", "end_s", "label"],
            ["clip01", "0.0", "1.0", "salient_speech"],
            ["clip01", "1.0", "2.0", "non_speech"],
        ],
    )

    loader = CorpusLoader(root=tmp_path)
    clips = loader.load_labels(csv_path)

    assert len(clips) == 1
    clip = clips[0]
    assert clip.clip_id == "clip01"
    assert clip.wav_path == wav
    assert len(clip.annotations) == 2
    assert clip.annotations[0] == Annotation(0.0, 1.0, "salient_speech")
    assert clip.annotations[1] == Annotation(1.0, 2.0, "non_speech")
    assert clip.split == "unspecified"


def test_corpus_loader_records_development_and_held_out_splits(tmp_path):
    (tmp_path / "dev01.wav").write_bytes(b"RIFF")
    (tmp_path / "held01.wav").write_bytes(b"RIFF")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            ["clip_id", "start_s", "end_s", "label", "split"],
            ["dev01", "0.0", "0.5", "speech", "dev"],
            ["held01", "0.0", "0.5", "speech", "held_out"],
        ],
    )

    clips = CorpusLoader(root=tmp_path).load_labels(csv_path)

    assert [(clip.clip_id, clip.split) for clip in clips] == [
        ("dev01", "dev"),
        ("held01", "held_out"),
    ]


def test_corpus_loader_uses_manifest_source_path_for_nested_audio(tmp_path):
    nested = tmp_path / "edited" / "site"
    nested.mkdir(parents=True)
    wav = nested / "USD01B.wav"
    wav.write_bytes(b"RIFF")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            ["clip_id", "start_s", "end_s", "label", "source_path"],
            ["USD01B", "0.0", "1.0", "speech", "edited/site/USD01B.wav"],
        ],
    )

    clip = CorpusLoader(root=tmp_path).load_labels(csv_path)[0]

    assert clip.wav_path == wav


def test_corpus_loader_rejects_nonmatching_wav(tmp_path):
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [["clip_id", "start_s", "end_s", "label"], ["clip01", "0.0", "1.0", "x"]],
    )

    loader = CorpusLoader(root=tmp_path)
    clips = loader.load_labels(csv_path)
    assert clips == []


def test_harness_scores_frames_against_annotations(tmp_path):
    wav = tmp_path / "clip01.wav"
    wav.write_bytes(b"RIFF")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            ["clip_id", "start_s", "end_s", "label"],
            ["clip01", "0.0", "1.0", "salient_speech"],
            ["clip01", "1.0", "2.0", "non_speech"],
        ],
    )
    clip = CorpusLoader(root=tmp_path).load_labels(csv_path)[0]

    def frames_for(clip, frame_ms):
        # 0-1000ms salient, 1000-2000ms quiet
        return [
            (0, [0.05, 0.05, 0.05, 0.8]),
            (500, [0.04, 0.04, 0.04, 0.7]),
            (1500, [0.02, 0.02, 0.02, 0.02]),
        ]

    harness = OfflineEvalHarness(saliency_threshold=0.7, frame_ms=500)
    report = harness.evaluate_clip(clip, frames_for)

    # Both labeled intervals are ground-truth events. Frames 0/500 fall inside
    # salient_speech and cross threshold -> matched. Frame 1500 is inside the
    # non_speech (event) interval but is quiet -> missed event (false negative).
    assert report.clip_id == "clip01"
    assert report.total_frames == 3
    assert report.salient_frames == 2
    assert report.matched_salient_frames == 2
    assert report.correct_frames == 2
    assert report.accuracy == 2 / 3
    assert report.label_frame_counts == {"salient_speech": 2, "non_speech": 1}


def test_harness_detects_missed_and_false_salient(tmp_path):
    wav = tmp_path / "clip01.wav"
    wav.write_bytes(b"RIFF")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [["clip_id", "start_s", "end_s", "label"], ["clip01", "0.0", "1.0", "salient_speech"]],
    )
    clip = CorpusLoader(root=tmp_path).load_labels(csv_path)[0]

    def frames_for(clip, frame_ms):
        return [
            (0, [0.02, 0.02, 0.02, 0.02]),   # below threshold: missed event
            (1500, [0.05, 0.05, 0.05, 0.9]),  # above threshold but no label: false event
        ]

    harness = OfflineEvalHarness(saliency_threshold=0.7, frame_ms=500)
    report = harness.evaluate_clip(clip, frames_for)

    # Frame 0: quiet inside event label -> missed. Frame 1500: salient but no
    # label (ground-truth silence) -> false event. Both incorrect.
    assert report.accuracy == 0.0
    assert report.correct_frames == 0
    assert report.matched_salient_frames == 0
    assert report.label_frame_counts == {"salient_speech": 1}


def test_harness_can_treat_annotated_control_labels_as_negative(tmp_path):
    clip = CorpusClip(
        clip_id="clip",
        wav_path=tmp_path / "clip.wav",
        annotations=(
            Annotation(0.0, 0.25, "infant_directed_speech"),
            Annotation(0.25, 0.5, "adult_directed_speech"),
        ),
    )
    frames = [
        FrameObservation(0, [0.2], source="pcm_wav"),
        FrameObservation(250, [0.2], source="pcm_wav"),
    ]

    report = OfflineEvalHarness(
        saliency_threshold=0.1,
        frame_ms=250,
        positive_labels={"infant_directed_speech"},
    ).evaluate_clip(clip, lambda clip, frame_ms: frames)

    assert report.true_positive_frames == 1
    assert report.false_positive_frames == 1
    assert report.positive_labels == ["infant_directed_speech"]


def test_harness_aggregates_over_corpus(tmp_path):
    wav = tmp_path / "clip01.wav"
    wav.write_bytes(b"RIFF")
    csv_path = tmp_path / "labels.csv"
    _write_csv(
        csv_path,
        [
            ["clip_id", "start_s", "end_s", "label"],
            ["clip01", "0.0", "1.0", "salient_speech"],
            ["clip01", "1.0", "2.0", "non_speech"],
        ],
    )
    clip = CorpusLoader(root=tmp_path).load_labels(csv_path)[0]

    def frames_for(clip, frame_ms):
        return [
            (0, [0.05, 0.05, 0.05, 0.8]),
            (500, [0.04, 0.04, 0.04, 0.7]),
            (1500, [0.02, 0.02, 0.02, 0.02]),
        ]

    harness = OfflineEvalHarness(saliency_threshold=0.7, frame_ms=500)
    corpus_report = harness.evaluate_corpus([clip], frames_for)

    assert corpus_report["clip_count"] == 1
    assert corpus_report["total_frames"] == 3
    assert corpus_report["correct_frames"] == 2
    assert corpus_report["accuracy"] == 2 / 3
    assert corpus_report["per_clip"][0]["clip_id"] == "clip01"


def test_baseline_threshold_is_selected_on_dev_and_frozen_for_held_out(tmp_path):
    dev = CorpusClip(
        clip_id="dev",
        wav_path=tmp_path / "dev.wav",
        annotations=(Annotation(0.0, 0.25, "speech"),),
        split="dev",
    )
    held = CorpusClip(
        clip_id="held",
        wav_path=tmp_path / "held.wav",
        annotations=(Annotation(0.0, 0.25, "speech"),),
        split="held_out",
    )
    frames = {
        "dev": [
            FrameObservation(0, [0.2, 0.2], source="pcm_wav"),
            FrameObservation(250, [0.01, 0.01], source="pcm_wav"),
        ],
        "held": [
            FrameObservation(0, [0.15, 0.15], source="pcm_wav"),
            FrameObservation(250, [0.01, 0.01], source="pcm_wav"),
        ],
    }

    report = compare_saliency_baselines(
        [dev, held],
        lambda clip, frame_ms: frames[clip.clip_id],
        frame_ms=250,
    )

    peak = report["baselines"]["peak_rms"]
    assert report["evidence_kind"] == "real_audio"
    assert peak["selected_threshold"] == 0.2
    assert peak["dev"]["f1"] == 1.0
    assert peak["held_out"]["f1"] == 0.0


def test_feature_baseline_is_tuned_on_dev_for_target_label(tmp_path):
    annotations = (
        Annotation(0.0, 0.25, "infant_directed_speech"),
        Annotation(0.25, 0.5, "adult_directed_speech"),
    )
    dev = CorpusClip("dev", tmp_path / "dev.wav", annotations, "dev")
    held = CorpusClip("held", tmp_path / "held.wav", annotations, "held_out")

    def frames_for(clip, frame_ms):
        pitches = (300.0, 150.0) if clip.split == "dev" else (320.0, 140.0)
        return [
            FrameObservation(
                timestamp_ms=index * 250,
                channel_energies=[0.2],
                features={"pitch_hz": pitch},
                source="pcm_wav",
            )
            for index, pitch in enumerate(pitches)
        ]

    report = compare_saliency_baselines(
        [dev, held],
        frames_for,
        positive_labels={"infant_directed_speech"},
    )

    pitch = report["baselines"]["feature:pitch_hz"]
    assert pitch["selected_threshold"] == 300.0
    assert pitch["dev"]["f1"] == 1.0
    assert pitch["held_out"]["f1"] == 1.0
    assert report["positive_labels"] == ["infant_directed_speech"]
