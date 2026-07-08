"""Tests for the offline corpus + evaluation harness (no hardware)."""

import csv

from cora_infans.io.offline_eval import (
    Annotation,
    CorpusClip,
    CorpusLoader,
    OfflineEvalHarness,
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
