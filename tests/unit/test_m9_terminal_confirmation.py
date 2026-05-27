"""Tests for M9c weak terminal confirmation."""

from __future__ import annotations

from m9_sound.similarity import SimilarityResult
from m9_sound.terminal_confirmation import (
    WeakTerminalJudge,
    attach_label_if_passed,
    combine_reward,
)
from sag.acoustic_models import AcousticNode


class TestWeakTerminalJudge:
    def test_passes_at_threshold(self):
        judge = WeakTerminalJudge(threshold=0.55)
        similarity = SimilarityResult(
            overall=0.56,
            mfcc_distance=0.1,
            pitch_distance=0.1,
            duration_distance=0.1,
            loudness_distance=0.1,
        )
        verdict = judge.judge(similarity, label="440_sine")
        assert verdict.passed is True
        assert verdict.reward == 1.0
        assert verdict.label == "440_sine"

    def test_fails_below_threshold(self):
        judge = WeakTerminalJudge(threshold=0.55)
        similarity = SimilarityResult(
            overall=0.51,
            mfcc_distance=0.1,
            pitch_distance=0.1,
            duration_distance=0.1,
            loudness_distance=0.1,
        )
        verdict = judge.judge(similarity, label="440_sine")
        assert verdict.passed is False
        assert verdict.reward < 0.0
        assert "< threshold" in verdict.reason


class TestCombinedReward:
    def test_combined_reward_blends_signals(self):
        similarity = SimilarityResult(
            overall=0.6,
            mfcc_distance=0.1,
            pitch_distance=0.1,
            duration_distance=0.1,
            loudness_distance=0.1,
        )
        judge = WeakTerminalJudge(threshold=0.55)
        verdict = judge.judge(similarity, label="440_sine")
        reward = combine_reward(similarity, verdict, similarity_weight=0.7, terminal_weight=0.3)
        assert reward.value > similarity.overall


class TestLabelAttachment:
    def test_attach_label_when_passed(self):
        node = AcousticNode.from_audio_summary(
            id="proto_440_sine",
            core="Prototype",
            grounding=["440hz"],
            label=None,
            source_trials=["target"],
        )
        judge = WeakTerminalJudge(threshold=0.55)
        similarity = SimilarityResult(
            overall=0.6,
            mfcc_distance=0.1,
            pitch_distance=0.1,
            duration_distance=0.1,
            loudness_distance=0.1,
        )
        verdict = judge.judge(similarity, label="440_sine")
        updated = attach_label_if_passed(node, verdict)
        assert updated.label == "440_sine"
        assert "440_sine" in updated.grounding
        assert updated.context["terminal_confirmed"] is True

    def test_no_label_on_failure(self):
        node = AcousticNode.from_audio_summary(
            id="proto_440_sine",
            core="Prototype",
            grounding=["440hz"],
            label=None,
            source_trials=["target"],
        )
        judge = WeakTerminalJudge(threshold=0.55)
        similarity = SimilarityResult(
            overall=0.5,
            mfcc_distance=0.1,
            pitch_distance=0.1,
            duration_distance=0.1,
            loudness_distance=0.1,
        )
        verdict = judge.judge(similarity, label="440_sine")
        updated = attach_label_if_passed(node, verdict)
        assert updated.label is None
        assert updated.grounding == ["440hz"]
