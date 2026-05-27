"""M9c — Weak terminal confirmation helpers.

Adds a binary correct/incorrect signal on top of acoustic similarity.
This is intentionally weak: the judge can be automated with a threshold,
or replaced with a human terminal verdict later.

M9c responsibilities:
- turn acoustic similarity into a weak terminal signal
- blend acoustic and terminal reward into a combined training signal
- attach simple labels to acoustic prototypes when the judge passes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sag.acoustic_models import AcousticNode
from m9_sound.similarity import SimilarityResult


@dataclass(frozen=True)
class TerminalJudgement:
    """Binary terminal verdict for a produced sound."""

    passed: bool
    threshold: float
    similarity: float
    reward: float
    label: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class CombinedReward:
    """Combined acoustic + terminal reward used for M9c updates."""

    acoustic: float
    terminal: float
    similarity_weight: float
    terminal_weight: float

    @property
    def value(self) -> float:
        return self.similarity_weight * self.acoustic + self.terminal_weight * self.terminal


class WeakTerminalJudge:
    """Threshold-based terminal judge for M9c.

    The judge is weak by design: it turns the continuous acoustic similarity
    score into a binary pass/fail signal that can be used to shape learning.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        pass_reward: float = 1.0,
        fail_reward: float = -0.25,
    ) -> None:
        self.threshold = float(threshold)
        self.pass_reward = float(pass_reward)
        self.fail_reward = float(fail_reward)

    def judge(self, similarity: SimilarityResult, label: Optional[str] = None) -> TerminalJudgement:
        passed = float(similarity.overall) >= self.threshold
        reward = self.pass_reward if passed else self.fail_reward
        reason = (
            f"similarity {similarity.overall:.4f} >= threshold {self.threshold:.4f}"
            if passed
            else f"similarity {similarity.overall:.4f} < threshold {self.threshold:.4f}"
        )
        return TerminalJudgement(
            passed=passed,
            threshold=self.threshold,
            similarity=float(similarity.overall),
            reward=reward,
            label=label,
            reason=reason,
        )


def combine_reward(
    similarity: SimilarityResult,
    terminal: TerminalJudgement,
    *,
    similarity_weight: float = 0.7,
    terminal_weight: float = 0.3,
) -> CombinedReward:
    """Blend acoustic similarity and terminal judgment into one reward."""
    return CombinedReward(
        acoustic=float(similarity.overall),
        terminal=float(terminal.reward),
        similarity_weight=float(similarity_weight),
        terminal_weight=float(terminal_weight),
    )


def attach_label_if_passed(
    node: AcousticNode,
    terminal: TerminalJudgement,
) -> AcousticNode:
    """Attach the label to a prototype only when the judge passes.

    This is the M9c label-association step: it preserves sound prototypes but
    only tags them with labels after a weak terminal confirmation succeeds.
    """
    if not terminal.passed or not terminal.label:
        return node

    grounding = list(node.grounding)
    if terminal.label not in grounding:
        grounding.append(terminal.label)

    context = dict(node.context)
    context["terminal_confirmed"] = True
    context["terminal_threshold"] = terminal.threshold

    return AcousticNode(
        id=node.id,
        core=node.core,
        context=context,
        grounding=grounding,
        prototype_audio=list(node.prototype_audio),
        mfcc_centroid=list(node.mfcc_centroid),
        embedding=list(node.embedding),
        pitch_mean_hz=node.pitch_mean_hz,
        duration_mean_s=node.duration_mean_s,
        loudness_mean_db=node.loudness_mean_db,
        production_count=node.production_count,
        avg_similarity=node.avg_similarity,
        similarity_history=list(node.similarity_history),
        label=terminal.label,
        source_trials=list(node.source_trials),
        feature_version=node.feature_version,
    )
