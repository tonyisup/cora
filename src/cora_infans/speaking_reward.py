"""Reward shaping for simulation-only speaking-policy episodes.

The terms mirror og-cora M8's deterministic-first reward surface. This module is
pure scoring: no live audio, no TTS, and no controller side effects.
"""

from __future__ import annotations

from .speaking_policy import ConversationFeatures, SpeakingAction


def _add(terms: dict[str, float], name: str, value: float) -> None:
    if value != 0.0:
        terms[name] = round(value, 6)


def score_speaking_action(
    *,
    action: SpeakingAction,
    features: ConversationFeatures,
) -> dict[str, object]:
    """Score one policy action against current conversation features."""
    terms: dict[str, float] = {}

    needs_repair = features.was_collided or features.was_interrupted
    if action == "yield" and features.is_agent_speaking and needs_repair:
        _add(terms, "successful_yield", 1.0)
    elif action == "repair" and needs_repair:
        _add(terms, "successful_repair", 1.5)
    elif action != "repair" and needs_repair and not features.is_user_speaking:
        # A non-repair action while a collision/interrupt is unresolved is a
        # missed repair — unless the user is actively speaking, in which case
        # waiting (yielding the floor) is the correct policy, not a failure.
        _add(terms, "missed_repair", -1.5)

    if action == "clarify" and features.needs_clarification:
        _add(terms, "successful_clarify", 1.0)
    elif action == "clarify" and not features.needs_clarification:
        _add(terms, "unnecessary_clarify", -0.5)

    if action == "speak" and features.queued_output and 0 < features.silence_ms <= 2000:
        _add(terms, "smooth_turn_transition", 0.5)

    if action != "speak" and features.queued_output and features.silence_ms > 3000:
        stalled_seconds = (features.silence_ms - 3000) / 1000.0
        _add(terms, "stalled_conversation", -0.2 * stalled_seconds)

    total = round(sum(terms.values()), 6)
    return {"total": total, "terms": terms}
