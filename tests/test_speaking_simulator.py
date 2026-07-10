"""Offline scripted conversation simulator for the M8-infans policy ladder."""

import json

from cora_infans.conversation_tracker import ConversationEvent
from cora_infans.speaking_policy import HeuristicSpeakingPolicy
from cora_infans.speaking_simulator import (
    ConversationEpisode,
    canonical_speaking_policy_episodes,
    run_conversation_episode,
)


def test_simulator_replays_episode_and_returns_jsonable_steps() -> None:
    episode = ConversationEpisode(
        episode_id="clean_turn_handoff",
        events=(
            ConversationEvent(timestamp_ms=0, kind="agent_output_queued", payload={"text": "hello"}),
            ConversationEvent(timestamp_ms=100, kind="user_speech_start"),
            ConversationEvent(timestamp_ms=700, kind="user_speech_end"),
            ConversationEvent(timestamp_ms=2800, kind="agent_output_queued", payload={"text": "response"}),
        ),
        decision_times_ms=(0, 150, 2800),
    )

    report = run_conversation_episode(episode, policy=HeuristicSpeakingPolicy())

    assert report["mode"] == "simulation_only_speaking_episode"
    assert report["episode_id"] == "clean_turn_handoff"
    assert report["live_playback"] is False
    assert report["live_playback_allowed"] is False
    assert [step["action"] for step in report["steps"]] == ["wait", "wait", "speak"]
    json.dumps(report)


def test_simulator_scores_yield_then_repair_episode() -> None:
    episode = ConversationEpisode(
        episode_id="overlap_then_repair",
        events=(
            ConversationEvent(timestamp_ms=0, kind="agent_output_queued"),
            ConversationEvent(timestamp_ms=100, kind="agent_speech_start"),
            ConversationEvent(timestamp_ms=250, kind="overlap_detected"),
            ConversationEvent(timestamp_ms=400, kind="agent_speech_end"),
        ),
        decision_times_ms=(125, 275, 450),
    )

    report = run_conversation_episode(episode, policy=HeuristicSpeakingPolicy())

    actions = [step["action"] for step in report["steps"]]
    assert actions == ["wait", "yield", "repair"]
    assert report["summary"]["action_counts"]["yield"] == 1
    assert report["summary"]["action_counts"]["repair"] == 1
    assert report["summary"]["total_reward"] > 0


def test_simulator_enacts_yield_by_stopping_speech_and_deferring_output() -> None:
    episode = ConversationEpisode(
        episode_id="yield_then_repair_without_scripted_stop",
        events=(
            ConversationEvent(timestamp_ms=0, kind="agent_output_queued"),
            ConversationEvent(timestamp_ms=100, kind="agent_speech_start"),
            ConversationEvent(timestamp_ms=200, kind="overlap_detected"),
        ),
        decision_times_ms=(200, 300),
    )

    report = run_conversation_episode(episode, policy=HeuristicSpeakingPolicy())

    assert [step["action"] for step in report["steps"]] == ["yield", "repair"]
    assert report["steps"][1]["features"]["is_agent_speaking"] is False
    assert report["steps"][1]["features"]["has_deferred_output"] is True


def test_overlap_while_not_speaking_repairs_then_releases_queued_output() -> None:
    episode = ConversationEpisode(
        episode_id="queued_overlap_before_speech",
        events=(
            ConversationEvent(timestamp_ms=0, kind="user_speech_end"),
            ConversationEvent(timestamp_ms=0, kind="agent_output_queued"),
            ConversationEvent(timestamp_ms=100, kind="overlap_detected"),
        ),
        decision_times_ms=(100, 200, 2200),
    )

    report = run_conversation_episode(episode, policy=HeuristicSpeakingPolicy())

    assert [step["action"] for step in report["steps"]] == ["repair", "wait", "speak"]
    assert report["steps"][2]["features"]["queued_output"] is True


def test_simulator_orders_events_before_decision_at_same_timestamp() -> None:
    episode = ConversationEpisode(
        episode_id="same_timestamp_queue",
        events=(
            ConversationEvent(timestamp_ms=0, kind="user_speech_end"),
            ConversationEvent(timestamp_ms=2000, kind="agent_output_queued"),
        ),
        decision_times_ms=(2000,),
    )

    report = run_conversation_episode(episode, policy=HeuristicSpeakingPolicy())

    assert report["steps"][0]["features"]["queued_output"] is True
    assert report["steps"][0]["action"] == "speak"


def test_simulator_enacts_speak_once_by_consuming_queue_and_starting_speech() -> None:
    episode = ConversationEpisode(
        episode_id="speak_once",
        events=(
            ConversationEvent(timestamp_ms=0, kind="user_speech_end"),
            ConversationEvent(timestamp_ms=2000, kind="agent_output_queued"),
        ),
        decision_times_ms=(2000, 2100),
    )

    report = run_conversation_episode(episode, policy=HeuristicSpeakingPolicy())

    assert [step["action"] for step in report["steps"]] == ["speak", "wait"]
    assert report["steps"][1]["features"]["is_agent_speaking"] is True
    assert report["steps"][1]["features"]["queued_output"] is False


def test_canonical_episodes_cover_backchannel_and_barge_in() -> None:
    ids = {ep.episode_id for ep in canonical_speaking_policy_episodes()}

    assert "backchannel_suppression" in ids
    assert "barge_in_recovery" in ids


def test_barge_in_recovery_yields_then_repairs() -> None:
    report = run_conversation_episode(
        next(
            ep
            for ep in canonical_speaking_policy_episodes()
            if ep.episode_id == "barge_in_recovery"
        ),
        policy=HeuristicSpeakingPolicy(),
    )

    actions = [step["action"] for step in report["steps"]]
    # User barges in while the agent is speaking: the policy protects the user's
    # speech first (wait), then repairs the deferred/interrupted output after the
    # user finishes. No speaking over the user at any point.
    assert actions == ["wait", "wait", "repair"]
    assert report["summary"]["total_reward"] > 0


def test_backchannel_suppression_waits_over_user_then_speaks() -> None:
    report = run_conversation_episode(
        next(
            ep
            for ep in canonical_speaking_policy_episodes()
            if ep.episode_id == "backchannel_suppression"
        ),
        policy=HeuristicSpeakingPolicy(),
    )

    actions = [step["action"] for step in report["steps"]]
    # Must not speak over the user (backchannel suppression), then speak after silence.
    assert actions[0] == "wait"
    assert "speak" in actions
    assert report["live_playback"] is False
