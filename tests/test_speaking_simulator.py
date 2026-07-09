"""Offline scripted conversation simulator for the M8-infans policy ladder."""

import json

from cora_infans.conversation_tracker import ConversationEvent
from cora_infans.speaking_policy import HeuristicSpeakingPolicy
from cora_infans.speaking_simulator import ConversationEpisode, run_conversation_episode


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
