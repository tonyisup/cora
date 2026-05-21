#!/usr/bin/env python3
"""M3 dialogue generator (local-first).

This script replays the existing M1 conversation scripts, but replaces the
scripted user turn with a local LLM user agent. It logs both the generated
response and the original scripted reference so we can compare diversity and
quality later.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure sibling imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m1_speaking_policy_sim import CONVERSATION_SCRIPTS  # type: ignore
from m3_user_agent import M3UserAgent, UserAgentPacket, packet_to_dict, result_to_dict


def reference_kind(ref: str) -> str:
    if ref == "__SILENCE__":
        return "silence"
    if ref == "__INTERRUPT__":
        return "interrupt"
    if ref == "__OVERLAP__":
        return "overlap"
    return "reply"


def make_goal(script_id: int) -> str:
    return f"Replay and diversify scripted conversation #{script_id}."


def generate_dialogue(script_id: int, agent: M3UserAgent, max_turns: int | None = None) -> List[Dict[str, Any]]:
    script = CONVERSATION_SCRIPTS[script_id % len(CONVERSATION_SCRIPTS)]
    history: List[Dict[str, str]] = []
    rows: List[Dict[str, Any]] = []

    for turn_index, pair in enumerate(script):
        if max_turns is not None and turn_index >= max_turns:
            break
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        agent_utterance, scripted_user = pair
        agent_utterance = str(agent_utterance)
        scripted_user = str(scripted_user)

        packet = UserAgentPacket(
            scenario_id=script_id,
            scenario_name=f"script_{script_id}",
            turn_index=turn_index,
            history=history[-8:],
            last_assistant_utterance=agent_utterance,
            last_user_utterance=history[-1]["text"] if history and history[-1]["speaker"] == "user" else "",
            goal=make_goal(script_id),
            style="natural",
            allow_silence=True,
            allow_interrupt=True,
            allow_overlap=True,
            seed=script_id * 1000 + turn_index,
        )

        result = agent.generate(packet)
        user_text = result.text
        if result.mode == "silence":
            user_text = ""

        # Update conversation history with the generated turn.
        history.append({"speaker": "assistant", "text": agent_utterance})
        history.append({"speaker": "user", "text": user_text})

        rows.append(
            {
                "scenario_id": script_id,
                "scenario_name": f"script_{script_id}",
                "turn_index": turn_index,
                "agent_utterance": agent_utterance,
                "scripted_user": scripted_user,
                "scripted_kind": reference_kind(scripted_user),
                "input_packet": packet_to_dict(packet),
                "generated": result_to_dict(result),
                "generated_kind": result.mode,
                "generated_text": user_text,
                "history_after": history[-10:],
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dialogue traces with a local LLM user agent")
    parser.add_argument("--output", default="results/m3_dialogues.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="gemma-4-e4b-it-4bit", help="Local OMLX model name")
    parser.add_argument("--timeout", type=float, default=45.0, help="Per-turn timeout in seconds")
    parser.add_argument("--scripts", default="", help="Comma-separated script ids to replay")
    parser.add_argument("--max-scripts", type=int, default=4, help="Maximum number of scripts to replay")
    parser.add_argument("--max-turns", type=int, default=0, help="Optional per-script turn cap (0 = no cap)")
    args = parser.parse_args()

    if args.scripts.strip():
        script_ids = [int(x.strip()) for x in args.scripts.split(",") if x.strip()]
    else:
        script_ids = list(range(min(args.max_scripts, len(CONVERSATION_SCRIPTS))))

    agent = M3UserAgent(model=args.model, timeout_s=args.timeout)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with out_path.open("w") as f:
        for script_id in script_ids:
            rows = generate_dialogue(script_id, agent, max_turns=args.max_turns or None)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += len(rows)
            print(f"script_{script_id}: {len(rows)} turns")

    print(f"Saved {total_rows} rows to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
