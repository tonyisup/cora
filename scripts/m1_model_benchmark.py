#!/usr/bin/env python3
"""Benchmark several LLMs on speaking-policy decision prompts.

This script probes a handful of representative turn-taking situations and
records:
- latency per prompt
- raw response text
- parsed ACTION
- action correctness against an expected label
- errors / parse failures

The goal is not to declare a winner from a single run, but to start collecting
comparative data under a fixed prompt set.

Usage:
  .venv/bin/python scripts/m1_model_benchmark.py
  .venv/bin/python scripts/m1_model_benchmark.py --output results/bench.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HERMES_CLI = ["hermes", "chat", "-q", "{prompt}", "--quiet"]


@dataclass(frozen=True)
class Scenario:
    name: str
    expected_action: str
    prompt: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "hermes" or "openrouter"
    model: str
    timeout_s: float


SCENARIOS: List[Scenario] = [
    Scenario(
        name="speak_on_silence",
        expected_action="speak",
        prompt=(
            "You are a speaking policy for a conversational AI agent. Your job is to decide when "
            "the agent should speak, wait, clarify, repair, or yield in a turn-taking conversation.\n\n"
            "Rules:\n"
            "- Speak only when it's the agent's turn (user has finished speaking + brief silence)\n"
            "- Wait while the user is speaking\n"
            "- Yield immediately if the user interrupts the agent mid-sentence\n"
            "- Repair (say \"Sorry, go ahead.\") if the agent and user spoke simultaneously\n"
            "- Clarify (ask user to repeat) if the agent spoke and the user didn't respond\n\n"
            "Current conversation state:\n"
            "- Agent state: listen\n"
            "- Last user utterance: (none)\n"
            "- Last agent utterance: \"Hello! How can I help you today?\"\n"
            "- Silence duration: 400ms\n"
            "- VAD energy: -50.0 dB\n"
            "- Attribution: silence\n"
            "- Has queued utterance: True\n"
            "- Recent history:\n"
            "  Agent: Hello! How can I help you today?\n\n"
            "Available actions: speak, wait, clarify, repair, yield\n\n"
            "What should the agent do next? Respond with:\n"
            "ACTION: <one of: speak, wait, clarify, repair, yield>\n"
            "EXPLANATION: <1-2 sentences explaining why>\n"
            "TEXT: <the exact text to speak, or \"none\">"
        ),
    ),
    Scenario(
        name="yield_on_collision",
        expected_action="yield",
        prompt=(
            "You are a speaking policy for a conversational AI agent. Your job is to decide when "
            "the agent should speak, wait, clarify, repair, or yield in a turn-taking conversation.\n\n"
            "Rules:\n"
            "- Speak only when it's the agent's turn (user has finished speaking + brief silence)\n"
            "- Wait while the user is speaking\n"
            "- Yield immediately if the user interrupts the agent mid-sentence\n"
            "- Repair (say \"Sorry, go ahead.\") if the agent and user spoke simultaneously\n"
            "- Clarify (ask user to repeat) if the agent spoke and the user didn't respond\n\n"
            "Current conversation state:\n"
            "- Agent state: speak\n"
            "- Last user utterance: (none)\n"
            "- Last agent utterance: \"Hello! I would like to tell you about our new service tier.\"\n"
            "- Silence duration: 0ms\n"
            "- VAD energy: -20.0 dB\n"
            "- Attribution: both\n"
            "- Has queued utterance: False\n"
            "- Recent history:\n"
            "  Agent: Hello! I would like to tell you about our new service tier.\n\n"
            "Available actions: speak, wait, clarify, repair, yield\n\n"
            "What should the agent do next? Respond with:\n"
            "ACTION: <one of: speak, wait, clarify, repair, yield>\n"
            "EXPLANATION: <1-2 sentences explaining why>\n"
            "TEXT: <the exact text to speak, or \"none\">"
        ),
    ),
    Scenario(
        name="clarify_after_silence",
        expected_action="clarify",
        prompt=(
            "You are a speaking policy for a conversational AI agent. Your job is to decide when "
            "the agent should speak, wait, clarify, repair, or yield in a turn-taking conversation.\n\n"
            "Rules:\n"
            "- Speak only when it's the agent's turn (user has finished speaking + brief silence)\n"
            "- Wait while the user is speaking\n"
            "- Yield immediately if the user interrupts the agent mid-sentence\n"
            "- Repair (say \"Sorry, go ahead.\") if the agent and user spoke simultaneously\n"
            "- Clarify (ask user to repeat) if the agent spoke and the user didn't respond\n\n"
            "Current conversation state:\n"
            "- Agent state: listen\n"
            "- Last user utterance: \"Yes, it keeps making a weird noise.\"\n"
            "- Last agent utterance: \"Could you describe the sound?\"\n"
            "- Silence duration: 2500ms\n"
            "- VAD energy: -52.0 dB\n"
            "- Attribution: silence\n"
            "- Has queued utterance: False\n"
            "- Recent history:\n"
            "  Agent: Could you describe the sound?\n"
            "  User: Yes, it keeps making a weird noise.\n\n"
            "Available actions: speak, wait, clarify, repair, yield\n\n"
            "What should the agent do next? Respond with:\n"
            "ACTION: <one of: speak, wait, clarify, repair, yield>\n"
            "EXPLANATION: <1-2 sentences explaining why>\n"
            "TEXT: <the exact text to speak, or \"none\">"
        ),
    ),
    Scenario(
        name="wait_while_user_speaks",
        expected_action="wait",
        prompt=(
            "You are a speaking policy for a conversational AI agent. Your job is to decide when "
            "the agent should speak, wait, clarify, repair, or yield in a turn-taking conversation.\n\n"
            "Rules:\n"
            "- Speak only when it's the agent's turn (user has finished speaking + brief silence)\n"
            "- Wait while the user is speaking\n"
            "- Yield immediately if the user interrupts the agent mid-sentence\n"
            "- Repair (say \"Sorry, go ahead.\") if the agent and user spoke simultaneously\n"
            "- Clarify (ask user to repeat) if the agent spoke and the user didn't respond\n\n"
            "Current conversation state:\n"
            "- Agent state: listen\n"
            "- Last user utterance: \"I'm still talking about the account issue.\"\n"
            "- Last agent utterance: \"Okay, tell me more.\"\n"
            "- Silence duration: 120ms\n"
            "- VAD energy: -18.0 dB\n"
            "- Attribution: user\n"
            "- Has queued utterance: True\n"
            "- Recent history:\n"
            "  Agent: Okay, tell me more.\n"
            "  User: I'm still talking about the account issue.\n\n"
            "Available actions: speak, wait, clarify, repair, yield\n\n"
            "What should the agent do next? Respond with:\n"
            "ACTION: <one of: speak, wait, clarify, repair, yield>\n"
            "EXPLANATION: <1-2 sentences explaining why>\n"
            "TEXT: <the exact text to speak, or \"none\">"
        ),
    ),
]


MODELS: List[ModelSpec] = [
    ModelSpec(name="hermes/openai-codex:gpt-5.4-mini", kind="hermes", model="gpt-5.4-mini", timeout_s=60.0),
    ModelSpec(name="openrouter/owl-alpha", kind="openrouter", model="openrouter/owl-alpha", timeout_s=60.0),
    ModelSpec(name="openrouter/openai-gpt-oss-safeguard-20b", kind="openrouter", model="openai/gpt-oss-safeguard-20b", timeout_s=60.0),
    ModelSpec(name="openrouter/liquid-lfm-2.5-1.2b-instruct-free", kind="openrouter", model="liquid/lfm-2.5-1.2b-instruct:free", timeout_s=60.0),
]


ACTION_RE = re.compile(r"^ACTION:\s*([a-zA-Z_\- ]+)$", re.IGNORECASE | re.MULTILINE)
TEXT_RE = re.compile(r"^TEXT:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def read_openrouter_key() -> str:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        raise RuntimeError(f"OpenRouter key file not found: {env_path}")
    for line in env_path.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in ~/.hermes/.env")


def call_hermes(prompt: str, model: str, timeout_s: float) -> Dict[str, Any]:
    cmd = ["hermes", "chat", "-q", prompt, "--quiet"]
    if model:
        cmd.extend(["-m", model])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    latency = time.perf_counter() - t0
    stdout = proc.stdout or ""
    lines = [line for line in stdout.splitlines() if not line.startswith("session_id:")]
    content = "\n".join(lines).strip()
    return {
        "latency_s": latency,
        "raw_text": stdout,
        "content": content,
        "error": None if proc.returncode == 0 else proc.stderr.strip() or f"exit={proc.returncode}",
        "model_used": f"hermes:{model or 'default'}",
    }


def call_openrouter(prompt: str, model: str, timeout_s: float) -> Dict[str, Any]:
    key = read_openrouter_key()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 160,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        latency = time.perf_counter() - t0
        data = json.loads(raw)
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content")
            # some models may return reasoning only
            if content is None:
                content = msg.get("reasoning") or msg.get("refusal") or ""
            return {
                "latency_s": latency,
                "raw_text": raw,
                "content": content or "",
                "error": None,
                "model_used": data.get("model", model),
            }
        return {
            "latency_s": latency,
            "raw_text": raw,
            "content": "",
            "error": data.get("error", raw),
            "model_used": data.get("model", model),
        }
    except Exception as e:
        latency = time.perf_counter() - t0
        return {
            "latency_s": latency,
            "raw_text": "",
            "content": "",
            "error": str(e),
            "model_used": model,
        }


def parse_action(text: str) -> Optional[str]:
    if not text:
        return None
    m = ACTION_RE.search(text)
    if not m:
        return None
    return m.group(1).strip().lower().replace(" ", "_")


def extract_text_field(text: str) -> str:
    if not text:
        return ""
    m = TEXT_RE.search(text)
    if not m:
        return ""
    val = m.group(1).strip()
    if val.lower() in {"none", '"none"', "n/a", "na", ""}:
        return ""
    return val


def benchmark_model(model: ModelSpec) -> Dict[str, Any]:
    results = []
    for scenario in SCENARIOS:
        if model.kind == "hermes":
            out = call_hermes(scenario.prompt, model.model, model.timeout_s)
        else:
            out = call_openrouter(scenario.prompt, model.model, model.timeout_s)
        parsed_action = parse_action(out["content"])
        parsed_text = extract_text_field(out["content"])
        results.append(
            {
                "scenario": scenario.name,
                "expected_action": scenario.expected_action,
                "parsed_action": parsed_action,
                "parsed_text": parsed_text,
                "latency_s": round(out["latency_s"], 3),
                "ok": parsed_action == scenario.expected_action,
                "content": out["content"],
                "error": out["error"],
                "model_used": out["model_used"],
            }
        )
    successful = sum(1 for r in results if r["ok"])
    parseable = sum(1 for r in results if r["parsed_action"] is not None)
    latencies = [r["latency_s"] for r in results]
    return {
        "model": model.name,
        "kind": model.kind,
        "model_id": model.model,
        "timeout_s": model.timeout_s,
        "results": results,
        "summary": {
            "accuracy": successful / len(results),
            "parse_rate": parseable / len(results),
            "avg_latency_s": round(statistics.mean(latencies), 3),
            "p95_latency_s": round(max(latencies), 3),
            "scenarios": len(results),
            "correct": successful,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark speaking-policy prompts across multiple models")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--models", type=str, default="", help="Comma-separated model names to run (subset of defaults)")
    args = parser.parse_args()

    models = MODELS
    if args.models.strip():
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        models = [m for m in MODELS if m.name in wanted or m.model in wanted]
        if not models:
            raise SystemExit(f"No matching models for --models={args.models}")

    started = dt.datetime.utcnow().isoformat() + "Z"
    payload = {
        "started_at": started,
        "scenarios": [
            {"name": s.name, "expected_action": s.expected_action} for s in SCENARIOS
        ],
        "models": [],
    }

    print(f"Benchmarking {len(models)} models on {len(SCENARIOS)} prompts...")
    print()
    for model in models:
        print(f"[{model.name}] running...")
        model_result = benchmark_model(model)
        payload["models"].append(model_result)
        s = model_result["summary"]
        print(
            f"  accuracy={s['accuracy']:.2%} parse_rate={s['parse_rate']:.2%} "
            f"avg_latency={s['avg_latency_s']:.3f}s p95={s['p95_latency_s']:.3f}s"
        )
        for row in model_result["results"]:
            print(
                f"    - {row['scenario']:<22} expected={row['expected_action']:<7} "
                f"got={str(row['parsed_action']):<8} latency={row['latency_s']:.3f}s"
            )
        print()

    # Rank by accuracy then latency
    ranked = sorted(
        payload["models"],
        key=lambda m: (-m["summary"]["accuracy"], m["summary"]["avg_latency_s"]),
    )
    payload["ranked"] = [
        {
            "model": m["model"],
            "accuracy": m["summary"]["accuracy"],
            "avg_latency_s": m["summary"]["avg_latency_s"],
            "parse_rate": m["summary"]["parse_rate"],
        }
        for m in ranked
    ]
    payload["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"

    out_path = args.output
    if not out_path:
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"results/m1_model_benchmark_{stamp}.json"
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved results to: {out_file.resolve()}")
    print("\nRanking:")
    for idx, item in enumerate(payload["ranked"], 1):
        print(
            f"  {idx}. {item['model']} | accuracy={item['accuracy']:.2%} | "
            f"avg_latency={item['avg_latency_s']:.3f}s | parse_rate={item['parse_rate']:.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
