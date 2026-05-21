#!/usr/bin/env python3
"""Scenario-level benchmark for speaking-policy models.

This benchmark uses a fixed set of turn-taking scenarios that mirror the
simulator's important decision points, then records:
- raw response
- parsed ACTION
- expected ACTION
- correctness
- latency
- parse failures / errors

It is intentionally small and repeatable so we can compare models as the
candidate set changes.

Usage:
  .venv/bin/python scripts/m1_scenario_benchmark.py
  .venv/bin/python scripts/m1_scenario_benchmark.py --models "hermes/current-config,openrouter/owl-alpha"
  .venv/bin/python scripts/m1_scenario_benchmark.py --output results/m1_scenario_benchmark_pilot.json
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
ACTION_RE = re.compile(r"^ACTION:\s*([a-zA-Z_\- ]+)$", re.IGNORECASE | re.MULTILINE)
TEXT_RE = re.compile(r"^TEXT:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Scenario:
    name: str
    expected_action: str
    prompt: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # hermes | openrouter
    model: str
    timeout_s: float = 60.0

SYSTEM_PROMPT = """/nothink\n"
            <|im_start|>system\n"
            "nothinking\n"
            "Do not overthink. Be direct and strict with the formatting. Do not add any additional text or explanation.
            "You are a speaking policy.\n"
            "Rules: speak only when the user is silent and a queued utterance exists;\n"
            "wait while the user is speaking; yield on collision; clarify after no reply;\n"
            "repair when already in repair state.\n\n"
            "REPLY ONLY IN THE FOLLOWING FORMAT EXACTLY WITH NO ADDITIONAL TEXT:\n"
            "ACTION: <speak|wait|clarify|repair|yield>\n"
            "EXPLANATION: <1 sentence>\n"
            "TEXT: <text or none>\n"
            "<|im_end|>\n"""

SCENARIOS: List[Scenario] = [
    Scenario(
        name="speak_on_silence",
        expected_action="speak",
        prompt=(
            SYSTEM_PROMPT + "\n" +
            "<|im_start|>user\n"
            "State: listen\n"
            "Attribution: silence\n"
            "Silence duration: 400ms\n"
            "Has queued utterance: yes\n"
            "Last agent utterance: Hello! How can I help you today?\n"
            "Last user utterance: (none)\n"
            "<|im_end|>\n"
        ),
    ),
    Scenario(
        name="wait_while_user_speaks",
        expected_action="wait",
        prompt=(
            SYSTEM_PROMPT + "\n" +
            "<|im_start|>user\n"    
            "State: listen\n"
            "Attribution: user\n"
            "Silence duration: 120ms\n"
            "Has queued utterance: yes\n"
            "Last agent utterance: Okay, tell me more.\n"
            "Last user utterance: I'm still talking about the account issue.\n\n"
            "<|im_end|>\n"
        ),
    ),
    Scenario(
        name="yield_on_interrupt",
        expected_action="yield",
        prompt=(
            SYSTEM_PROMPT + "\n" +
            "<|im_start|>user\n"
            "State: speak\n"
            "Attribution: user\n"
            "Silence duration: 0ms\n"
            "Has queued utterance: no\n"
            "Last agent utterance: Hello! I would like to tell you about our new service tier.\n"
            "Last user utterance: Sorry to interrupt, but I need help right now.\n\n"
            "<|im_end|>\n"
        ),
    ),
    Scenario(
        name="yield_on_collision",
        expected_action="yield",
        prompt=(
            SYSTEM_PROMPT + "\n" +
            "<|im_start|>user\n"
            "State: speak\n"
            "Attribution: both\n"
            "Silence duration: 0ms\n"
            "Has queued utterance: no\n"
            "Last agent utterance: Hello! I would like to tell you about our new service tier.\n"
            "Last user utterance: (overlap)\n"
            "<|im_end|>\n"
        ),
    ),
    Scenario(
        name="clarify_after_silence",
        expected_action="clarify",
        prompt=(
            SYSTEM_PROMPT + "\n" +
            "<|im_start|>user\n"
            "State: listen\n"
            "Attribution: silence\n"
            "Silence duration: 2500ms\n"
            "Has queued utterance: no\n"
            "Last agent utterance: Could you describe the sound?\n"
            "Last user utterance: Yes, it keeps making a weird noise.\n"
            "<|im_end|>\n"
        ),
    ),
    Scenario(
        name="repair_in_repair_state",
        expected_action="repair",
        prompt=(
            SYSTEM_PROMPT + "\n" +
            "<|im_start|>user\n"
            "State: repair\n"
            "Attribution: silence\n"
            "Silence duration: 0ms\n"
            "Has queued utterance: no\n"
            "Repair utterance queued: Sorry, go ahead.\n"
            "Last agent utterance: Sorry, go ahead.\n"
            "Last user utterance: (none)\n"
            "<|im_end|>\n"
        ),
    ),
]

MODELS: List[ModelSpec] = [
    ModelSpec(name="local-omlx/gemma-4-26b-a4b-it-4bit", kind="local-omlx", model="gemma-4-26b-a4b-it-4bit"),
    # ModelSpec(name="local-omlx/gemma-4-e4b-it-4bit", kind="local-omlx", model="gemma-4-e4b-it-4bit"),
    # ModelSpec(name="local-omlx/Qwen3.5-9B-MLX-4bit", kind="local-omlx", model="Qwen3.5-9B-MLX-4bit"),
    # ModelSpec(name="hermes/openai-codex:gpt-5.4-mini", kind="hermes", model="gpt-5.4-mini"),
    # ModelSpec(name="hermes/openai-codex:gpt-5.4-codex", kind="hermes", model="gpt-5.3-codex"),
    # ModelSpec(name="hermes/openai-codex:gpt-5.5", kind="hermes", model="gpt-5.5"),
    # ModelSpec(name="openrouter/owl-alpha", kind="openrouter", model="openrouter/owl-alpha"),
    # ModelSpec(name="openrouter/liquid-lfm-2.5-1.2b-instruct:free", kind="openrouter", model="liquid/lfm-2.5-1.2b-instruct:free"),
    # ModelSpec(name="openrouter/openai-gpt-oss-safeguard-20b", kind="openrouter", model="openai/gpt-oss-safeguard-20b"),
    # ModelSpec(name="nvidia/nemotron-nano-9b-v2", kind="openrouter", model="nvidia/nemotron-nano-9b-v2"),
    # ModelSpec(name="meta-llama/llama-3.2-3b-instruct", kind="openrouter", model="meta-llama/llama-3.2-3b-instruct"),
    # ModelSpec(name="ibm-granite/granite-4.1-8b", kind="openrouter", model="ibm-granite/granite-4.1-8b"),
    # ModelSpec(name="deepseek/deepseek-r1-distill-llama-70b", kind="openrouter", model="deepseek/deepseek-r1-distill-llama-70b"),
    # ModelSpec(name="google/gemma-4-26b-a4b-instruct", kind="openrouter", model="google/gemma-4-26b-a4b-instruct"),
    # ModelSpec(name="anthropic/claude-3.5-haiku", kind="openrouter", model="anthropic/claude-3.5-haiku"),
    # ModelSpec(name="google/gemini-3.1-flash-lite", kind="openrouter", model="google/gemini-3.1-flash-lite"),
    # ModelSpec(name="moonshotai/kimi-k2.5-flash", kind="openrouter", model="moonshotai/kimi-k2.5-flash"),
    # ModelSpec(name="minimax/minimax-m2.5-flash", kind="openrouter", model="minimax/minimax-m2.5-flash")
]

FEW_SHOT_PREFIX = (
    "Few-shot examples of the required decision style:\n\n"
    "Example 1\n"
    "State: listen\n"
    "Attribution: silence\n"
    "Silence duration: 500ms\n"
    "Has queued utterance: yes\n"
    "Expected: ACTION: speak\n"
    "Explanation: The user is silent and a queued response exists, so it is the agent's turn.\n"
    "Text: Hello! How can I help you today?\n\n"
    "Example 2\n"
    "State: speak\n"
    "Attribution: both\n"
    "Silence duration: 0ms\n"
    "Has queued utterance: no\n"
    "Expected: ACTION: yield\n"
    "Explanation: The user and agent overlapped, so the agent should yield immediately.\n"
    "Text: none\n\n"
    "Example 3\n"
    "State: listen\n"
    "Attribution: silence\n"
    "Silence duration: 2500ms\n"
    "Has queued utterance: no\n"
    "Last agent utterance: Could you repeat that?\n"
    "Last user utterance: (no reply)\n"
    "Expected: ACTION: clarify\n"
    "Explanation: The agent asked a question and the user did not answer, so the right recovery is clarification.\n"
    "Text: Sorry — could you repeat that?\n\n"
    "Example 4\n"
    "State: repair\n"
    "Attribution: silence\n"
    "Silence duration: 0ms\n"
    "Has queued utterance: no\n"
    "Repair utterance queued: Sorry, go ahead.\n"
    "Expected: ACTION: repair\n"
    "Explanation: The repair state already contains a repair utterance, so emit it now.\n"
    "Text: Sorry, go ahead.\n\n"
)


def read_openrouter_key() -> str:
    env_path = Path.home() / ".hermes" / ".env"
    if not env_path.exists():
        raise RuntimeError(f"missing OpenRouter env file: {env_path}")
    for line in env_path.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in ~/.hermes/.env")


def parse_action(text: str) -> Optional[str]:
    if not text:
        return None
    m = ACTION_RE.search(text)
    return m.group(1).strip().lower().replace(" ", "_") if m else None


def parse_text(text: str) -> str:
    if not text:
        return ""
    m = TEXT_RE.search(text)
    if not m:
        return ""
    val = m.group(1).strip()
    if val.lower() in {"none", '"none"', "n/a", "na", ""}:
        return ""
    return val


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
        "content": content,
        "raw": stdout,
        "error": None if proc.returncode == 0 else (proc.stderr.strip() or f"exit={proc.returncode}"),
        "model_used": f"hermes:{model or 'current-config'}",
    }

LOCAL_OMLX_URL = "http://127.0.0.1:8666/v1/chat/completions"


def call_local_omlx(prompt: str, model: str, timeout_s: float) -> Dict[str, Any]:
    return call_openrouter(prompt, model, timeout_s, key="a2b4", url=LOCAL_OMLX_URL)

def call_openrouter(prompt: str, model: str, timeout_s: float, key: str = None, url: str = OPENROUTER_URL) -> Dict[str, Any]:
    key = key or read_openrouter_key()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 160,
    }
    req = urllib.request.Request(
        url,
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
            content = msg.get("content") or msg.get("reasoning") or msg.get("refusal") or ""
            return {
                "latency_s": latency,
                "content": content,
                "raw": raw,
                "error": None,
                "model_used": data.get("model", model),
            }
        return {
            "latency_s": latency,
            "content": "",
            "raw": raw,
            "error": data.get("error", raw),
            "model_used": data.get("model", model),
        }
    except Exception as e:
        latency = time.perf_counter() - t0
        return {
            "latency_s": latency,
            "content": "",
            "raw": "",
            "error": str(e),
            "model_used": model,
        }


def benchmark_model(model: ModelSpec, few_shot: bool = False) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        prompt = (FEW_SHOT_PREFIX + "\n" + scenario.prompt) if few_shot else scenario.prompt

        if model.kind == "hermes":
            out = call_hermes(prompt, model.model, model.timeout_s)
        elif model.kind == "local-omlx":
            out = call_local_omlx(prompt, model.model, model.timeout_s)
        else:
            out = call_openrouter(prompt, model.model, model.timeout_s)

        parsed_action = parse_action(out["content"])
        parsed_text = parse_text(out["content"])
        rows.append(
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

    latencies = [r["latency_s"] for r in rows]
    correct = sum(1 for r in rows if r["ok"])
    parseable = sum(1 for r in rows if r["parsed_action"] is not None)
    return {
        "model": model.name,
        "kind": model.kind,
        "model_id": model.model,
        "few_shot": few_shot,
        "results": rows,
        "summary": {
            "accuracy": correct / len(rows),
            "parse_rate": parseable / len(rows),
            "avg_latency_s": round(statistics.mean(latencies), 3),
            "p95_latency_s": round(max(latencies), 3),
            "correct": correct,
            "scenarios": len(rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scenario-level benchmark for speaking-policy models")
    parser.add_argument("--models", type=str, default="", help="Comma-separated subset of model names or ids")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--few-shot", action="store_true", help="Prepend few-shot decision examples to every prompt")
    args = parser.parse_args()

    models = MODELS
    if args.models.strip():
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        models = [m for m in MODELS if m.name in wanted or m.model in wanted]
        if not models:
            raise SystemExit(f"No matching models for --models={args.models}")

    started = dt.datetime.now(dt.UTC).isoformat()
    payload: Dict[str, Any] = {
        "started_at": started,
        "few_shot": args.few_shot,
        "scenarios": [{"name": s.name, "expected_action": s.expected_action} for s in SCENARIOS],
        "models": [],
    }

    print(f"Scenario benchmark: {len(models)} models × {len(SCENARIOS)} scenarios")
    print()
    for model in models:
        print(f"[{model.name}] running...")
        model_result = benchmark_model(model, few_shot=args.few_shot)
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

    ranked = sorted(
        payload["models"],
        key=lambda m: (-m["summary"]["accuracy"], m["summary"]["avg_latency_s"]),
    )
    payload["ranked"] = [
        {
            "model": m["model"],
            "accuracy": m["summary"]["accuracy"],
            "parse_rate": m["summary"]["parse_rate"],
            "avg_latency_s": m["summary"]["avg_latency_s"],
        }
        for m in ranked
    ]
    payload["finished_at"] = dt.datetime.now(dt.UTC).isoformat()

    out_path = args.output or f"results/m1_scenario_benchmark_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}.json"
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"Saved: {out_file.resolve()}")
    print("Ranking:")
    for idx, item in enumerate(payload["ranked"], 1):
        print(
            f"  {idx}. {item['model']} | accuracy={item['accuracy']:.2%} | "
            f"parse_rate={item['parse_rate']:.2%} | avg_latency={item['avg_latency_s']:.3f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
