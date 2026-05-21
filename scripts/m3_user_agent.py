#!/usr/bin/env python3
"""M3 local-first LLM user agent.

This module turns a scenario packet into a user reply using a local OMLX
OpenAI-compatible endpoint. It is intentionally small and dependency-light so
it can be reused by dialogue generators and future simulator hooks.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

LOCAL_OMLX_URL = "http://127.0.0.1:8666/v1/chat/completions"
DEFAULT_LOCAL_MODEL = "gemma-4-e4b-it-4bit"
DEFAULT_TIMEOUT_S = 45.0
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class UserAgentPacket:
    scenario_id: int
    scenario_name: str
    turn_index: int
    history: List[Dict[str, str]]
    last_assistant_utterance: str
    last_user_utterance: str
    goal: str
    style: str = "natural"
    allow_silence: bool = True
    allow_interrupt: bool = True
    allow_overlap: bool = True
    seed: int = 0


@dataclass(frozen=True)
class UserAgentResult:
    mode: str
    text: str
    stance: str = ""
    raw: str = ""
    parsed_ok: bool = False
    latency_s: float = 0.0
    model_used: str = ""
    error: str = ""


class LocalChatClient:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.model = model
        self.timeout_s = timeout_s

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 180,
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            LOCAL_OMLX_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer a2b4",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            latency_s = time.perf_counter() - t0
            data = json.loads(raw)
            if data.get("choices"):
                msg = data["choices"][0].get("message", {})
                content = msg.get("content") or msg.get("reasoning") or msg.get("refusal") or ""
                return {
                    "latency_s": latency_s,
                    "content": content,
                    "raw": raw,
                    "error": "",
                    "model_used": data.get("model", self.model),
                }
            return {
                "latency_s": latency_s,
                "content": "",
                "raw": raw,
                "error": data.get("error", raw),
                "model_used": data.get("model", self.model),
            }
        except Exception as exc:
            latency_s = time.perf_counter() - t0
            return {
                "latency_s": latency_s,
                "content": "",
                "raw": "",
                "error": str(exc),
                "model_used": self.model,
            }


class M3UserAgent:
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.client = LocalChatClient(model=model, timeout_s=timeout_s)

    @staticmethod
    def system_prompt() -> str:
        return (
            "You are a human user in a conversation. "
            "Respond naturally and only as the user. "
            "Return JSON only, with keys mode, text, and optionally stance. "
            "Allowed mode values: reply, silence, interrupt, overlap."
        )

    @staticmethod
    def packet_to_prompt(packet: UserAgentPacket) -> str:
        history = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in packet.history[-8:])
        return (
            f"Scenario: {packet.scenario_name} (id={packet.scenario_id})\n"
            f"Turn: {packet.turn_index}\n"
            f"Goal: {packet.goal}\n"
            f"Style: {packet.style}\n"
            f"Allow silence: {packet.allow_silence}\n"
            f"Allow interrupt: {packet.allow_interrupt}\n"
            f"Allow overlap: {packet.allow_overlap}\n"
            f"Seed: {packet.seed}\n\n"
            f"Conversation history:\n{history or '(none)'}\n\n"
            f"Last assistant utterance: {packet.last_assistant_utterance or '(none)'}\n"
            f"Last user utterance: {packet.last_user_utterance or '(none)'}\n\n"
            "Return JSON only. Example:\n"
            '{"mode":"reply","text":"Sure, the app keeps logging me out.","stance":"informative"}\n'
            "If the user would not answer, return {\"mode\":\"silence\",\"text\":\"\"}."
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        # Prefer a fenced JSON block or a direct object.
        candidates = []
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        candidates.extend(fenced)
        m = JSON_RE.search(text)
        if m:
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def generate(self, packet: UserAgentPacket) -> UserAgentResult:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.packet_to_prompt(packet)},
        ]
        out = self.client.chat(messages)
        parsed = self._extract_json(out["content"])
        if parsed is None:
            return UserAgentResult(
                mode="silence" if packet.allow_silence else "reply",
                text="",
                raw=out["content"],
                parsed_ok=False,
                latency_s=out["latency_s"],
                model_used=out["model_used"],
                error=out["error"] or "json_parse_failed",
            )

        mode = str(parsed.get("mode", "reply")).strip().lower()
        text = str(parsed.get("text", ""))
        stance = str(parsed.get("stance", ""))
        if mode not in {"reply", "silence", "interrupt", "overlap"}:
            mode = "reply"
        if mode == "silence":
            text = ""
        return UserAgentResult(
            mode=mode,
            text=text,
            stance=stance,
            raw=out["content"],
            parsed_ok=True,
            latency_s=out["latency_s"],
            model_used=out["model_used"],
            error=out["error"],
        )


def packet_to_dict(packet: UserAgentPacket) -> Dict[str, Any]:
    return asdict(packet)


def result_to_dict(result: UserAgentResult) -> Dict[str, Any]:
    return asdict(result)
