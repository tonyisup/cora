#!/usr/bin/env python3
"""M2 — Guarded runtime wrapper for the distilled speaking policy.

The distilled model handles the ambiguous policy decisions, but deterministic
safety rails remain in place for collision/yield and repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from m2_train import DistilledSpeakingPolicy


@dataclass
class GuardedDistilledPolicy:
    """Runtime guard around the distilled model.

    Safety rules:
    - If we're already in repair state, repair takes precedence.
    - If both user and agent are speaking (collision/interrupt), yield.
    - Otherwise delegate to the learned model.
    """

    model: DistilledSpeakingPolicy

    def predict(self, features: Dict[str, Any]) -> Tuple[str, int]:
        state = str(features.get("state", ""))
        user_speaking = bool(features.get("user_speaking", False))
        agent_speaking = bool(features.get("agent_speaking", False))
        attribution = str(features.get("attribution", ""))

        # Hard safety rails
        if state == "repair":
            return "repair", 3

        if agent_speaking and user_speaking:
            # Collision / interruption: yield immediately
            return "yield", 4

        # Delegate to learned model
        return self.model.predict(features)
