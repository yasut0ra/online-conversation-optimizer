"""Shared dataclasses and type aliases for the conversation optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Represents a single message in the dialog history."""

    role: str
    content: str


@dataclass
class Candidate:
    """A generated candidate reply with attached feature metadata."""

    text: str
    style: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationContext:
    """Input context for generating reply candidates."""

    messages: list[Message]
    user_profile: dict[str, Any] | None = None
    goal: str | None = None
    constraints: dict[str, Any] | None = None
    styles_allowed: list[str] | None = None
    candidate_count: int = 3


@dataclass
class BanditDecision:
    """Decision returned by the bandit policy."""

    chosen_index: int
    propensities: list[float]
    scores: list[float]


@dataclass
class InteractionLogRecord:
    """Structure recorded for every interaction step."""

    context_hash: str
    session_id: str
    turn_id: str
    candidates: list[Candidate]
    chosen_idx: int
    propensity: float
    reward: float | None
    features: dict[str, Any]
