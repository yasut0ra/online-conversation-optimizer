"""Shared dataclasses and type aliases for the conversation optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Represents a single message in the dialog history."""

    role: str
    content: str


@dataclass
class Candidate:
    """A generated candidate reply with optional metadata."""

    text: str
    style: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationContext:
    """Input context for generating reply candidates."""

    messages: List[Message]
    user_profile: Optional[Dict[str, Any]] = None
    goal: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    styles_allowed: Optional[List[str]] = None
    candidate_count: int = 3


@dataclass
class BanditDecision:
    """Decision returned by the bandit policy."""

    chosen_index: int
    propensities: List[float]
    scores: List[float]


@dataclass
class InteractionLogRecord:
    """Structure recorded for every interaction step."""

    context_hash: str
    candidates: List[Candidate]
    chosen_idx: int
    propensity: float
    reward: Optional[float]
    features: Dict[str, Any]

