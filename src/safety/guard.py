"""Basic safety guard for candidate filtering and rewriting."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from ..types import Candidate

PII_PATTERN = re.compile(r"\b(\d{3}-\d{4}|\d{3}-\d{3}-\d{4}|[0-9]{8,})\b")
BANNED_TERMS = [
    "kill yourself",
    "bomb",
    "credit card",
]
MAX_LENGTH = 640


def _contains_pii(text: str) -> bool:
    return bool(PII_PATTERN.search(text))


def _contains_banned_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in BANNED_TERMS)


def _score_candidate(text: str) -> float:
    score = 1.0
    if _contains_pii(text):
        score -= 0.6
    if _contains_banned_term(text):
        score -= 0.7
    if len(text) > MAX_LENGTH:
        score -= 0.2
    return max(0.0, min(1.0, score))


def _rewrite(text: str) -> str:
    trimmed = text[:MAX_LENGTH]
    sanitized = re.sub(PII_PATTERN, "[REDACTED]", trimmed)
    return sanitized


def review_candidates(
    candidates: Sequence[Candidate],
    min_score: float | None = None,
) -> tuple[list[int], list[float], list[str]]:
    """Return approved indices, safety scores, and rewrites."""

    if min_score is None:
        try:
            min_score = float(os.getenv("SAFETY_MIN_SCORE", "0.2"))
        except ValueError:
            min_score = 0.2

    approved: list[int] = []
    scores: list[float] = []
    rewrites: list[str] = []
    for idx, candidate in enumerate(candidates):
        text = candidate.text
        score = _score_candidate(text)
        scores.append(score)
        rewrite = ""
        if score < min_score:
            rewrite = _rewrite(text)
        else:
            approved.append(idx)
        rewrites.append(rewrite)
    return approved, scores, rewrites
