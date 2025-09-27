"""Feature extraction for context and candidate combinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from ..types import Candidate, GenerationContext, Message


def _last_user_message(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role.lower() == "user":
            return message.content
    return ""


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


@dataclass
class FeatureExtractor:
    """Compute feature vectors used by the contextual bandit."""

    styles_catalog: Dict[str, Dict[str, object]]

    def build_features(
        self, context: GenerationContext, candidates: Iterable[Candidate]
    ) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Return numeric features and a friendly mapping for logging."""

        base_features = self._context_features(context)
        vectors: List[List[float]] = []
        dense_mappings: List[Dict[str, float]] = []

        for candidate in candidates:
            style_meta = self.styles_catalog.get(candidate.style, {})
            vec = list(base_features)
            vec.extend(self._candidate_features(candidate, style_meta))
            vectors.append(vec)
            dense_mappings.append(
                {
                    "bias": 1.0,
                    "ctx_len": base_features[1],
                    "last_user_chars": base_features[2],
                    "candidate_words": vec[3],
                    "candidate_question": vec[4],
                    "style_initiative": vec[5],
                    "style_risk": vec[6],
                }
            )

        return vectors, dense_mappings

    def _context_features(self, context: GenerationContext) -> List[float]:
        messages = context.messages
        last_user = _last_user_message(messages)
        last_len = len(last_user)
        total_msgs = len(messages)

        return [
            1.0,  # bias
            _clip(total_msgs / 10.0, 0.0, 1.5),
            _clip(last_len / 400.0, 0.0, 1.5),
        ]

    def _candidate_features(
        self, candidate: Candidate, style_meta: Dict[str, object]
    ) -> List[float]:
        words = len(candidate.text.split())
        question = 1.0 if "?" in candidate.text else 0.0
        initiative = float(style_meta.get("initiative", 0.5))
        risk = float(style_meta.get("risk", 0.2))

        return [
            _clip(words / 80.0, 0.0, 2.0),
            question,
            _clip(initiative, 0.0, 1.0),
            _clip(risk, 0.0, 1.0),
        ]

