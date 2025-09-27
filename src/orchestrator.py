"""High-level orchestration for turns and feedback."""

from __future__ import annotations

import hashlib
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from .bandit import BanditManager, LinUCB
from .feature import FeatureExtractor
from .generation import CandidateGenerator
from .logging_utils import JsonlInteractionLogger
from .prompt_loader import PromptLoader
from .types import BanditDecision, Candidate, GenerationContext, InteractionLogRecord


def _hash_context(context: GenerationContext) -> str:
    payload = {
        "messages": [f"{m.role}:{m.content}" for m in context.messages],
        "goal": context.goal,
        "constraints": context.constraints,
        "user_profile": context.user_profile,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class TurnResult:
    context_hash: str
    candidates: List[Candidate]
    decision: BanditDecision
    feature_vectors: List[List[float]]
    feature_logs: List[Dict[str, float]]

    @property
    def chosen_candidate(self) -> Candidate:
        return self.candidates[self.decision.chosen_index]


@dataclass
class PendingInteraction:
    context_hash: str
    feature_vectors: List[List[float]]
    feature_logs: List[Dict[str, float]]
    decision: BanditDecision
    candidates: List[Candidate]


class ConversationOrchestrator:
    """Coordinates candidate generation, selection, and logging."""

    def __init__(
        self,
        prompt_loader: PromptLoader,
        generator: Optional[CandidateGenerator] = None,
        bandit_manager: Optional[BanditManager] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        logger: Optional[JsonlInteractionLogger] = None,
    ) -> None:
        self._generator = generator or CandidateGenerator(prompt_loader)
        if feature_extractor is None:
            feature_extractor = FeatureExtractor(self._generator.styles_catalog)
        self._feature_extractor = feature_extractor
        self._bandit = bandit_manager or BanditManager(LinUCB())
        self._logger = logger
        self._pending: Dict[str, PendingInteraction] = {}

    def run_turn(self, context: GenerationContext) -> TurnResult:
        candidates = self._generator.generate(context)
        feature_vectors, feature_logs = self._feature_extractor.build_features(
            context, candidates
        )

        feature_matrix = np.asarray(feature_vectors, dtype=float)
        prior_scores = np.zeros(feature_matrix.shape[0])
        decision = self._bandit.select(prior_scores, feature_matrix)
        context_hash = _hash_context(context)

        propensity = decision.propensities[decision.chosen_index]
        log_record = InteractionLogRecord(
            context_hash=context_hash,
            candidates=candidates,
            chosen_idx=decision.chosen_index,
            propensity=propensity,
            reward=None,
            features={
                "vectors": feature_vectors,
                "mappings": feature_logs,
                "scores": decision.scores,
            },
        )
        if self._logger:
            self._logger.log(log_record)

        self._pending[context_hash] = PendingInteraction(
            context_hash=context_hash,
            feature_vectors=feature_vectors,
            feature_logs=feature_logs,
            decision=decision,
            candidates=candidates,
        )

        return TurnResult(
            context_hash=context_hash,
            candidates=candidates,
            decision=decision,
            feature_vectors=feature_vectors,
            feature_logs=feature_logs,
        )

    def apply_feedback(self, context_hash: str, reward: float) -> None:
        pending = self._pending.pop(context_hash, None)
        if not pending:
            raise KeyError(f"No pending interaction found for {context_hash}")

        feature_matrix = np.asarray(pending.feature_vectors, dtype=float)
        self._bandit.update(feature_matrix, reward, pending.decision.chosen_index)

        if self._logger:
            propensity = pending.decision.propensities[pending.decision.chosen_index]
            record = InteractionLogRecord(
                context_hash=context_hash,
                candidates=pending.candidates,
                chosen_idx=pending.decision.chosen_index,
                propensity=propensity,
                reward=reward,
                features={
                    "vectors": pending.feature_vectors,
                    "mappings": pending.feature_logs,
                    "scores": pending.decision.scores,
                },
            )
            self._logger.log(record)
