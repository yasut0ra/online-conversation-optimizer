"""High-level orchestration for turns and feedback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .bandit import BanditManager, LinUCB
from .features import FeatureExtractor
from .generation import CandidateGenerator
from .logging_utils import JsonlInteractionLogger, log_turn
from .prompt_loader import PromptLoader
from .safety.guard import review_candidates
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
    session_id: str
    turn_id: str
    candidates: list[Candidate]
    decision: BanditDecision
    feature_vectors: list[list[float]]
    feature_logs: list[dict[str, float]]

    @property
    def chosen_candidate(self) -> Candidate:
        return self.candidates[self.decision.chosen_index]


@dataclass
class PendingInteraction:
    context_hash: str
    session_id: str
    turn_id: str
    feature_vectors: list[list[float]]
    feature_logs: list[dict[str, float]]
    decision: BanditDecision
    candidates: list[Candidate]
    safety: dict[str, object]


class ConversationOrchestrator:
    """Coordinates candidate generation, selection, and logging."""

    def __init__(
        self,
        prompt_loader: PromptLoader,
        generator: CandidateGenerator | None = None,
        bandit_manager: BanditManager | None = None,
        feature_extractor: FeatureExtractor | None = None,
        logger: JsonlInteractionLogger | None = None,
        bandit_algo: str = "linucb",
    ) -> None:
        self._generator = generator or CandidateGenerator(prompt_loader)
        if feature_extractor is None:
            feature_extractor = FeatureExtractor(self._generator.styles_catalog)
        self._feature_extractor = feature_extractor
        self._bandit = bandit_manager or BanditManager(LinUCB())
        self._logger = logger
        self._bandit_algo = bandit_algo
        self._pending: dict[tuple[str, str], PendingInteraction] = {}

    @property
    def styles_catalog(self) -> dict[str, dict[str, object]]:
        """Expose the generator's styles catalog for UI consumption."""
        return self._generator.styles_catalog

    def run_turn(
        self,
        context: GenerationContext,
        session_id: str | None = None,
        turn_id: str | None = None,
    ) -> TurnResult:
        candidates = self._generator.generate(context)
        candidates, safety_meta = self._apply_safety(context, candidates)
        feature_vectors, feature_logs = self._feature_extractor.build_features(
            context, candidates
        )

        feature_matrix = np.asarray(feature_vectors, dtype=float)
        prior_scores = np.zeros(feature_matrix.shape[0])
        decision = self._bandit.select(prior_scores, feature_matrix)
        context_hash = _hash_context(context)
        session = session_id or context_hash
        turn = turn_id or context_hash

        propensity = decision.propensities[decision.chosen_index]
        log_record = InteractionLogRecord(
            context_hash=context_hash,
            session_id=session,
            turn_id=turn,
            candidates=candidates,
            chosen_idx=decision.chosen_index,
            propensity=propensity,
            reward=None,
            features={
                "vectors": feature_vectors,
                "mappings": feature_logs,
                "scores": decision.scores,
                "safety": safety_meta,
            },
        )
        if self._logger:
            self._logger.log(log_record)

        log_turn(
            session_id=session,
            turn_id=turn,
            payload={
                "phase": "turn",
                "context_hash": context_hash,
                "candidates": candidates,
                "chosen_idx": decision.chosen_index,
                "propensity": propensity,
                "reward": None,
                "features": log_record.features,
                "bandit_algo": self._bandit_algo,
            },
        )

        key = (session, turn)
        self._pending[key] = PendingInteraction(
            context_hash=context_hash,
            session_id=session,
            turn_id=turn,
            feature_vectors=feature_vectors,
            feature_logs=feature_logs,
            decision=decision,
            candidates=candidates,
            safety=safety_meta,
        )

        return TurnResult(
            context_hash=context_hash,
            session_id=session,
            turn_id=turn,
            candidates=candidates,
            decision=decision,
            feature_vectors=feature_vectors,
            feature_logs=feature_logs,
        )

    def apply_feedback(
        self,
        session_id: str,
        turn_id: str,
        chosen_idx: int,
        reward: float,
    ) -> None:
        key = (session_id, turn_id)
        pending = self._pending.pop(key, None)
        if not pending:
            raise KeyError("該当のターンが見つかりませんでした")

        feature_matrix = np.asarray(pending.feature_vectors, dtype=float)
        if chosen_idx < 0 or chosen_idx >= feature_matrix.shape[0]:
            raise ValueError("選択された候補が一致しません")

        self._bandit.update(feature_matrix, reward, chosen_idx)

        propensity = (
            pending.decision.propensities[chosen_idx]
            if 0 <= chosen_idx < len(pending.decision.propensities)
            else None
        )
        log_features = {
            "vectors": pending.feature_vectors,
            "mappings": pending.feature_logs,
            "scores": pending.decision.scores,
            "safety": pending.safety,
        }

        if self._logger:
            record = InteractionLogRecord(
                context_hash=pending.context_hash,
                session_id=session_id,
                turn_id=turn_id,
                candidates=pending.candidates,
                chosen_idx=chosen_idx,
                propensity=propensity,
                reward=reward,
                features=log_features,
            )
            self._logger.log(record)

        log_turn(
            session_id=session_id,
            turn_id=turn_id,
            payload={
                "phase": "feedback",
                "context_hash": pending.context_hash,
                "candidates": pending.candidates,
                "chosen_idx": chosen_idx,
                "propensity": propensity,
                "reward": reward,
                "features": log_features,
                "bandit_algo": self._bandit_algo,
            },
        )

    def _apply_safety(
        self, context: GenerationContext, candidates: list[Candidate]
    ) -> tuple[list[Candidate], dict[str, object]]:
        attempts = 0
        last_scores: list[float] = []
        last_rewrites: list[str] = []
        original_candidates = candidates
        while attempts < 2:
            approved, scores, rewrites = review_candidates(candidates)
            last_scores = scores
            last_rewrites = rewrites
            if approved:
                filtered: list[Candidate] = []
                for idx in approved:
                    cand = candidates[idx]
                    cand_features = dict(cand.features)
                    cand_features["safety_score"] = scores[idx]
                    filtered.append(Candidate(text=cand.text, style=cand.style, features=cand_features))
                meta = {
                    "approved_indices": approved,
                    "scores": scores,
                    "rewrites": rewrites,
                    "attempts": attempts + 1,
                }
                return filtered, meta
            attempts += 1
            if attempts < 2:
                candidates = self._generator.generate(context)

        sanitized: list[Candidate] = []
        sanitized_scores: list[float] = []
        source_candidates = candidates or original_candidates
        for idx, cand in enumerate(source_candidates):
            text = last_rewrites[idx] if idx < len(last_rewrites) and last_rewrites[idx] else cand.text
            cand_features = dict(cand.features)
            score = last_scores[idx] if idx < len(last_scores) else 0.0
            cand_features.update({
                "safety_score": score,
                "sanitized": True,
            })
            sanitized.append(Candidate(text=text, style=cand.style, features=cand_features))
            sanitized_scores.append(score)

        meta = {
            "approved_indices": [],
            "scores": sanitized_scores,
            "rewrites": last_rewrites,
            "attempts": attempts,
            "sanitized": True,
        }
        return sanitized, meta
