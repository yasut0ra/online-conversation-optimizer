"""FastAPI application exposing turn and feedback endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .bandit import BanditManager, LinTSPolicy, LinUCBPolicy
from .config import AppConfig, load_config
from .feature import FeatureExtractor
from .generation import CandidateGenerator
from .logging_utils import JsonlInteractionLogger
from .orchestrator import ConversationOrchestrator
from .prompt_loader import PromptLoader
from .types import GenerationContext, Message


CONFIG: AppConfig = load_config()


class MessageIn(BaseModel):
    role: str = Field(..., description="speaker role, e.g. user/assistant")
    content: str = Field(..., description="message text")


class TurnRequest(BaseModel):
    messages: List[MessageIn]
    user_profile: Optional[Dict[str, Any]] = None
    goal: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    styles_allowed: Optional[List[str]] = None
    candidate_count: Optional[int] = Field(
        None, description="Override default number of candidates"
    )

    def to_generation_context(self, default_count: int) -> GenerationContext:
        candidate_count = self.candidate_count or default_count
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        return GenerationContext(
            messages=[Message(role=m.role, content=m.content) for m in self.messages],
            user_profile=self.user_profile,
            goal=self.goal,
            constraints=self.constraints,
            styles_allowed=self.styles_allowed,
            candidate_count=candidate_count,
        )


class CandidateOut(BaseModel):
    text: str
    style: str
    meta: Dict[str, Any]


class TurnResponse(BaseModel):
    context_hash: str
    chosen_idx: int
    candidate: CandidateOut
    candidates: List[CandidateOut]
    propensities: List[float]
    scores: List[float]
    features: Dict[str, Any]


class FeedbackRequest(BaseModel):
    context_hash: str
    reward: float


def _build_orchestrator() -> ConversationOrchestrator:
    config = CONFIG
    repo_root = Path(__file__).resolve().parent.parent
    prompts_dir = repo_root / "prompts"
    prompt_loader = PromptLoader(prompts_dir)
    generator = CandidateGenerator(prompt_loader)
    feature_extractor = FeatureExtractor(generator.styles_catalog)

    if config.bandit_policy.lower() == "lints":
        policy = LinTSPolicy()
    else:
        policy = LinUCBPolicy()

    bandit_manager = BanditManager(policy, config.bandit_state_path)
    logger = JsonlInteractionLogger(config.log_path)
    orchestrator = ConversationOrchestrator(
        prompt_loader,
        generator=generator,
        bandit_manager=bandit_manager,
        feature_extractor=feature_extractor,
        logger=logger,
    )
    return orchestrator


app = FastAPI(title="Online Conversation Optimizer")
_orchestrator = _build_orchestrator()
_lock = asyncio.Lock()


@app.post("/turn", response_model=TurnResponse)
async def turn(request: TurnRequest) -> TurnResponse:
    try:
        context = request.to_generation_context(CONFIG.candidate_count)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    async with _lock:
        result = _orchestrator.run_turn(context)

    chosen = result.chosen_candidate
    candidates = [CandidateOut(**c.__dict__) for c in result.candidates]
    features = {
        "vectors": result.feature_vectors,
        "mappings": result.feature_logs,
    }
    return TurnResponse(
        context_hash=result.context_hash,
        chosen_idx=result.decision.chosen_index,
        candidate=CandidateOut(**chosen.__dict__),
        candidates=candidates,
        propensities=result.decision.propensities,
        scores=result.decision.scores,
        features=features,
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest) -> Dict[str, str]:
    async with _lock:
        try:
            _orchestrator.apply_feedback(request.context_hash, request.reward)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"status": "ok"}


__all__ = ["app"]
