"""FastAPI application exposing turn and feedback endpoints."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .bandit import BanditManager, LinTS, LinUCB
from .config import AppConfig, load_config
from .features import FeatureExtractor
from .generation import CandidateGenerator
from .logging_utils import JsonlInteractionLogger
from .orchestrator import ConversationOrchestrator
from .prompt_loader import PromptLoader
from .types import GenerationContext, Message

CONFIG: AppConfig = load_config()


class TurnRequest(BaseModel):
    history: list[str] = Field(default_factory=list, description="これまでの会話履歴")
    user_utterance: str = Field(..., description="現在のユーザ発話")
    N: int | None = Field(None, description="生成する候補数")
    session_id: str | None = Field(None, description="セッションID（任意）")

    @field_validator("user_utterance", mode="before")
    @classmethod
    def _coerce_utterance(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("ユーザ発話が空です")
        value = str(value).strip()
        if not value:
            raise ValueError("ユーザ発話が空です")
        return value

    @field_validator("history", mode="before")
    @classmethod
    def _coerce_history(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [line for line in value.splitlines() if line.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        raise TypeError("historyは文字列またはリストで指定してください")

    @field_validator("history")
    @classmethod
    def _validate_history(cls, value: list[str]) -> list[str]:
        if len(value) > 50:
            raise ValueError("historyは50件までにしてください")
        return value


class DebugInfo(BaseModel):
    scores: list[float]
    styles: list[str]


class TurnResponse(BaseModel):
    session_id: str
    turn_id: str
    reply: str
    chosen_idx: int
    propensity: float
    debug: DebugInfo | None = None


class FeedbackRequest(BaseModel):
    session_id: str
    turn_id: str
    chosen_idx: int
    reward: float


def _build_orchestrator() -> ConversationOrchestrator:
    config = CONFIG
    repo_root = Path(__file__).resolve().parent.parent
    prompts_dir = repo_root / "prompts"
    prompt_loader = PromptLoader(prompts_dir)
    generator = CandidateGenerator(prompt_loader)
    feature_extractor = FeatureExtractor(generator.styles_catalog)

    if config.bandit_algo.lower() == "lints":
        policy = LinTS()
    else:
        policy = LinUCB()

    bandit_manager = BanditManager(policy)
    logger = JsonlInteractionLogger(config.log_path)
    orchestrator = ConversationOrchestrator(
        prompt_loader,
        generator=generator,
        bandit_manager=bandit_manager,
        feature_extractor=feature_extractor,
        logger=logger,
        bandit_algo=config.bandit_algo,
    )
    return orchestrator


app = FastAPI(title="Online Conversation Optimizer")
_orchestrator = _build_orchestrator()
_lock = asyncio.Lock()


def _messages_from_history(history: list[str], user_utterance: str) -> list[Message]:
    messages: list[Message] = []
    role = "user"
    for entry in history:
        messages.append(Message(role=role, content=entry))
        role = "assistant" if role == "user" else "user"
    messages.append(Message(role="user", content=user_utterance))
    return messages


@app.post("/turn", response_model=TurnResponse)
async def turn(request: TurnRequest) -> TurnResponse:
    candidate_count = request.N or CONFIG.candidate_count
    if candidate_count <= 0:
        raise HTTPException(status_code=400, detail="Nは正の整数にしてください")

    messages = _messages_from_history(request.history, request.user_utterance)
    context = GenerationContext(
        messages=messages,
        candidate_count=candidate_count,
        styles_allowed=CONFIG.styles_whitelist,
    )
    session_id = request.session_id.strip() if request.session_id else None
    if session_id and len(session_id) > 128:
        raise HTTPException(status_code=400, detail="session_idが長すぎます")
    session = session_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())

    async with _lock:
        result = _orchestrator.run_turn(context, session_id=session, turn_id=turn_id)

    chosen = result.chosen_candidate
    debug = DebugInfo(
        scores=result.decision.scores,
        styles=[candidate.style for candidate in result.candidates],
    )

    return TurnResponse(
        session_id=result.session_id,
        turn_id=result.turn_id,
        reply=chosen.text,
        chosen_idx=result.decision.chosen_index,
        propensity=result.decision.propensities[result.decision.chosen_index],
        debug=debug,
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, str]:
    if request.reward < -1.0 or request.reward > 1.0:
        raise HTTPException(status_code=400, detail="rewardは-1.0から1.0の範囲で指定してください")

    async with _lock:
        try:
            _orchestrator.apply_feedback(
                request.session_id,
                request.turn_id,
                request.chosen_idx,
                request.reward,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok"}


__all__ = ["app"]
