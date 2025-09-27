"""FastAPI application exposing turn and feedback endpoints."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from .bandit import BanditManager, LinTS, LinUCB
from .config import AppConfig, load_config
from .features import FeatureExtractor
from .generation import CandidateGenerator
from .logging_utils import JsonlInteractionLogger
from .metrics import compute_metrics
from .orchestrator import ConversationOrchestrator
from .prompt_loader import PromptLoader
from .types import GenerationContext, Message

CONFIG: AppConfig = load_config()
BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
templates = Jinja2Templates(directory=str(UI_DIR / "templates"))


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
app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")
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


@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request) -> HTMLResponse:
    session_id = str(uuid.uuid4())
    context = {
        "request": request,
        "session_id": session_id,
        "history_json": "[]",
        "candidate_count": CONFIG.candidate_count,
    }
    return templates.TemplateResponse(request, "index.html", context)


@app.get("/metrics")
async def metrics() -> dict[str, object]:
    return compute_metrics()


@app.post("/api/turn", response_class=HTMLResponse)
async def api_turn(request: Request) -> HTMLResponse:
    form = await request.form()
    user_utterance = (form.get("user_utterance") or "").strip()
    if not user_utterance:
        return HTMLResponse("<div class='text-red-400 text-sm'>ユーザ発話を入力してください。</div>", status_code=400)

    history_raw = form.get("history_json") or "[]"
    try:
        history_data = json.loads(history_raw)
        if not isinstance(history_data, list):
            history_data = []
    except json.JSONDecodeError:
        history_data = []

    session_id = (form.get("session_id") or "").strip() or None
    candidate_count_value = form.get("candidate_count")
    n_override = None
    if candidate_count_value:
        try:
            n_override = int(candidate_count_value)
        except ValueError:
            return HTMLResponse("<div class='text-red-400 text-sm'>候補数は数値で指定してください。</div>", status_code=400)

    payload = {
        "history": history_data,
        "user_utterance": user_utterance,
        "N": n_override,
        "session_id": session_id,
    }
    turn_request = TurnRequest.model_validate(payload)

    candidate_count = turn_request.N or CONFIG.candidate_count
    if candidate_count <= 0:
        return HTMLResponse("<div class='text-red-400 text-sm'>Nは正の整数にしてください。</div>", status_code=400)
    if turn_request.session_id and len(turn_request.session_id) > 128:
        return HTMLResponse("<div class='text-red-400 text-sm'>session_idが長すぎます。</div>", status_code=400)

    messages = _messages_from_history(turn_request.history, turn_request.user_utterance)
    context = GenerationContext(
        messages=messages,
        candidate_count=candidate_count,
        styles_allowed=CONFIG.styles_whitelist,
    )
    session = turn_request.session_id.strip() if turn_request.session_id else str(uuid.uuid4())
    turn_id = str(uuid.uuid4())

    async with _lock:
        result = _orchestrator.run_turn(context, session_id=session, turn_id=turn_id)

    candidates_data = []
    scores = list(result.decision.scores)
    propensities = list(result.decision.propensities)
    for idx, candidate in enumerate(result.candidates):
        candidates_data.append(
            {
                "index": idx,
                "text": candidate.text,
                "style": candidate.style,
                "score": float(scores[idx]),
                "propensity": float(propensities[idx]),
                "safety_score": candidate.features.get("safety_score"),
                "session_id": result.session_id,
                "turn_id": result.turn_id,
            }
        )

    context = {
        "request": request,
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "scores": scores,
        "propensities": propensities,
        "candidates": candidates_data,
    }
    return templates.TemplateResponse(request, "partials/candidates.html", context)


@app.post("/api/feedback")
async def api_feedback(request: Request) -> dict[str, object]:
    payload = await request.json()
    data = {
        "session_id": payload.get("session_id"),
        "turn_id": payload.get("turn_id"),
        "chosen_idx": payload.get("chosen_idx"),
        "reward": payload.get("reward", 1.0),
    }
    feedback_request = FeedbackRequest.model_validate(data)
    latency_ms = payload.get("latency_ms")
    continued = payload.get("continued")

    if feedback_request.reward < -1.0 or feedback_request.reward > 1.0:
        raise HTTPException(status_code=400, detail="rewardは-1.0から1.0の範囲で指定してください")

    async with _lock:
        try:
            _orchestrator.apply_feedback(
                feedback_request.session_id,
                feedback_request.turn_id,
                feedback_request.chosen_idx,
                feedback_request.reward,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"status": "ok", "latency_ms": latency_ms, "continued": continued}


__all__ = ["app"]
