"""FastAPI application exposing turn and feedback endpoints."""

from __future__ import annotations

import asyncio
import html
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError, field_validator

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
    history: list[str] = Field(default_factory=list, description='\u3053\u308c\u307e\u3067\u306e\u4f1a\u8a71\u5c65\u6b74')
    user_utterance: str = Field(..., description='\u73fe\u5728\u306e\u30e6\u30fc\u30b6\u767a\u8a71')
    N: int | None = Field(None, description='\u751f\u6210\u3059\u308b\u5019\u88dc\u6570')
    session_id: str | None = Field(None, description='\u30bb\u30c3\u30b7\u30e7\u30f3ID\uff08\u4efb\u610f\uff09')
    goal: str | None = Field(None, description='\u5bfe\u8a71\u306e\u76ee\u7684\u30fb\u30b4\u30fc\u30eb')
    user_profile: dict[str, Any] | None = Field(None, description='\u30e6\u30fc\u30b6\u30fc\u30d7\u30ed\u30d5\u30a1\u30a4\u30eb(JSON)')
    constraints: dict[str, Any] | None = Field(None, description='\u5236\u7d04\u6761\u4ef6(JSON)')
    styles: list[str] | None = Field(None, description='\u30b9\u30bf\u30a4\u30eb\u306e\u30db\u30ef\u30a4\u30c8\u30ea\u30b9\u30c8')

    @staticmethod
    def _parse_dict_field(value: Any, field_name: str) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return None
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field_name}\u306f\u6709\u52b9\u306aJSON\u3092\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044") from exc
        elif isinstance(value, dict):
            parsed = value
        else:
            raise TypeError(f"{field_name}\u306f\u8f9e\u66f8\u5f62\u5f0f\u3067\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044")
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name}\u306f\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u5f62\u5f0f\u306eJSON\u3067\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044")
        return parsed

    @field_validator('user_utterance', mode='before')
    @classmethod
    def _coerce_utterance(cls, value: str | None) -> str:
        if value is None:
            raise ValueError('\u30e6\u30fc\u30b6\u767a\u8a71\u304c\u7a7a\u3067\u3059')
        value = str(value).strip()
        if not value:
            raise ValueError('\u30e6\u30fc\u30b6\u767a\u8a71\u304c\u7a7a\u3067\u3059')
        return value

    @field_validator('history', mode='before')
    @classmethod
    def _coerce_history(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [line for line in value.splitlines() if line.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        raise TypeError('history\u306f\u6587\u5b57\u5217\u307e\u305f\u306f\u30ea\u30b9\u30c8\u3067\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044')

    @field_validator('history')
    @classmethod
    def _validate_history(cls, value: list[str]) -> list[str]:
        if len(value) > 50:
            raise ValueError('history\u306f50\u4ef6\u307e\u3067\u306b\u3057\u3066\u304f\u3060\u3055\u3044')
        return value

    @field_validator('goal', mode='before')
    @classmethod
    def _coerce_goal(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) > 500:
            raise ValueError('goal\u306f500\u6587\u5b57\u4ee5\u5185\u3067\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044')
        return text

    @field_validator('user_profile', mode='before')
    @classmethod
    def _coerce_user_profile(cls, value: Any) -> dict[str, Any] | None:
        return cls._parse_dict_field(value, 'user_profile')

    @field_validator('constraints', mode='before')
    @classmethod
    def _coerce_constraints(cls, value: Any) -> dict[str, Any] | None:
        return cls._parse_dict_field(value, 'constraints')

    @field_validator('styles', mode='before')
    @classmethod
    def _coerce_styles(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            items = [part.strip() for part in value.split(',') if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            items = [str(part).strip() for part in value if str(part).strip()]
        else:
            raise TypeError('styles\u306f\u6587\u5b57\u5217\u307e\u305f\u306f\u30ea\u30b9\u30c8\u3067\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044')
        if not items:
            return None
        deduped = list(dict.fromkeys(items))
        return deduped

    @field_validator('styles')
    @classmethod
    def _validate_styles(cls, value: list[str] | None) -> list[str] | None:
        if value and len(value) > 20:
            raise ValueError('styles\u306f20\u4ef6\u4ee5\u5185\u3067\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044')
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
        styles_allowed=request.styles or CONFIG.styles_whitelist,
        goal=request.goal,
        user_profile=request.user_profile,
        constraints=request.constraints,
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
        "styles_catalog": sorted(_orchestrator.styles_catalog.keys()),
        "default_styles": CONFIG.styles_whitelist or [],
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
        return HTMLResponse("<div class='text-red-400 text-sm'>\u30e6\u30fc\u30b6\u767a\u8a71\u3092\u5165\u529b\u3057\u3066\u304f\u3060\u3055\u3044\u3002</div>", status_code=400)

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
            return HTMLResponse("<div class='text-red-400 text-sm'>\u5019\u88dc\u6570\u306f\u6570\u5024\u3067\u6307\u5b9a\u3057\u3066\u304f\u3060\u3055\u3044\u3002</div>", status_code=400)

    goal_value = (form.get("goal") or "").strip()
    user_profile_raw = form.get("user_profile")
    constraints_raw = form.get("constraints")
    styles_selected = form.getlist("styles") if hasattr(form, "getlist") else form.get("styles")

    payload = {
        "history": history_data,
        "user_utterance": user_utterance,
        "N": n_override,
        "session_id": session_id,
        "goal": goal_value or None,
        "user_profile": user_profile_raw,
        "constraints": constraints_raw,
        "styles": styles_selected,
    }
    try:
        turn_request = TurnRequest.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors()
        message = errors[0].get("msg", "入力値に誤りがあります。") if errors else "入力値に誤りがあります。"
        escaped = html.escape(message)
        return HTMLResponse(f"<div class='text-red-400 text-sm'>{escaped}</div>", status_code=400)

    candidate_count = turn_request.N or CONFIG.candidate_count
    if candidate_count <= 0:
        return HTMLResponse("<div class='text-red-400 text-sm'>N\u306f\u6b63\u306e\u6574\u6570\u306b\u3057\u3066\u304f\u3060\u3055\u3044\u3002</div>", status_code=400)
    if turn_request.session_id and len(turn_request.session_id) > 128:
        return HTMLResponse("<div class='text-red-400 text-sm'>session_id\u304c\u9577\u3059\u304e\u307e\u3059\u3002</div>", status_code=400)

    messages = _messages_from_history(turn_request.history, turn_request.user_utterance)
    context = GenerationContext(
        messages=messages,
        candidate_count=candidate_count,
        styles_allowed=turn_request.styles or CONFIG.styles_whitelist,
        goal=turn_request.goal,
        user_profile=turn_request.user_profile,
        constraints=turn_request.constraints,
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
                "features": candidate.features,
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
