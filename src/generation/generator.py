"""Prompt-wired candidate generator using OpenAI with fallbacks."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..prompt_loader import PromptLoader
from ..types import Candidate, GenerationContext, Message

DEFAULT_MODEL_NAME = "gpt-4o-mini"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_ORDER = ["00_system_core", "10_generator", "11_styles_catalog"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a conversation response generator. Always reply with a JSON array of "
    "candidate objects. Each object must include 'text' and 'style' fields, and a "
    "'features' object with metadata when available."
)

logger = logging.getLogger(__name__)


def _load_styles_catalog(loader: PromptLoader) -> dict:
    raw = loader.load("11_styles_catalog")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(raw[start : end + 1])


def _compose_system_prompt(loader: PromptLoader) -> str:
    parts: list[str] = []
    for prompt_id in PROMPT_ORDER:
        try:
            parts.append(loader.load(prompt_id))
        except FileNotFoundError:
            continue
    return "\n\n".join(parts)




def _parse_candidates_payload(raw: str) -> list[dict]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty payload")

    decoder = json.JSONDecoder()
    snippets: list[str] = []
    for candidate in (raw, raw.lstrip()):
        if candidate and candidate not in snippets:
            snippets.append(candidate)
    for token in ("[", "{"):
        idx = raw.find(token)
        if idx != -1:
            snippet = raw[idx:]
            if snippet and snippet not in snippets:
                snippets.append(snippet)

    for snippet in snippets:
        try:
            parsed, _ = decoder.raw_decode(snippet)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("candidates", "outputs", "choices", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    try:
                        decoded = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(decoded, list):
                        return decoded
    raise ValueError("could not parse candidates from payload")

def _detect_language(text: str) -> str:
    if re.search("[\u3040-\u30ff\u4e00-\u9fff]", text):
        return "ja"
    if re.search("[A-Za-z]", text):
        return "en"
    return "ja"


def _build_features(text: str, style: str, style_meta: dict, language: str) -> dict:
    words = text.split()
    return {
        "length_chars": len(text),
        "length_words": len(words),
        "is_question": text.strip().endswith("?"),
        "language": language,
        "style_initiative": float(style_meta.get("initiative", 0.5)),
        "style_risk": float(style_meta.get("risk", 0.2)),
        "style": style,
    }


class CandidateGenerator:
    """Generate candidate replies by calling OpenAI or using deterministic fallback."""

    def __init__(
        self,
        prompt_loader: PromptLoader | None = None,
        model: str = DEFAULT_MODEL_NAME,
        temperature: float = 0.8,
        max_output_tokens: int = 600,
    ) -> None:
        self._loader = prompt_loader or PromptLoader(PROMPTS_DIR)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        composed_prompt = _compose_system_prompt(self._loader).strip()
        self._system_prompt = composed_prompt or DEFAULT_SYSTEM_PROMPT
        self._styles_catalog = _load_styles_catalog(self._loader)

    @property
    def styles_catalog(self) -> dict:
        return self._styles_catalog

    def generate(self, context: GenerationContext) -> list[Candidate]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                return self._generate_via_openai(context, api_key)
            except Exception as exc:
                logger.exception("LLM generation failed; falling back", exc_info=exc)
        return self._generate_fallback(context)

    def _generate_via_openai(
        self, context: GenerationContext, api_key: str
    ) -> list[Candidate]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for LLM generation") from exc

        client = OpenAI(api_key=api_key)
        history_str = self._format_history(context.messages)
        payload = {
            "history": history_str,
            "user_profile": context.user_profile or {},
            "goal": context.goal or "",
            "constraints": context.constraints or {},
            "styles_allowed": context.styles_allowed
            or list(self._styles_catalog.keys()),
            "N": context.candidate_count,
        }

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        completion = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            messages=messages,
        )

        text_fragments: list[str] = []
        for choice in completion.choices:
            message = getattr(choice, "message", None)
            if not message:
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str):
                text_fragments.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        text_fragments.append(part)
                    elif isinstance(part, dict):
                        value = part.get("text") or part.get("content") or part.get("value")
                        if isinstance(value, str):
                            text_fragments.append(value)

        raw = "\n".join(text_fragments).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = _parse_candidates_payload(raw)
        except ValueError as exc:
            logger.error("LLM response not valid JSON: %s", raw)
            raise RuntimeError("LLM response was not valid JSON") from exc
        candidates: list[Candidate] = []
        language = self._infer_language(context.messages)
        for item in parsed:
            style = item.get("style", "unknown")
            features = item.get("features") or item.get("meta") or {}
            text = item.get("text", "")
            style_meta = self._styles_catalog.get(style, {})
            merged_features = {
                **_build_features(text, style, style_meta, language),
                **features,
            }
            candidates.append(Candidate(text=text, style=style, features=merged_features))

        if not candidates:
            return self._generate_fallback(context)
        return candidates[: context.candidate_count]

    def _generate_fallback(self, context: GenerationContext) -> list[Candidate]:
        styles = context.styles_allowed or list(self._styles_catalog.keys())
        if not styles:
            styles = ["empathetic", "logical", "coach"]
        language = self._infer_language(context.messages)
        last_user = self._last_user_message(context.messages)
        results: list[Candidate] = []
        for style in styles[: context.candidate_count]:
            text = self._fallback_text(style, last_user, language)
            style_meta = self._styles_catalog.get(style, {})
            features = _build_features(text, style, style_meta, language)
            results.append(Candidate(text=text, style=style, features=features))
        return results

    def _fallback_text(self, style: str, last_user: str, language: str) -> str:
        prompts = {
            "empathetic": {
                "en": (
                    f"I hear how that feels: {last_user[:120]}".rstrip(".")
                    + ". Would one small step today help you steady things?"
                ),
                "ja": (
                    f"気持ち、伝わってきました：{last_user[:60]}".rstrip("。")
                    + "。まず一歩、どんな行動が安心につながりそう？"
                ),
            },
            "logical": {
                "en": (
                    "Let's map it quickly. Core issue: "
                    f"{last_user[:80]}".rstrip(".")
                    + ". Next, pick one actionable constraint to test."
                ),
                "ja": (
                    "論点を整理しよう。焦点は" + f"{last_user[:40]}".rstrip("。")
                    + "。次に試せる制約をひとつ選ぼう。"
                ),
            },
            "coach": {
                "en": "Picture the progress you want this week. What's one move you can commit to?",
                "ja": "今週進めたい形は？ 直近でやれる一手を一緒に決めよう。",
            },
            "playful": {
                "en": "If this were a game, your move would set the tone—want to try a tiny bold experiment?",
                "ja": "ゲーム感覚でいこう！次の一手で雰囲気が決まるよ。小さな実験、試してみない？",
            },
            "concise_expert": {
                "en": "Focus on the single lever with the biggest upside and schedule a quick review after acting.",
                "ja": "一番リターンの高いレバーに絞って動こう。実行後すぐにセルフレビューを。",
            },
        }
        language = language if language in {"ja", "en"} else "ja"
        style_prompts = prompts.get(style, prompts["empathetic"])
        return style_prompts.get(language, style_prompts["ja"])

    def _format_history(self, messages: Iterable[Message], last_k: int = 4) -> str:
        recent = list(messages)[-last_k:]
        return "\n".join(f"[{msg.role.upper()}] {msg.content}" for msg in recent)

    def _last_user_message(self, messages: Sequence[Message]) -> str:
        for message in reversed(messages):
            if message.role.lower() == "user":
                return message.content
        return ""

    def _infer_language(self, messages: Sequence[Message]) -> str:
        last_user = self._last_user_message(messages)
        if last_user:
            return _detect_language(last_user)
        return "ja"


_GLOBAL_GENERATOR = CandidateGenerator()


def _normalise_messages(history: Sequence | None) -> list[Message]:
    messages: list[Message] = []
    if not history:
        return messages
    for item in history:
        if isinstance(item, Message):
            messages.append(item)
        elif isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            messages.append(Message(role=role, content=content))
        else:
            messages.append(Message(role="user", content=str(item)))
    return messages


def generate_candidates(
    history: Sequence | None,
    user_utterance: str,
    candidate_count: int,
    styles_whitelist: Sequence[str] | None = None,
) -> list[Candidate]:
    """Generate candidates given history and current user utterance."""

    messages = _normalise_messages(history)
    if user_utterance:
        if not messages or messages[-1].content != user_utterance:
            messages.append(Message(role="user", content=user_utterance))

    context = GenerationContext(
        messages=messages,
        candidate_count=candidate_count,
        styles_allowed=list(styles_whitelist) if styles_whitelist else None,
    )
    return _GLOBAL_GENERATOR.generate(context)
