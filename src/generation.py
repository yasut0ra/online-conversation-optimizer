"""Candidate generation pipeline that wraps the LLM call with fallbacks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from .prompt_loader import PromptLoader
from .types import Candidate, GenerationContext, Message

DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _format_history(messages: Iterable[Message], last_k: int = 4) -> str:
    """Render the last-k messages for the generator prompt."""

    recent = list(messages)[-last_k:]
    lines: List[str] = []
    for msg in recent:
        prefix = msg.role.upper()
        lines.append(f"[{prefix}] {msg.content}")
    return "\n".join(lines)


class CandidateGenerator:
    """Generate reply candidates via OpenAI with a deterministic fallback."""

    def __init__(
        self,
        prompt_loader: PromptLoader,
        model: str = DEFAULT_MODEL_NAME,
        temperature: float = 0.8,
        max_output_tokens: int = 500,
    ) -> None:
        self._loader = prompt_loader
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._styles_catalog = self._load_styles_catalog()

    @property
    def styles_catalog(self) -> Dict[str, Dict[str, object]]:
        """Expose the parsed styles catalog."""

        return self._styles_catalog

    def generate(self, context: GenerationContext) -> List[Candidate]:
        """Generate candidate replies using OpenAI if available, otherwise fallback."""

        if os.getenv("OPENAI_API_KEY"):
            try:
                return self._generate_via_openai(context)
            except Exception:  # pragma: no cover - defensive for runtime failures
                pass
        return self._generate_fallback(context)

    def _compose_system_prompt(self) -> str:
        parts = []
        for prompt_id in (
            "00_system_core",
            "20_safety_guard",
            "10_generator",
            "11_styles_catalog",
        ):
            try:
                parts.append(self._loader.load(prompt_id))
            except FileNotFoundError:
                continue
        return "\n\n".join(parts)

    def _generate_via_openai(self, context: GenerationContext) -> List[Candidate]:
        """Call the OpenAI Responses API to obtain structured candidates."""

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is required for LLM generation") from exc

        client = OpenAI()
        system_prompt = self._compose_system_prompt()
        history_rendered = _format_history(context.messages)
        user_prompt = json.dumps(
            {
                "history": history_rendered,
                "user_profile": context.user_profile or {},
                "goal": context.goal or "",
                "constraints": context.constraints or {},
                "styles_allowed": context.styles_allowed or list(self._styles_catalog.keys()),
                "N": context.candidate_count,
            },
            ensure_ascii=False,
        )

        response = client.responses.create(
            model=self._model,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text_fragments: List[str] = []
        for output in response.output:
            if output.type == "message":
                for segment in output.message.content:
                    if segment.type == "text":
                        text_fragments.append(segment.text)

        payload = "\n".join(text_fragments)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM response was not valid JSON") from exc

        candidates: List[Candidate] = []
        for item in parsed:
            candidates.append(
                Candidate(
                    text=item.get("text", ""),
                    style=item.get("style", "unknown"),
                    meta=item.get("meta", {}),
                )
            )

        # Ensure we have at least one candidate even if LLM misbehaves.
        if not candidates:
            return self._generate_fallback(context)
        return candidates[: context.candidate_count]

    def _generate_fallback(self, context: GenerationContext) -> List[Candidate]:
        """Deterministic heuristic generator for offline development."""

        allowed_styles = context.styles_allowed or list(self._styles_catalog.keys())
        if not allowed_styles:
            allowed_styles = list(self._styles_catalog.keys())

        last_message = ""
        for message in reversed(context.messages):
            if message.role.lower() == "user":
                last_message = message.content
                break

        candidates: List[Candidate] = []
        for style in allowed_styles[: context.candidate_count]:
            style_meta = self._styles_catalog.get(style, {})
            rationale = style_meta.get("en", "Style rationale unavailable")
            tone = style_meta.get("en", "")
            text = self._compose_style_response(style, tone, last_message)
            candidates.append(
                Candidate(
                    text=text,
                    style=style,
                    meta={
                        "rationale": rationale,
                        "len": len(text.split()),
                        "question": "?" in text,
                    },
                )
            )

        return candidates

    def _compose_style_response(self, style: str, tone: str, last_message: str) -> str:
        """Simple heuristic for fallback text synthesis."""

        if not last_message:
            last_message = "Thanks for sharing your update."

        if style == "empathetic":
            return (
                f"I hear how that feels: {last_message[:120]}".rstrip(".")
                + ". Would it help to take one small step to steady yourself today?"
            )
        if style == "logical":
            return (
                "Let's break this down. Main concern: "
                f"{last_message[:80]}. A quick next action could clarify the path forward."
            )
        if style == "coach":
            return (
                "Picture the progress you want this week."
                " What's one step you can commit to before tomorrow?"
            )
        if style == "playful":
            return (
                "If we turned this into a game, the next move would be yours—"
                "shall we try a bold but tiny experiment?"
            )
        if style == "concise_expert":
            return (
                "Given what you shared, focus on the single lever with the highest upside"
                " and timebox a review after you act."
            )
        return f"{tone} | {last_message[:100]}"

    def _load_styles_catalog(self) -> Dict[str, Dict[str, object]]:
        """Parse the styles catalog prompt into a JSON dict."""

        try:
            raw = self._loader.load("11_styles_catalog")
        except FileNotFoundError:
            return {}

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return {}

        catalog = json.loads(raw[start : end + 1])
        assert isinstance(catalog, dict)
        return catalog
