from __future__ import annotations

from pathlib import Path

from src.generation.generator import CandidateGenerator
from src.prompt_loader import PromptLoader
from src.types import GenerationContext, Message


def test_candidate_generator_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    loader = PromptLoader(Path("prompts"))
    generator = CandidateGenerator(loader)

    context = GenerationContext(
        messages=[Message(role="user", content="今日は何をすべき？")],
        candidate_count=2,
    )

    candidates = generator.generate(context)

    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.text
        assert candidate.features.get("language") in {"ja", "en"}
