"""Utilities for loading prompt markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable


class PromptLoader:
    """Loads prompt assets from the prompts directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._cache: Dict[str, str] = {}

    def list_prompt_ids(self) -> Iterable[str]:
        """Yield prompt identifiers sorted lexicographically."""

        for path in sorted(self._root.glob("*.md")):
            yield path.stem

    def load(self, prompt_id: str) -> str:
        """Return prompt text, loading from disk on first use."""

        if prompt_id in self._cache:
            return self._cache[prompt_id]

        path = self._root / f"{prompt_id}.md"
        if not path.exists():  # pragma: no cover - guard for runtime issues
            raise FileNotFoundError(f"Prompt {prompt_id} not found at {path}")

        content = path.read_text(encoding="utf-8")
        self._cache[prompt_id] = content
        return content

    def load_all(self) -> Dict[str, str]:
        """Return all prompts keyed by their identifier."""

        for prompt_id in self.list_prompt_ids():
            self.load(prompt_id)
        return dict(self._cache)

