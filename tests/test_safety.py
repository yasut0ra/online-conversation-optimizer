from __future__ import annotations

from src.safety.guard import review_candidates
from src.types import Candidate


def test_review_candidates_filters_pii():
    candidates = [
        Candidate(text="You can call me at 090-1234-5678", style="logical", features={}),
        Candidate(text="安全なメッセージ", style="empathetic", features={}),
    ]

    approved, scores, rewrites = review_candidates(candidates, min_score=0.5)

    assert approved == [1]
    assert scores[0] < 0.5
    assert rewrites[0]
    assert not rewrites[1]
