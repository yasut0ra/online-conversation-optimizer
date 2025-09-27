from __future__ import annotations

import numpy as np

from src.bandit.lints import LinTS
from src.bandit.linucb import LinUCB
from src.bandit.utils import softmax


def _simulate(policy, rng, iterations: int = 200, noise: float = 0.0) -> np.ndarray:
    true_theta = np.array([0.6, -0.2, 0.4])
    rewards = []
    for _ in range(iterations):
        features = rng.normal(size=(4, true_theta.size))
        prior_scores = np.zeros(features.shape[0])
        idx = policy.select(prior_scores, features)
        chosen_features = features[idx]
        reward = float(chosen_features @ true_theta + rng.normal(scale=noise))
        rewards.append(reward)
        policy.update(features, reward, idx)
        probs = policy.propensity()
        assert 0.0 <= probs <= 1.0
        full_probs = softmax(policy.last_scores, beta=policy.temperature)
        assert np.isclose(full_probs.sum(), 1.0)
        assert np.isclose(probs, full_probs[idx])
    return np.asarray(rewards)


def test_linucb_learns_linear_task():
    rng = np.random.default_rng(0)
    policy = LinUCB(alpha=0.7, lam=1.0, beta=1.0)
    _simulate(policy, rng)
    assert hasattr(policy, "_A") and policy._A is not None
    assert hasattr(policy, "_b") and policy._b is not None
    assert not np.allclose(policy._A, policy._lambda * np.eye(policy._A.shape[0]))
    assert np.linalg.norm(policy._b) > 0.0


def test_lints_learns_linear_task():
    rng = np.random.default_rng(1)
    policy = LinTS(sigma2=0.5, lam=1.0, beta=1.0, random_state=42)
    _simulate(policy, rng)
    assert hasattr(policy, "_A") and policy._A is not None
    assert hasattr(policy, "_b") and policy._b is not None
    assert not np.allclose(policy._A, policy._lambda * np.eye(policy._A.shape[0]))
    assert np.linalg.norm(policy._b) > 0.0
