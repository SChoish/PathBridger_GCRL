"""Shared helpers for rollout command-line entrypoints."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np

RolloutEnvFamily = Literal['manip_play', 'maze', 'generic']


def slug_from_env(env_name: str) -> str:
    """Return a filesystem-friendly slug for an environment name."""
    return re.sub(r'[^a-z0-9]+', '_', str(env_name).lower()).strip('_') or 'env'


def manip_play_family(env_name: str) -> str:
    """Return ``cube`` or ``puzzle`` for supported ManipSpace play envs."""
    en = (env_name or '').lower()
    if 'play' not in en:
        raise ValueError(f'Expected a *-play-v0 style env_name, got {env_name!r}')
    if 'cube' in en:
        return 'cube'
    if 'puzzle' in en:
        return 'puzzle'
    raise ValueError(
        f'Unsupported manip play env {env_name!r} (only cube-*-play-v0 and puzzle-*-play-v0 for now).'
    )


def classify_rollout_env(env_name: str) -> RolloutEnvFamily:
    """Classify a training run env for the unified rollout CLI."""
    en = (env_name or '').lower()
    if 'play' in en and ('cube' in en or 'puzzle' in en):
        return 'manip_play'
    if 'maze' in en or 'navigate' in en:
        return 'maze'
    return 'generic'


def align_action_to_env(action: np.ndarray, env_dim: int) -> np.ndarray:
    """Pad or truncate an action vector to match the environment action dimension."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[-1] == int(env_dim):
        return a
    if a.shape[-1] < int(env_dim):
        out = np.zeros((int(env_dim),), dtype=np.float32)
        out[: int(a.shape[-1])] = a
        return out
    return a[: int(env_dim)].copy()
