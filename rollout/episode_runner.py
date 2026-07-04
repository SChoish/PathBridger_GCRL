"""Chunked env rollout runner shared by manip and maze rollout entrypoints.

Success is decided **only** by ``info['success']`` reported by the env (and
``terminated`` / ``truncated`` for episode termination). No user-defined
distance tolerance or goal-dim selection is applied here; the env is the single
source of truth for whether an evaluation episode succeeded.

Maze state-space plotting (``rollout/subgoal.py``) is the only place where an
arbitrary tolerance threshold is allowed, since it is a visualization artifact
rather than an evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from rollout.common import align_action_to_env
from rollout.env import env_render_rgb_u8


SampleActionChunk = Callable[[np.ndarray, np.ndarray], np.ndarray]
PreChunkHook = Callable[[np.ndarray, np.ndarray], None]


@dataclass
class RolloutOutcome:
    """Result returned by :func:`run_chunked_episode`."""

    states: np.ndarray
    rgb_frames: np.ndarray | None
    n_chunks: int
    ok_env: bool
    terminated: bool
    truncated: bool
    extras: dict[str, Any] = field(default_factory=dict)


def _maybe_append_rgb(env: Any, frames: List[np.ndarray]) -> None:
    fr = env_render_rgb_u8(env)
    if fr is not None:
        frames.append(fr)


def run_chunked_episode(
    env: Any,
    s0: np.ndarray,
    s_g: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    max_chunks: int,
    sample_action_chunk: SampleActionChunk,
    pre_chunk_hook: PreChunkHook | None = None,
    post_step_hook: Optional[Callable[[Any], None]] = None,
    record_rgb: bool = True,
) -> RolloutOutcome:
    """Replan + step loop shared by manip and maze rollouts.

    Each chunk: (optional) ``pre_chunk_hook`` for visualization side effects,
    ``sample_action_chunk(obs, goal)`` to get the next action chunk, then the
    env is stepped action-by-action, an RGB frame is captured per env step
    (when ``record_rgb=True``). If ``post_step_hook`` is set, it is called with
    ``env`` after each successful ``step`` (and after optional RGB capture).
    The episode ends when the env reports
    ``info['success']`` at any step, or returns ``terminated``/``truncated``,
    or the chunk budget is exhausted.
    """
    obs = np.asarray(s0, dtype=np.float32).reshape(-1)
    goal = np.asarray(s_g, dtype=np.float32).reshape(-1)
    states: list[np.ndarray] = [obs.copy()]
    rgb_frames: list[np.ndarray] = []
    if record_rgb:
        _maybe_append_rgb(env, rgb_frames)

    cum_env = False
    terminated = False
    truncated = False
    n_chunks = 0

    for _ in range(max(1, int(max_chunks))):
        if terminated or truncated or cum_env:
            break
        if pre_chunk_hook is not None:
            pre_chunk_hook(obs, goal)
        chunk = np.asarray(sample_action_chunk(obs, goal), dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f'sample_action_chunk must return rank-2 array, got shape {chunk.shape}.')
        for action in chunk:
            if terminated or truncated:
                break
            clipped = np.clip(action, low, high)
            ob, _reward, term, trunc, info = env.step(clipped)
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            terminated = bool(term)
            truncated = bool(trunc)
            success_flag = bool(info.get('success', False)) if isinstance(info, dict) else False
            cum_env = cum_env or success_flag
            if record_rgb:
                _maybe_append_rgb(env, rgb_frames)
            if post_step_hook is not None:
                post_step_hook(env)
            if success_flag:
                break
        states.append(obs.copy())
        n_chunks += 1

    rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
    return RolloutOutcome(
        states=np.stack(states, axis=0),
        rgb_frames=rgb,
        n_chunks=int(n_chunks),
        ok_env=bool(cum_env),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def make_idm_chunk_fn(
    dynamics_agent: Any,
    idm_horizon: int,
) -> SampleActionChunk:
    """Return ``(obs, goal) -> action_chunk`` using dynamics ``infer_subgoal`` + IDM."""
    from main import _idm_action_chunk

    def _chunk(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        s = jnp.asarray(obs, dtype=jnp.float32)
        g = jnp.asarray(goal, dtype=jnp.float32)
        pred = np.asarray(jax.device_get(dynamics_agent.infer_subgoal(s, g)), dtype=np.float32).reshape(-1)
        return _idm_action_chunk(dynamics_agent, np.asarray(obs, dtype=np.float32).reshape(-1), pred, int(idm_horizon))

    return _chunk


def make_actor_chunk_fn(
    dynamics_agent: Any,
    actor_agent: Any,
    actor_horizon: int,
    env_action_dim: int,
) -> SampleActionChunk:
    """Return ``(obs, goal) -> action_chunk`` using dynamics subgoal + actor."""

    def _chunk(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        s = jnp.asarray(obs, dtype=jnp.float32)
        g = jnp.asarray(goal, dtype=jnp.float32)
        pred = np.asarray(jax.device_get(dynamics_agent.infer_subgoal(s, g)), dtype=np.float32).reshape(-1)
        chunk = np.asarray(
            jax.device_get(actor_agent.sample_actions(s, jnp.asarray(pred, dtype=jnp.float32))),
            dtype=np.float32,
        ).reshape(int(actor_horizon), -1)
        if not chunk.flags.writeable:
            chunk = chunk.copy()
        for i in range(int(chunk.shape[0])):
            chunk[i] = align_action_to_env(chunk[i], int(env_action_dim))
        return chunk

    return _chunk
