"""Chunked env eval rollouts: success is decided **only** by ``info['success']`` from the env.

No user-defined distance tolerance is consulted here; the env is the single source of truth for
whether an evaluation episode succeeded.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from utils.ogbench_eval_helpers import append_ogbench_render, update_episode_env_success


def _env_max_episode_steps(env: Any) -> int:
    """Return the Gym/Gymnasium episode cap used as the eval rollout budget."""
    spec = getattr(env, 'spec', None)
    max_steps = getattr(spec, 'max_episode_steps', None) if spec is not None else None
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is None:
        raise ValueError('Evaluation env must expose max_episode_steps.')
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f'env max_episode_steps must be >= 1, got {max_steps}.')
    return max_steps


def execute_action_chunk_eval(
    env: Any,
    obs: np.ndarray,
    action_chunk: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    render_buf: list[np.ndarray] | None = None,
    goal_frame: np.ndarray | None = None,
    should_render: bool = False,
    video_frame_skip: int = 4,
    step_counter: list[int] | None = None,
    max_episode_steps: int | None = None,
) -> tuple[np.ndarray, bool, bool, bool]:
    """Advance env for one action chunk; stop stepping only on ``terminated`` or ``truncated``.

    Returns ``(next_obs, saw_env_success, terminated, truncated)``. ``saw_env_success`` is
    ``info['success']`` aggregated across the chunk (env's own judgement, no extra tolerance).
    """
    saw_env_success = False
    terminated = False
    truncated = False
    for action in np.asarray(action_chunk, dtype=np.float32):
        if terminated or truncated:
            break
        if step_counter is not None and max_episode_steps is not None and int(step_counter[0]) >= int(max_episode_steps):
            truncated = True
            break
        clipped = np.clip(action, low, high)
        step_ix = int(step_counter[0]) if step_counter is not None else 0
        _ob, _reward, term, trunc, info = env.step(clipped)
        obs = np.asarray(_ob, dtype=np.float32).reshape(-1)
        terminated = bool(term)
        truncated = bool(trunc)
        saw_env_success = update_episode_env_success(saw_env_success, info)
        if step_counter is not None:
            done = bool(terminated or truncated)
            if render_buf is not None:
                append_ogbench_render(
                    render_buf,
                    env,
                    goal_frame,
                    should_render=bool(should_render),
                    step=step_ix,
                    done=done,
                    video_frame_skip=int(video_frame_skip),
                )
            step_counter[0] = step_ix + 1
    return obs, saw_env_success, terminated, truncated


def rollout_chunked_eval_episode(
    env: Any,
    obs0: np.ndarray,
    goal0: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    sample_action_chunk: Callable[[np.ndarray, np.ndarray], np.ndarray],
    render_buf: list[np.ndarray] | None = None,
    goal_frame: np.ndarray | None = None,
    should_render: bool = False,
    video_frame_skip: int = 4,
) -> bool:
    """Replanning rollout until the environment reaches its max episode length.

    Returns ``True`` iff the env reported ``info['success']`` at any step.
    """
    obs = np.asarray(obs0, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal0, dtype=np.float32).reshape(-1)
    max_episode_steps = _env_max_episode_steps(env)
    step_counter = [0]
    cum_env = False
    terminated = False
    truncated = False
    while not (terminated or truncated) and int(step_counter[0]) < max_episode_steps:
        chunk = sample_action_chunk(obs, goal)
        obs, saw_e, term, trunc = execute_action_chunk_eval(
            env,
            obs,
            chunk,
            low=low,
            high=high,
            render_buf=render_buf,
            goal_frame=goal_frame,
            should_render=bool(should_render and render_buf is not None),
            video_frame_skip=video_frame_skip,
            step_counter=step_counter,
            max_episode_steps=max_episode_steps,
        )
        cum_env = cum_env or saw_e
        terminated = terminated or term
        truncated = truncated or trunc
    return bool(cum_env)
