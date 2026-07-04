"""OGBench-aligned eval helpers: primary success from ``info[\"success\"]``, relaxed tol diagnostic, RGB capture."""

from __future__ import annotations

from typing import Any

import numpy as np


def info_success(info: dict[str, Any] | Any) -> bool:
    """Normalize OGBench ``info['success']`` (bool, 0/1 float, ndarray scalar) to ``bool``."""
    if not isinstance(info, dict):
        return False
    v = info.get('success', False)
    try:
        return bool(np.asarray(v).item())
    except (ValueError, TypeError):
        return bool(v)


def update_episode_env_success(episode_success: bool, info: dict[str, Any] | Any) -> bool:
    return bool(episode_success or info_success(info))


def append_ogbench_render(
    render: list[np.ndarray],
    env: Any,
    goal_frame: np.ndarray | None,
    *,
    should_render: bool,
    step: int,
    done: bool,
    video_frame_skip: int,
) -> None:
    """Append one RGB uint8 frame when ``should_render`` and skip/done rule matches OGBench-style capture."""
    if not should_render:
        return
    sk = max(1, int(video_frame_skip))
    if (step % sk != 0) and not done:
        return
    raw = np.asarray(env.render())
    frame = raw.astype(np.uint8, copy=False) if raw.dtype != np.uint8 else raw.copy()
    if goal_frame is not None:
        gf = np.asarray(goal_frame, dtype=np.uint8)
        frame = np.concatenate([gf, frame], axis=0)
    render.append(frame)
