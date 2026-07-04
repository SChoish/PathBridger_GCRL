"""Shared run-directory / checkpoint / parsing helpers used by training and rollout scripts."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import flax
import numpy as np
from ml_collections import ConfigDict

from agents.dynamics import get_dynamics_config
from utils.datasets import Dataset


# --- checkpoint helpers -------------------------------------------------------


def list_checkpoint_suffixes(checkpoints_dir: Path) -> list[int]:
    """Integers ``n`` such that ``params_<n>.pkl`` exists under ``checkpoints_dir``."""
    out: list[int] = []
    for p in Path(checkpoints_dir).glob('params_*.pkl'):
        m = re.search(r'params_(\d+)\.pkl$', p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def pick_epoch(requested: int, suffixes: list[int], *, label: str = 'checkpoint') -> int:
    """``requested < 0`` → latest. Otherwise nearest available with a warning."""
    if not suffixes:
        raise FileNotFoundError(f'No {label} suffixes available.')
    if int(requested) < 0:
        return int(suffixes[-1])
    if int(requested) in suffixes:
        return int(requested)
    nearest = min(suffixes, key=lambda x: abs(x - int(requested)))
    print(f'Warning: {label} {int(requested)} not found; using {nearest}')
    return int(nearest)


def load_checkpoint_pkl(agent: Any, pkl_path: Path) -> Any:
    """Load ``params_*.pkl`` (training runs save under ``{'agent': state_dict}``) into ``agent``."""
    with open(pkl_path, 'rb') as f:
        load_dict = pickle.load(f)
    return flax.serialization.from_state_dict(agent, load_dict['agent'])


def resolve_dynamics_checkpoint_dir(run_dir: Path) -> Path:
    """Return the directory holding dynamics ``params_*.pkl`` (runs save under ``checkpoints/dynamics/``)."""
    base = Path(run_dir) / 'checkpoints'
    if not base.is_dir():
        raise FileNotFoundError(f'No checkpoints/ under {run_dir}')
    if list_checkpoint_suffixes(base):
        return base
    nested = base / 'dynamics'
    if nested.is_dir() and list_checkpoint_suffixes(nested):
        return nested
    raise FileNotFoundError(
        f'No params_*.pkl under {base} or {nested} (expected dynamics checkpoints).'
    )


def resolve_critic_checkpoint_dir(run_dir: Path) -> Path:
    d = Path(run_dir) / 'checkpoints' / 'critic'
    if not d.is_dir():
        raise FileNotFoundError(f'Missing critic checkpoints directory: {d}')
    if not list_checkpoint_suffixes(d):
        raise FileNotFoundError(f'No params_*.pkl under {d}')
    return d


def resolve_actor_checkpoint_dir(run_dir: Path, *, required: bool = False) -> Path | None:
    d = Path(run_dir) / 'checkpoints' / 'actor'
    if not d.is_dir() or not list_checkpoint_suffixes(d):
        if required:
            raise FileNotFoundError(f'Missing actor checkpoints under {d}')
        return None
    return d


# --- flags.json ---------------------------------------------------------------


def load_run_flags(run_dir: Path) -> tuple[ConfigDict, str]:
    """Return ``(merged dynamics config, env_name)`` from ``flags.json`` in ``run_dir``."""
    flags_path = Path(run_dir) / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Missing flags.json under {run_dir}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags = json.load(f)
    env_name = flags.get('env_name')
    if not env_name and isinstance(flags.get('flags'), dict):
        env_name = flags['flags'].get('env_name')
    if not env_name:
        raise KeyError('flags.json must contain env_name (top-level or flags.env_name)')
    cfg = get_dynamics_config()
    agent_updates = flags.get('agent')
    if agent_updates:
        for k, v in agent_updates.items():
            cfg[k] = v
    elif isinstance(flags.get('dynamics'), dict):
        for k, v in flags['dynamics'].items():
            cfg[k] = v
    return cfg, env_name


# --- parsing ------------------------------------------------------------------


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse ``"1,2,3"`` → ``(1, 2, 3)``. Returns ``()`` for empty strings."""
    items = [item.strip() for item in str(text).split(',') if item.strip()]
    return tuple(int(item) for item in items)


# --- goal distance helpers ----------------------------------------------------


def goal_distance(s: np.ndarray, g: np.ndarray, dims: tuple[int, ...] | None) -> float:
    if dims:
        idx = np.asarray(dims, dtype=np.int32)
        return float(np.linalg.norm(s[idx] - g[idx]))
    return float(np.linalg.norm(s - g))


def goal_within_tol(
    s: np.ndarray, g: np.ndarray, dims: tuple[int, ...] | None, tol: float
) -> bool:
    """``tol > 0``일 때만 사용; ``tol`` 이하(``<=``)면 도달."""
    if tol is None or float(tol) <= 0.0:
        return False
    return goal_distance(s, g, dims) <= float(tol)


# --- offline trajectory helpers (used by rollout scripts) ---------------------


def episode_slices(terminals: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` inclusive indices for each episode."""
    terminals = np.asarray(terminals).reshape(-1)
    ends = np.nonzero(terminals > 0)[0]
    if len(ends) == 0:
        raise ValueError('No terminal flags found; cannot split episodes.')
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def get_trajectory(dataset: Dataset, traj_idx: int) -> np.ndarray:
    obs = np.asarray(dataset['observations'])
    terms = np.asarray(dataset['terminals'])
    slices = episode_slices(terms)
    if traj_idx < 0 or traj_idx >= len(slices):
        raise IndexError(f'traj_idx={traj_idx} out of range [0, {len(slices) - 1}]')
    s, e = slices[traj_idx]
    return obs[s : e + 1].copy()


__all__ = [
    'list_checkpoint_suffixes',
    'pick_epoch',
    'load_checkpoint_pkl',
    'resolve_dynamics_checkpoint_dir',
    'resolve_critic_checkpoint_dir',
    'resolve_actor_checkpoint_dir',
    'load_run_flags',
    'parse_int_list',
    'goal_distance',
    'goal_within_tol',
    'episode_slices',
    'get_trajectory',
]
