"""Goal-conditioned scalar value V(s, g) on an (x, y) grid for rollout video overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agents.critic import CriticAgent, get_config as get_critic_config
from utils.datasets import Dataset
from utils.critic_sequence_dataset import CriticSequenceDataset
from utils.run_io import (
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    pick_epoch,
    resolve_critic_checkpoint_dir,
)


def load_critic_for_run(
    run_dir: Path,
    critic_epoch: int,
    env: Any,
    train_raw: dict,
    *,
    seed: int,
) -> Any:
    """Load ``CriticAgent`` from a run (``flags.json`` + ``checkpoints/critic/``)."""
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Missing {flags_path}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    ca = root.get('critic_agent')
    if not isinstance(ca, dict):
        raise KeyError('flags.json must contain critic_agent.')
    cfg = get_critic_config()
    for k, v in ca.items():
        cfg[k] = v
    cfg['action_dim'] = int(np.prod(env.action_space.shape))

    dataset = Dataset.create(**train_raw)
    cds = CriticSequenceDataset(dataset, cfg)
    if len(cds.valid_starts) == 0:
        raise ValueError('CriticSequenceDataset has no valid starts.')
    idx0 = int(cds.valid_starts[0])
    ex = cds.sample(1, idxs=np.asarray([idx0], dtype=np.int64), evaluation=True)
    agent = CriticAgent.create(
        int(seed),
        ex['observations'],
        ex['full_chunk_actions'],
        ex['action_chunk_actions'],
        cfg,
        ex_goals=ex.get('value_goals'),
    )
    ckpt_dir = resolve_critic_checkpoint_dir(run_dir)
    critic_epoch = pick_epoch(int(critic_epoch), list_checkpoint_suffixes(ckpt_dir), label='critic checkpoint')
    pkl_path = ckpt_dir / f'params_{critic_epoch}.pkl'
    return load_checkpoint_pkl(agent, pkl_path)


def value_mesh_for_xy(
    critic_agent: Any,
    template_obs: np.ndarray,
    goal: np.ndarray,
    d0: int,
    d1: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    grid_n: int = 56,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return ``(XX, YY, ZZ, vmin, vmax)`` for ``pcolormesh`` (``ZZ`` matches ``XX``/``YY`` shape)."""
    if grid_n < 4:
        raise ValueError('grid_n must be >= 4')
    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_n), dtype=np.float32)
    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_n), dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    flat_n = int(XX.size)
    base = np.asarray(template_obs, dtype=np.float32).reshape(1, -1)
    base = np.repeat(base, flat_n, axis=0)
    base[:, int(d0)] = np.asarray(XX.reshape(-1), dtype=np.float32)
    base[:, int(d1)] = np.asarray(YY.reshape(-1), dtype=np.float32)
    g = np.asarray(goal, dtype=np.float32).reshape(1, -1)
    g = np.repeat(g, flat_n, axis=0)

    apply_v = critic_agent.network.select('value')
    params = critic_agent.network.params

    @jax.jit
    def _v_batch(obs: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        logits = apply_v(obs, goals, params=params)
        return jax.nn.sigmoid(logits)

    outs: list[np.ndarray] = []
    for s in range(0, flat_n, int(batch_size)):
        e = min(s + int(batch_size), flat_n)
        ob = jnp.asarray(base[s:e], dtype=jnp.float32)
        gb = jnp.asarray(g[s:e], dtype=jnp.float32)
        outs.append(np.asarray(jax.device_get(_v_batch(ob, gb)), dtype=np.float32))
    zz_flat = np.concatenate(outs, axis=0)
    ZZ = zz_flat.reshape(XX.shape)
    flat = ZZ[np.isfinite(ZZ)]
    if flat.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        # Sigmoid V(s,g) lies in ~[0, 1]. Tight percentile windows made vmin/vmax almost equal
        # so the colormap used only a sliver of ``magma``. Prefer data min/max with padding; if
        # the field is still almost flat, use the full unit interval so hues span the whole map.
        lo = float(np.min(flat))
        hi = float(np.max(flat))
        span = hi - lo
        if span < 1e-5:
            vmin, vmax = 0.0, 1.0
        else:
            pad = max(0.02 * span, 1e-4)
            vmin = max(0.0, lo - pad)
            vmax = min(1.0, hi + pad)
            if vmax - vmin < 0.12:
                vmin, vmax = 0.0, 1.0
    return XX, YY, ZZ, vmin, vmax
