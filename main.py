"""Offline training for linear-SDE dynamics + critic + SPI actor."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import yaml
from absl import app, flags

from agents.critic import (
    CriticAgent,
    extract_critic_primary_score,
    get_config as get_critic_config,
    validate_config,
)
from agents.actor import ActorAgent, get_actor_config
from agents.dynamics import DynamicsAgent, get_dynamics_config
from utils.datasets import Dataset, PathHGCDataset
from utils.critic_sequence_dataset import CriticSequenceDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.ogbench_eval_rollout import rollout_chunked_eval_episode
from utils.eval_results_io import eval_result_path, save_eval_results
from utils.run_io import parse_int_list
from utils.goal_representation import infer_phi_goal_obs_indices, normalize_phi_goal_obs_indices

FLAGS = flags.FLAGS
_DEFAULT_HORIZON = 25


def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'antmaze-medium-navigate.yaml')


def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^\w.\-]+', '_', s)
    return s[:120] if len(s) > 120 else s


def _block_until_ready(tree: Any) -> Any:
    """Synchronize a JAX pytree so wall-clock timing reflects real compute."""

    def _ready(x):
        return x.block_until_ready() if hasattr(x, 'block_until_ready') else x

    return jax.tree_util.tree_map(_ready, tree)


def _require_gpu_jax(logger: logging.Logger) -> None:
    """Fail fast if JAX did not pick the CUDA GPU backend (avoids silent CPU training)."""
    if bool(FLAGS.allow_cpu):
        logger.warning(
            'allow_cpu=True: skipping GPU-only check (jax default_backend=%s devices=%s)',
            jax.default_backend(),
            jax.devices(),
        )
        return
    backend = str(jax.default_backend()).lower()
    devs = jax.devices()
    dev_str = ', '.join(str(d) for d in devs)
    cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
    jax_plat = os.environ.get('JAX_PLATFORMS', '<unset>')
    if backend != 'gpu':
        msg = (
            f'GPU-only mode: JAX default_backend is {backend!r} (expected "gpu"). devices=[{dev_str}]. '
            f'CUDA_VISIBLE_DEVICES={cuda_vis!r} JAX_PLATFORMS={jax_plat!r}. '
            'Install a CUDA-enabled jaxlib matching your driver (e.g. cuSPARSE), confirm `nvidia-smi`, '
            'or pass --allow_cpu=True for intentional CPU runs.'
        )
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info('GPU-only check passed: default_backend=%s device_count=%d', backend, len(devs))


flags.DEFINE_string('run_config', '', 'YAML config; empty uses config/antmaze-medium-navigate.yaml.')
flags.DEFINE_string('runs_root', '', 'Run root; default <repo>/runs.')
flags.DEFINE_string(
    'dataset_dir',
    '',
    'Optional dataset directory override. If this points to a sharded OGBench directory, load and concatenate all '
    'train/val NPZ shards from there.',
)
flags.DEFINE_string('resume_run_dir', '', 'Existing run dir: resume from checkpoint or reuse path (see resume_step).')
flags.DEFINE_integer(
    'resume_step',
    -1,
    'With resume_run_dir: load params_<step>.pkl and continue if >= 0. '
    '-1 falls back to deprecated resume_epoch.',
)
flags.DEFINE_integer(
    'resume_epoch',
    0,
    'Deprecated alias for resume_step; checkpoint suffix is now a gradient step.',
)
flags.DEFINE_boolean(
    'resume_use_run_snapshot_config',
    True,
    'When resuming: if --run_config is not set on argv, load hyperparameters from '
    'resume_run_dir/flags.json (preferred; written as a temp YAML) or else config_used.yaml, '
    'so checkpoints and hparams match the original run.',
)
flags.DEFINE_string('run_group', 'Debug', 'W&B group.')
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'OGBench env / dataset name.')
flags.DEFINE_integer('train_steps', 0, 'Total gradient steps. If <= 0, use train_epochs * steps_per_epoch.')
flags.DEFINE_integer('log_every_n_steps', 10000, 'Log interval in gradient steps. If <= 0, use log_every_n_epochs * steps_per_epoch.')
flags.DEFINE_integer('save_every_n_steps', 0, 'Checkpoint interval in gradient steps. If <= 0, use save_every_n_epochs * steps_per_epoch.')
flags.DEFINE_integer('eval_every_n_steps', 0, 'Non-final eval interval in gradient steps. If <= 0, use eval_freq * steps_per_epoch.')
flags.DEFINE_integer('train_epochs', 600, 'Deprecated: converted to train_steps when train_steps <= 0.')
flags.DEFINE_integer('log_every_n_epochs', 10, 'Deprecated: converted to log_every_n_steps when log_every_n_steps <= 0.')
flags.DEFINE_integer('save_every_n_epochs', 100, 'Deprecated: converted to save_every_n_steps when save_every_n_steps <= 0.')
flags.DEFINE_boolean('use_wandb', False, 'W&B.')
flags.DEFINE_boolean('use_tqdm', False, 'tqdm over epochs.')
flags.DEFINE_integer('batch_size', 1024, 'Shared batch size for dynamics, critic, and actor.')
flags.DEFINE_integer(
    'horizon', _DEFAULT_HORIZON, 'Shared horizon for dynamics_N, subgoal_steps, and full_chunk_horizon.'
)
flags.DEFINE_boolean('measure_timing', False, 'Whether to measure and log per-phase wall-clock timings.')
flags.DEFINE_boolean(
    'async_prefetch',
    True,
    'Overlap host-side batch sampling with GPU work via a single-worker prefetch thread.',
)
flags.DEFINE_integer(
    'eval_freq',
    100,
    'Deprecated: converted to eval_every_n_steps when eval_every_n_steps <= 0; <= 0 disables non-final eval.',
)
flags.DEFINE_integer('eval_episodes_per_task', 10, 'Number of env evaluation episodes to run for each task id.')
flags.DEFINE_integer(
    'final_eval_episodes_per_task',
    25,
    'If > 0, override eval_episodes_per_task for the final training step evaluation only.',
)
flags.DEFINE_string(
    'final_eval_subgoal_eval_num_samples',
    '',
    'Comma-separated subgoal_eval_num_samples for final-step env eval only (e.g. "1,2,4,8,16"). '
    'Empty = single eval using dynamics.subgoal_eval_num_samples.',
)
flags.DEFINE_boolean(
    'subgoal_override_goal',
    False,
    'Inference/eval ablation: ignore predicted subgoals and condition IDM/actor directly on the final goal.',
)
flags.DEFINE_boolean(
    'allow_cpu',
    False,
    'If True, allow JAX to run on CPU when CUDA is unavailable. Default False: require GPU and exit '
    'with an error if jax.default_backend() is not gpu (no silent CPU fallback).',
)
flags.DEFINE_boolean(
    'train_actor_spi',
    False,
    'If True, train the SPI actor jointly with dynamics/critic (legacy behavior). '
    'Default False: skip actor proposal build/rescore/update/checkpoint during the main run; '
    'deployment workflow keeps this false for IDM training, then runs train_actor_spi.py for 50K actor finetuning.',
)

_SPI_ACTOR_KEYS = {
    'spi_tau',
    'spi_beta',
    'spi_actor_layer_norm',
    'spi_q_norm_eps',
}
def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


def _resolve_total_steps(steps_per_epoch: int) -> int:
    train_steps = int(FLAGS.train_steps)
    if train_steps <= 0:
        train_steps = int(FLAGS.train_epochs) * int(steps_per_epoch)
    if train_steps < 1:
        raise ValueError(f'train_steps must resolve to >= 1, got {train_steps}.')
    return train_steps


def _resolve_step_interval(step_value: int, epoch_value: int, steps_per_epoch: int, *, allow_disable: bool = False) -> int:
    if int(step_value) > 0:
        return int(step_value)
    if int(epoch_value) > 0:
        return int(epoch_value) * int(steps_per_epoch)
    return 0 if allow_disable else int(steps_per_epoch)


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _resolve_resume_snapshot_config_path(run_dir: str) -> str | None:
    """Prefer flags.json (full merged hparams at run start); else config_used.yaml; else None."""
    fj = os.path.join(run_dir, 'flags.json')
    if not os.path.isfile(fj):
        used = os.path.join(run_dir, 'config_used.yaml')
        return used if os.path.isfile(used) else None
    with open(fj, encoding='utf-8') as fp:
        snap = json.load(fp)
    fg = dict(snap.get('flags') or {})
    skip = {
        'resume_run_dir',
        'resume_step',
        'resume_epoch',
        'run_config',
        'runs_root',
        'help',
        'helpshort',
        'helpfull',
        'helpxml',
        '?',
    }
    root: dict[str, Any] = {}
    for key, value in fg.items():
        if key in skip:
            continue
        if hasattr(FLAGS, key):
            root[key] = value
    if snap.get('dynamics') is not None:
        root['dynamics'] = snap['dynamics']
    if snap.get('critic_agent') is not None:
        root['critic_agent'] = snap['critic_agent']
    if snap.get('actor') is not None:
        root['actor'] = snap['actor']
    fd, tmp_path = tempfile.mkstemp(prefix='resume_flags_', suffix='.yaml', text=True)
    os.close(fd)
    with open(tmp_path, 'w', encoding='utf-8') as out:
        yaml.safe_dump(root, out, sort_keys=False, default_flow_style=False)
    return tmp_path


def _argv_sets_flag(flag_name: str) -> bool:
    dashed = flag_name.replace('_', '-')
    for arg in sys.argv[1:]:
        if arg.startswith(f'--{flag_name}=') or arg.startswith(f'--{dashed}='):
            return True
        if arg in (f'--{flag_name}', f'--{dashed}'):
            return True
    return False


def _apply_yaml_to_flags(data: dict) -> tuple[dict, dict, dict]:
    dynamics_updates = data.pop('dynamics', None)
    critic_updates = data.pop('critic_agent', None)
    actor_updates = data.pop('actor', None)
    for name, updates in [('dynamics', dynamics_updates), ('critic_agent', critic_updates), ('actor', actor_updates)]:
        if updates is not None and not isinstance(updates, dict):
            raise ValueError(f'YAML key "{name}" must be a mapping.')

    # Allow forward-bridge planner knobs at the top level for ergonomic YAML
    # (route them into dynamics_updates so the dynamics-agent config picks them up).
    if 'forward_bridge' in data:
        dynamics_updates = dynamics_updates or {}
        dynamics_updates.setdefault('forward_bridge', data.pop('forward_bridge'))
    # Deprecated: eval now always runs to the environment's max episode length.
    data.pop('eval_max_chunks', None)
    # When the step-based eval interval is explicitly configured, do not let
    # the legacy default eval_freq=100 re-enable intermediate eval.
    if 'eval_every_n_steps' in data and 'eval_freq' not in data and not _argv_sets_flag('eval_freq'):
        data['eval_freq'] = 0

    # Flatten ``dynamics.forward_bridge: { mode, noise_scale, ... }`` into the
    # individual ``forward_bridge_*`` keys recognised by the dynamics config.
    if isinstance(dynamics_updates, dict) and isinstance(dynamics_updates.get('forward_bridge'), dict):
        fb = dynamics_updates.pop('forward_bridge')
        for k, v in fb.items():
            dynamics_updates.setdefault(f'forward_bridge_{k}', v)

    for key, value in data.items():
        if not hasattr(FLAGS, key):
            raise ValueError(f'Unknown YAML top-level key: {key!r}')
        if _argv_sets_flag(key):
            continue
        setattr(FLAGS, key, value)
    return dynamics_updates or {}, critic_updates or {}, actor_updates or {}


def _setup_file_logger(run_dir: str, *, resume_step: int = 0) -> tuple[logging.Logger, str]:
    if resume_step > 0:
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_name = f'run_resume_from_step{int(resume_step)}_{ts}.log'
    else:
        log_name = 'run.log'
    log_path = os.path.join(run_dir, log_name)
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger, log_path


def _update_config(config: Any, updates: dict) -> Any:
    for key, value in updates.items():
        config[key] = value
    return config


def _accumulate_metric_sums(metric_sums: dict, info: dict | None) -> None:
    """Accumulate scalars *on device*; ``float()`` is deferred to log-emit time.

    Avoids one host sync per metric per step (~30 syncs/step otherwise), which
    serialised the GPU pipeline against the Python loop.
    """
    if info is None:
        return
    for key, value in info.items():
        prev = metric_sums.get(key)
        metric_sums[key] = value if prev is None else prev + value


def _emit_metric_means(metrics: dict[str, float], prefix: str, metric_sums: dict, count: int) -> None:
    if count < 1 or not metric_sums:
        return
    inv = 1.0 / float(count)
    # Single device→host transfer for the whole tree avoids one sync per key.
    host_sums = jax.device_get(metric_sums)
    for key, total in host_sums.items():
        metrics[f'{prefix}/{key}_interval_mean'] = float(total) * inv


def _to_host_metrics(prefix: str, info: dict | None) -> dict[str, float]:
    if not info:
        return {}
    host = jax.device_get(info)
    return {f'{prefix}/{k}': float(v) for k, v in host.items()}


def _accumulate_time_sums(time_sums: dict[str, float], values: dict[str, float] | None) -> None:
    if values is None:
        return
    for key, value in values.items():
        time_sums[key] = time_sums.get(key, 0.0) + float(value)


def _emit_time_sums(metrics: dict[str, float], prefix: str, time_sums: dict[str, float], count: int) -> None:
    if count < 1:
        return
    for key, total in time_sums.items():
        metrics[f'{prefix}/{key}_interval_sec'] = float(total)
        metrics[f'{prefix}/{key}_step_sec'] = float(total / count)


def _format_step_log(metrics: dict[str, float]) -> str:
    parts = [
        f"dyn={metrics['train/dynamics/phase1/loss_interval_mean']:.6f}",
        f"critic={metrics['train/critic/total_loss_interval_mean']:.6f}",
    ]
    for label, key in (
        ('actor', 'train/actor/spi_actor/actor_loss_interval_mean'),
        ('coupling', 'train/coupling/critic_score_mean_interval_mean'),
    ):
        if key in metrics:
            parts.append(f'{label}={metrics[key]:.6f}')

    detail_keys = [
        ('dyn_g', 'train/dynamics/phase1/loss_dynamics_interval_mean'),
        ('dyn_path', 'train/dynamics/phase1/loss_path_step_interval_mean'),
        ('dyn_roll', 'train/dynamics/phase1/loss_roll_interval_mean'),
        ('dyn_sub', 'train/dynamics/phase1/loss_subgoal_interval_mean'),
        ('dyn_idm', 'train/dynamics/phase1/loss_idm_interval_mean'),
        ('fb_path', 'train/dynamics/forward_bridge/loss_path_interior_interval_mean'),
        ('fb_next', 'train/dynamics/forward_bridge/loss_path_next_interval_mean'),
        ('critic_chunk', 'train/critic/chunk_critic/critic_loss_interval_mean'),
        ('critic_distill', 'train/critic/action_critic/distill_loss_interval_mean'),
        ('critic_value', 'train/critic/action_critic/value_loss_interval_mean'),
        ('t_data', 'time/data_interval_sec'),
        ('t_dyn', 'time/dynamics_update_interval_sec'),
        ('t_critic', 'time/critic_update_interval_sec'),
        ('t_interval', 'time/interval_compute_sec'),
    ]
    actor_detail_keys = [
        ('actor_q', 'train/actor/spi_actor/q_mean_interval_mean'),
        ('actor_prox', 'train/actor/spi_actor/prox_mean_interval_mean'),
        ('actor_entropy', 'train/actor/spi_actor/rho_entropy_interval_mean'),
        ('t_build', 'time/build_batches_interval_sec'),
        ('t_prop', 'time/build/proposal_build_interval_sec'),
        ('t_sg', 'time/build/predict_subgoal_interval_sec'),
        ('t_mean', 'time/build/mean_ode_interval_sec'),
        ('t_plan', 'time/build/plan_det_interval_sec'),
        ('t_sample', 'time/build/sample_plan_interval_sec'),
        ('t_idm', 'time/build/idm_interval_sec'),
        ('t_score', 'time/build/score_interval_sec'),
        ('t_actor_rescore', 'time/actor_rescore_interval_sec'),
        ('t_actor', 'time/actor_update_interval_sec'),
    ]
    for label, key in detail_keys + actor_detail_keys:
        if key in metrics:
            parts.append(f'{label}={metrics[key]:.6f}')
    return ' '.join(parts)


def _apply_horizon(dynamics_config: Any, critic_config: Any) -> tuple[Any, Any]:
    horizon = int(FLAGS.horizon)
    if horizon < 1:
        raise ValueError(f'horizon must be >= 1, got {horizon}.')
    dynamics_config['dynamics_N'] = horizon
    dynamics_config['subgoal_steps'] = horizon
    critic_config['full_chunk_horizon'] = horizon
    return dynamics_config, critic_config


def _env_max_episode_steps(env: Any) -> int:
    """Return the environment episode cap advertised by Gym/Gymnasium wrappers."""
    spec = getattr(env, 'spec', None)
    max_steps = getattr(spec, 'max_episode_steps', None) if spec is not None else None
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is None:
        raise ValueError(
            'max_goal_steps="env" requested, but the environment does not expose max_episode_steps.'
        )
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f'env max_episode_steps must be >= 1, got {max_steps}.')
    return max_steps


def _resolve_max_goal_steps_from_env(config: Any, env: Any) -> bool:
    """Resolve ``max_goal_steps: env`` in-place. Return True when resolved."""
    value = config.get('max_goal_steps', None)
    if isinstance(value, str) and value.lower() in ('env', 'env_max_episode_steps', 'max_episode_steps'):
        with config.ignore_type():
            config['max_goal_steps'] = _env_max_episode_steps(env)
        return True
    return False


def _require_matching_frame_stack(dynamics_config: Any, critic_config: Any) -> None:
    frame_stacks = {
        'dynamics': dynamics_config.get('frame_stack', None),
        'critic': critic_config.get('frame_stack', None),
    }
    if len({str(v) for v in frame_stacks.values()}) != 1:
        raise ValueError(f'Training requires matching frame_stack across modules, got {frame_stacks}.')


def _make_critic_dataset(train_plain: dict, critic_config: Any):
    dataset = Dataset.create(**train_plain)
    return CriticSequenceDataset(dataset, critic_config)


def _intersect_valid_starts(dynamics_dataset: PathHGCDataset, critic_dataset: Any) -> np.ndarray:
    common = np.intersect1d(dynamics_dataset.path_valid_idxs, critic_dataset.valid_starts, assume_unique=False)
    common = np.asarray(common, dtype=np.int64)
    if len(common) == 0:
        raise ValueError('No shared valid starts across dynamics and critic datasets.')
    return common


def _sample_shared_idxs(common_valid_starts: np.ndarray, batch_size: int) -> np.ndarray:
    picked = np.random.randint(len(common_valid_starts), size=batch_size)
    return common_valid_starts[picked]


def _prepare_train_batch(
    common_valid_starts: np.ndarray,
    batch_size: int,
    dynamics_dataset,
    critic_dataset,
):
    """Sample one (dynamics, critic) batch from the host datasets.

    Pure CPU work; safe to run on a prefetch worker thread because the only
    shared state is NumPy's global RNG, and we serialize prefetch through a
    single-worker executor so randint() calls remain deterministic in order.
    """
    idxs = _sample_shared_idxs(common_valid_starts, batch_size)
    dynamics_batch = dynamics_dataset.sample(batch_size, idxs=idxs)
    critic_batch = critic_dataset.sample(batch_size, idxs=idxs)
    return dynamics_batch, critic_batch


def _eval_batch_size(common_valid_starts: np.ndarray, batch_size: int) -> int:
    return max(1, min(int(batch_size), int(len(common_valid_starts))))


@partial(jax.jit, static_argnames=('horizon',))
def _idm_actions_from_trajectories_jit(
    network: Any,
    trajectories: jnp.ndarray,
    horizon: int,
) -> jnp.ndarray:
    prev_states = trajectories[:, :horizon, :]
    next_states = trajectories[:, 1 : horizon + 1, :]
    flat_prev = prev_states.reshape(-1, prev_states.shape[-1])
    flat_next = next_states.reshape(-1, next_states.shape[-1])
    pred = network.select('idm_net')(flat_prev, flat_next)
    return jnp.asarray(pred, dtype=jnp.float32).reshape(trajectories.shape[0], horizon, -1)


def _idm_actions_from_trajectories(dynamics_agent: DynamicsAgent, trajectories: np.ndarray, horizon: int) -> jnp.ndarray:
    if trajectories.shape[1] <= horizon:
        raise ValueError(
            f'Dynamics trajectory length {trajectories.shape[1]} is too short for horizon={horizon}. '
            'Increase dynamics_N / subgoal_steps or reduce chunk horizons.'
        )
    trajectories = jnp.asarray(trajectories, dtype=jnp.float32)
    return dynamics_agent._idm_actions_from_trajectories(trajectories, horizon)


def _rank_candidate_actions(
    candidate_actions: jnp.ndarray,
    scores: jnp.ndarray,
    keep_topk: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    keep_topk = max(1, min(int(keep_topk), candidate_actions.shape[1]))
    order = jnp.argsort(-scores, axis=1)[:, :keep_topk]
    gather_idx = order[:, :, None, None]
    gathered = jnp.take_along_axis(candidate_actions, gather_idx, axis=1)
    gathered_scores = jnp.take_along_axis(scores, order, axis=1)
    return jnp.asarray(gathered, dtype=jnp.float32), jnp.asarray(gathered_scores, dtype=jnp.float32)


@partial(jax.jit, static_argnames=('keep_topk', 'use_partial_critic'))
def _score_and_rank_candidate_actions(
    critic_agent: Any,
    obs: jnp.ndarray,
    goals: jnp.ndarray,
    candidates: jnp.ndarray,
    network_params: Any,
    *,
    keep_topk: int,
    use_partial_critic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rescored = jnp.asarray(
        critic_agent.score_action_chunks(
            obs,
            goals,
            candidates,
            network_params=network_params,
            use_partial_critic=use_partial_critic,
        ),
        dtype=jnp.float32,
    )
    return _rank_candidate_actions(candidates, rescored, keep_topk=keep_topk)


@partial(jax.jit, static_argnames=('keep_topk', 'use_partial_critic'))
def _rescore_with_stats_jit(
    critic_agent: Any,
    obs: jnp.ndarray,
    spi_goals: jnp.ndarray,
    critic_goals: jnp.ndarray,
    candidates: jnp.ndarray,
    valids: jnp.ndarray,
    network_params: Any,
    *,
    keep_topk: int,
    use_partial_critic: bool,
) -> tuple[dict, dict]:
    """Single-graph rescore + ranking + score statistics.

    Replaces a chain of separate dispatches (score → rank → mean/max/min/gap)
    with one compiled function so that all stats live in the same XLA graph
    as the critic forward.
    """
    proposal_chunks, proposal_scores = _score_and_rank_candidate_actions(
        critic_agent,
        obs,
        critic_goals,
        candidates,
        network_params,
        keep_topk=keep_topk,
        use_partial_critic=use_partial_critic,
    )
    score_mean = proposal_scores.mean()
    score_max = proposal_scores.max()
    score_min = proposal_scores.min()
    if proposal_scores.shape[1] >= 2:
        gap = (proposal_scores[:, 0] - proposal_scores[:, 1]).mean()
    else:
        gap = jnp.zeros((), dtype=jnp.float32)
    out_batch = {
        'observations': obs,
        'spi_goals': spi_goals,
        'proposal_partial_chunks': proposal_chunks,
        'proposal_scores': proposal_scores,
        'valids': valids,
    }
    coupling_stats = {
        'critic_score_mean': score_mean,
        'critic_score_max': score_max,
        'critic_score_min': score_min,
        'critic_score_gap_top1_top2': gap,
    }
    return out_batch, coupling_stats


@partial(jax.jit, static_argnames=('use_partial_critic',))
def _rescore_top1_proposal_with_stats_jit(
    critic_agent: Any,
    obs: jnp.ndarray,
    spi_goals: jnp.ndarray,
    high_goals: jnp.ndarray,
    critic_goals: jnp.ndarray,
    candidates: jnp.ndarray,
    valids: jnp.ndarray,
    network_params: Any,
    *,
    use_partial_critic: bool,
) -> tuple[dict, dict]:
    """Score ``[B, K, ...]`` candidates and keep the global best proposal.

    Dynamics may generate ``K = U*N`` action proposals from ``U`` sampled
    subgoal endpoints and ``N`` bridge/action samples per endpoint.  The SPI
    actor should condition on the subgoal associated with the winning proposal,
    so this keeps one global best candidate and forwards its goal as
    ``spi_goals``.
    """
    q_scores = jnp.asarray(
        critic_agent.score_action_chunks(
            obs,
            critic_goals,
            candidates,
            network_params=network_params,
            use_partial_critic=use_partial_critic,
        ),
        dtype=jnp.float32,
    )
    if hasattr(critic_agent, 'score_transitive_subgoals'):
        v_scores = jnp.asarray(
            critic_agent.score_transitive_subgoals(
                obs,
                critic_goals,
                high_goals,
                network_params=network_params,
            ),
            dtype=jnp.float32,
        )
    else:
        v_scores = jnp.zeros_like(q_scores)
    mode = str(critic_agent.config.get('proposal_score_mode', 'q_only')).lower()
    if mode == 'q_only':
        scores = q_scores
    elif mode == 'v_only':
        scores = v_scores
    elif mode == 'q_plus_v':
        scores = (
            float(critic_agent.config.get('proposal_q_weight', 1.0)) * q_scores
            + float(critic_agent.config.get('proposal_v_weight', 1.0)) * v_scores
        )
    else:
        raise ValueError(
            "proposal_score_mode must be one of 'q_only', 'v_only', or 'q_plus_v', "
            f"got {mode!r}."
        )
    best_idx = jnp.argmax(scores, axis=1)
    best_chunks = jnp.take_along_axis(candidates, best_idx[:, None, None, None], axis=1)
    best_scores = jnp.take_along_axis(scores, best_idx[:, None], axis=1)
    if critic_goals is not None and critic_goals.ndim == 3:
        best_goals = jnp.take_along_axis(critic_goals, best_idx[:, None, None], axis=1)[:, 0, :]
    else:
        best_goals = spi_goals

    score_mean = best_scores.mean()
    score_max = best_scores.max()
    score_min = best_scores.min()
    if scores.shape[1] >= 2:
        sorted_scores = jnp.sort(scores, axis=1)[:, ::-1]
        gap = (sorted_scores[:, 0] - sorted_scores[:, 1]).mean()
    else:
        gap = jnp.zeros((), dtype=jnp.float32)

    out_batch = {
        'observations': obs,
        'spi_goals': jnp.asarray(best_goals, dtype=jnp.float32),
        'proposal_partial_chunks': jnp.asarray(best_chunks, dtype=jnp.float32),
        'proposal_scores': jnp.asarray(best_scores, dtype=jnp.float32),
        'valids': valids,
    }
    coupling_stats = {
        'critic_score_mean': score_mean,
        'critic_score_max': score_max,
        'critic_score_min': score_min,
        'critic_score_gap_top1_top2': gap,
        'critic_score_pre_best_mean': scores.mean(),
        'critic_score_pre_best_max': scores.max(),
        'coupling/proposal_q_score_mean': q_scores.mean(),
        'coupling/proposal_v_score_mean': v_scores.mean(),
        'coupling/proposal_combined_score_mean': scores.mean(),
    }
    return out_batch, coupling_stats


@jax.jit
def _proposal_goal_stats_jit(
    actor_goal_mean: jnp.ndarray,
    candidate_goals: jnp.ndarray,
) -> dict:
    """Compute proposal-goal coupling stats in a single fused dispatch."""
    return {
        'predicted_subgoal_norm': jnp.linalg.norm(actor_goal_mean, axis=-1).mean(),
        'proposal_goal_norm_mean': jnp.linalg.norm(candidate_goals, axis=-1).mean(),
        'proposal_goal_std_mean': candidate_goals.std(axis=1).mean(),
    }


def _build_actor_batch_from_dynamics(
    dynamics_agent: DynamicsAgent,
    critic_agent: Any,
    dynamics_batch: dict,
    actor_config: Any,
) -> tuple[DynamicsAgent, dict, dict, dict[str, float]]:
    obs = jnp.asarray(dynamics_batch['observations'], dtype=jnp.float32)
    high_goals = jnp.asarray(dynamics_batch['high_actor_goals'], dtype=jnp.float32)
    measure_timing = bool(FLAGS.measure_timing)
    timing = {}

    plan_candidates = 1
    proposal_horizon = int(actor_config['actor_chunk_horizon'])
    if measure_timing:
        t0 = time.perf_counter()
    actor_goal_mean, candidate_actions, candidate_goals, plan_rng = dynamics_agent.build_actor_proposals(
        obs,
        high_goals,
        dynamics_agent.rng,
        proposal_horizon=proposal_horizon,
        plan_candidates=plan_candidates,
        sample_noise_scale=0.0,
    )
    if measure_timing:
        _block_until_ready((actor_goal_mean, candidate_actions, candidate_goals, plan_rng))
        timing['proposal_build'] = time.perf_counter() - t0
    else:
        timing = {}
    dynamics_agent = dynamics_agent.replace(rng=plan_rng)
    # ``spi_goals`` is provisional here.  Rescoring replaces it with the
    # subgoal attached to the single best proposal before the actor update.
    use_mean_for_actor = bool(dynamics_agent.config.get('subgoal_use_mean_for_actor_goal', True))
    spi_goals = actor_goal_mean if use_mean_for_actor else high_goals
    actor_batch = {
        'observations': obs,
        'spi_goals': spi_goals,
        # Candidate action chunks generated from dynamics proposals; rescored after critic update.
        # Shape: [B, N, ha, A]
        'candidate_partial_chunks': candidate_actions,
        'valids': jnp.ones((obs.shape[0], proposal_horizon), dtype=jnp.float32),
        # Per-candidate sub-goal endpoints for critic rescoring (deterministic mode: mean broadcast).
        # Shape: [B, N, D]
        'candidate_goals': candidate_goals,
        'high_actor_goals': high_goals,
        # Each subgoal contributes ``plan_candidates`` bridge/action samples.
        # Rescoring keeps the global best proposal across the full candidate axis.
        'candidate_group_size': plan_candidates,
    }

    nan = jnp.full((), jnp.nan, dtype=jnp.float32)
    proposal_goal_stats = _proposal_goal_stats_jit(actor_goal_mean, candidate_goals)
    coupling_info = {
        **proposal_goal_stats,
        'critic_score_mean': nan,
        'critic_score_max': nan,
        'critic_score_min': nan,
        'critic_score_gap_top1_top2': nan,
        'proposal_count': jnp.asarray(float(candidate_actions.shape[1]), dtype=jnp.float32),
    }
    return dynamics_agent, actor_batch, coupling_info, timing


def _build_spi_actor_batch_from_dynamics(
    dynamics_agent: DynamicsAgent,
    critic_agent: Any,
    dynamics_batch: dict,
    actor_config: Any,
    *,
    spi_num_subgoal_samples: int,
    spi_subgoal_temperature: float,
    plan_candidates: int | None = None,
) -> tuple[DynamicsAgent, dict, dict, dict[str, float]]:
    """SPI actor batch: sample N subgoals @ T → critic best → bridge/IDM proposals."""
    obs = jnp.asarray(dynamics_batch['observations'], dtype=jnp.float32)
    high_goals = jnp.asarray(dynamics_batch['high_actor_goals'], dtype=jnp.float32)
    proposal_horizon = int(actor_config['actor_chunk_horizon'])
    plan_n = max(1, int(1 if plan_candidates is None else plan_candidates))
    noise_scale = 0.0
    num_subgoals = max(1, int(spi_num_subgoal_samples))
    score_mode = str(dynamics_agent.config.get('subgoal_eval_score_mode', 'product')).lower()

    sub_rng, plan_rng = jax.random.split(dynamics_agent.rng)
    subgoal_candidates, _ = dynamics_agent.sample_subgoal_candidates(
        obs,
        high_goals,
        sub_rng,
        num_candidates=num_subgoals,
        include_mean=False,
        temperature_override=float(spi_subgoal_temperature),
    )
    subgoal_scores = critic_agent.score_transitive_subgoals(
        obs,
        subgoal_candidates,
        high_goals,
        network_params=critic_agent.network.params,
        score_mode=score_mode,
    )
    best_subgoal_idx = jnp.argmax(subgoal_scores, axis=1)
    best_subgoal = jnp.take_along_axis(
        subgoal_candidates,
        best_subgoal_idx[:, None, None],
        axis=1,
    )[:, 0, :]

    candidate_actions, plan_rng = dynamics_agent.build_action_proposals_from_subgoal(
        obs,
        best_subgoal,
        high_goals,
        plan_rng,
        proposal_horizon=proposal_horizon,
        plan_candidates=plan_n,
        sample_noise_scale=noise_scale,
    )
    dynamics_agent = dynamics_agent.replace(rng=plan_rng)
    actor_batch = {
        'observations': obs,
        'spi_goals': jnp.asarray(best_subgoal, dtype=jnp.float32),
        'candidate_partial_chunks': candidate_actions,
        'valids': jnp.ones((obs.shape[0], proposal_horizon), dtype=jnp.float32),
        'high_actor_goals': high_goals,
    }
    nan = jnp.full((), jnp.nan, dtype=jnp.float32)
    coupling_info = {
        'predicted_subgoal_norm': jnp.linalg.norm(best_subgoal, axis=-1).mean(),
        'proposal_goal_norm_mean': jnp.linalg.norm(subgoal_candidates, axis=-1).mean(),
        'proposal_goal_std_mean': subgoal_candidates.std(axis=1).mean(),
        'critic_score_mean': nan,
        'critic_score_max': nan,
        'critic_score_min': nan,
        'critic_score_gap_top1_top2': nan,
        'proposal_count': jnp.asarray(float(candidate_actions.shape[1]), dtype=jnp.float32),
        'spi_subgoal_num_samples': jnp.asarray(float(num_subgoals), dtype=jnp.float32),
        'spi_subgoal_temperature': jnp.asarray(float(spi_subgoal_temperature), dtype=jnp.float32),
    }
    return dynamics_agent, actor_batch, coupling_info, {}


def _rescore_spi_actor_batch_for_update(actor_batch: dict, critic_agent: Any, actor_config: Any) -> tuple[dict, dict]:
    """Rescore all bridge/IDM proposals and keep every candidate for SPI rho weighting."""
    del actor_config
    obs = jnp.asarray(actor_batch['observations'], dtype=jnp.float32)
    goals = jnp.asarray(actor_batch['spi_goals'], dtype=jnp.float32)
    high_goals = jnp.asarray(actor_batch.get('high_actor_goals', goals), dtype=jnp.float32)
    candidates = jnp.asarray(actor_batch['candidate_partial_chunks'], dtype=jnp.float32)
    valids = jnp.asarray(actor_batch['valids'], dtype=jnp.float32)
    keep_topk = int(candidates.shape[1])
    out_batch, stats = _rescore_with_stats_jit(
        critic_agent,
        obs,
        goals,
        goals,
        candidates,
        valids,
        critic_agent.network.params,
        keep_topk=keep_topk,
        use_partial_critic=True,
    )
    stats = dict(stats)
    stats['proposal_best_of_n'] = jnp.asarray(float(candidates.shape[1]), dtype=jnp.float32)
    stats['proposal_pre_best_count'] = jnp.asarray(float(candidates.shape[1]), dtype=jnp.float32)
    stats['proposal_post_best_count'] = jnp.asarray(float(keep_topk), dtype=jnp.float32)
    stats['proposal_count'] = stats['proposal_post_best_count']
    stats['coupling/proposal_q_score_mean'] = stats.get('critic_score_mean', jnp.zeros((), dtype=jnp.float32))
    stats['coupling/proposal_v_score_mean'] = jnp.zeros((), dtype=jnp.float32)
    stats['coupling/proposal_combined_score_mean'] = stats.get('critic_score_mean', jnp.zeros((), dtype=jnp.float32))
    del high_goals
    return out_batch, stats


def _rescore_actor_batch_for_update(actor_batch: dict, critic_agent: Any, actor_config: Any) -> tuple[dict, dict]:
    obs = jnp.asarray(actor_batch['observations'], dtype=jnp.float32)
    goals = jnp.asarray(actor_batch['spi_goals'], dtype=jnp.float32)
    high_goals = jnp.asarray(actor_batch.get('high_actor_goals', goals), dtype=jnp.float32)
    candidates = jnp.asarray(actor_batch['candidate_partial_chunks'], dtype=jnp.float32)  # [B, N, ha, A]
    valids = jnp.asarray(actor_batch['valids'], dtype=jnp.float32)
    # Optional per-candidate sub-goal endpoints (distributional subgoal mode).
    cand_goals_in = actor_batch.get('candidate_goals', None)
    if cand_goals_in is not None:
        critic_goals = jnp.asarray(cand_goals_in, dtype=jnp.float32)  # [B, N, D]
    else:
        critic_goals = goals  # [B, D] - shared
    force_rescore_single = False
    if hasattr(critic_agent, '_is_trl'):
        force_rescore_single = bool(critic_agent._is_trl())
    force_rescore_single = force_rescore_single or bool(
        critic_agent.config.get('rescore_single_candidate', False)
    )
    # Fast path: single candidate -> skip critic call entirely (unless forced).
    if candidates.shape[1] == 1 and not force_rescore_single:
        zero = jnp.zeros((), dtype=jnp.float32)
        if cand_goals_in is not None:
            selected_goals = critic_goals[:, 0, :] if critic_goals.ndim == 3 else critic_goals
        else:
            selected_goals = goals
        return (
            {
                'observations': obs,
                'spi_goals': selected_goals,
                'proposal_partial_chunks': candidates,
                'proposal_scores': jnp.zeros((obs.shape[0], 1), dtype=jnp.float32),
                'valids': valids,
            },
            {
                'critic_score_mean': zero,
                'critic_score_max': zero,
                'critic_score_min': zero,
                'critic_score_gap_top1_top2': zero,
                'critic_score_pre_best_mean': zero,
                'critic_score_pre_best_max': zero,
                'proposal_best_of_n': jnp.asarray(1.0, dtype=jnp.float32),
                'proposal_pre_best_count': jnp.asarray(1.0, dtype=jnp.float32),
                'proposal_post_best_count': jnp.asarray(1.0, dtype=jnp.float32),
                'proposal_count': jnp.asarray(1.0, dtype=jnp.float32),
                'coupling/proposal_q_score_mean': zero,
                'coupling/proposal_v_score_mean': zero,
                'coupling/proposal_combined_score_mean': zero,
            },
        )
    # Multi-candidate path: score all U*N proposals and keep one global best.
    out_batch, stats = _rescore_top1_proposal_with_stats_jit(
        critic_agent,
        obs,
        goals,
        high_goals,
        critic_goals,
        candidates,
        valids,
        critic_agent.network.params,
        use_partial_critic=True,
    )
    stats = dict(stats)
    stats['proposal_best_of_n'] = jnp.asarray(float(candidates.shape[1]), dtype=jnp.float32)
    stats['proposal_pre_best_count'] = jnp.asarray(float(candidates.shape[1]), dtype=jnp.float32)
    stats['proposal_post_best_count'] = jnp.asarray(1.0, dtype=jnp.float32)
    stats['proposal_count'] = stats['proposal_post_best_count']
    return out_batch, stats


def _build_train_batches(
    dynamics_agent: DynamicsAgent,
    critic_agent: Any,
    dynamics_batch: dict,
    critic_batch: dict,
    actor_config: Any,
) -> tuple[DynamicsAgent, dict, dict, dict, dict[str, float]]:
    dynamics_agent, actor_batch, coupling_info, build_timing = _build_actor_batch_from_dynamics(
        dynamics_agent, critic_agent, dynamics_batch, actor_config
    )
    return dynamics_agent, critic_batch, actor_batch, coupling_info, build_timing


def _merge_actor_updates(actor_config: Any, actor_updates: dict) -> Any:
    ignored = sorted(k for k in actor_updates.keys() if k not in _SPI_ACTOR_KEYS)
    if ignored:
        logging.warning('Ignoring deprecated non-SPI actor keys: %s', ', '.join(ignored))
    for key in _SPI_ACTOR_KEYS:
        if key in actor_updates:
            actor_config[key] = actor_updates[key]
    return actor_config


def _prepare_configs(dynamics_updates: dict, critic_updates: dict, actor_updates: dict):
    dynamics_config = _update_config(get_dynamics_config(), dynamics_updates)
    critic_config = _update_config(get_critic_config(), critic_updates)
    actor_config = _merge_actor_updates(get_actor_config(), actor_updates)
    dynamics_config, critic_config = _apply_horizon(dynamics_config, critic_config)
    actor_config['actor_chunk_horizon'] = int(critic_config['action_chunk_horizon'])
    # Subgoal-value bonus net shares parameters with the critic value net, so its
    # architecture must mirror the critic; force-sync here so users only configure it once.
    dynamics_config['subgoal_value_hidden_dims'] = tuple(int(x) for x in critic_config['value_hidden_dims'])
    dynamics_config['subgoal_value_layer_norm'] = bool(critic_config['layer_norm'])
    dynamics_config['subgoal_value_goal_representation'] = str(
        critic_config.get('goal_representation', 'full'),
    )
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    critic_config['phi_goal_obs_indices'] = phi_idxs
    dynamics_config['phi_goal_obs_indices'] = phi_idxs
    # Propagate env_name so that 'phi' goal representation can dispatch to the
    # correct ManipSpace oracle layout (cube xyz vs. puzzle binary button state).
    env_name_for_phi = str(FLAGS.env_name)
    dynamics_config['env_name'] = env_name_for_phi
    critic_config['env_name'] = env_name_for_phi
    validate_config(critic_config, actor_config)
    shared_batch = int(FLAGS.batch_size)
    if shared_batch < 1:
        raise ValueError(f'batch_size must be >= 1, got {shared_batch}.')
    dynamics_config['batch_size'] = shared_batch
    critic_config['batch_size'] = shared_batch
    actor_config['batch_size'] = shared_batch
    _require_matching_frame_stack(dynamics_config, critic_config)
    return dynamics_config, critic_config, actor_config


def _create_critic_agent(seed: int, ex: dict, critic_config):
    return CriticAgent.create(
        seed,
        ex['observations'],
        None,
        ex['action_chunk_actions'],
        critic_config,
        ex_goals=ex.get('value_goals'),
    )


def _create_actor_agent(seed: int, ex_dynamics: dict, actor_config):
    return ActorAgent.create(
        seed,
        ex_dynamics['observations'],
        actor_config,
        ex_goals=ex_dynamics.get('high_actor_targets'),
    )


def _extract_critic_value_params(critic_agent: Any) -> Any | None:
    if critic_agent is None:
        return None
    if hasattr(critic_agent, '_is_trl') and bool(critic_agent._is_trl()):
        return critic_agent.network.params.get('modules_target_value', None)
    return critic_agent.network.params.get('modules_value', None)


def _idm_action_chunk(
    dynamics_agent: DynamicsAgent,
    obs: np.ndarray,
    predicted_subgoal: np.ndarray,
    horizon: int,
) -> np.ndarray:
    traj = np.asarray(dynamics_agent.plan(obs, predicted_subgoal)['trajectory'], dtype=np.float32)
    if traj.ndim != 2:
        raise RuntimeError(f'Expected single-trajectory plan with rank 2, got shape={traj.shape}.')
    action_chunk = np.asarray(_idm_actions_from_trajectories(dynamics_agent, traj[None, ...], horizon), dtype=np.float32)
    return action_chunk[0]


def _log_eval_run_logger(
    run_logger: logging.Logger,
    *,
    step: int,
    eval_task_ids: tuple[int, ...],
    metrics: dict[str, Any],
    header: str,
    idm_only: bool = False,
) -> None:
    run_logger.info(
        '%s step=%d num_tasks=%d stat_episodes_per_task=%d',
        header,
        step,
        int(metrics.get('eval/num_tasks', 0.0)),
        int(metrics.get('eval/episodes_per_task', 0.0)),
    )
    run_logger.info('[IDM POLICY] primary_success=any_step_info_success')
    run_logger.info('idm env_success_rate_mean=%.2f', metrics.get('eval_idm/success_rate_mean', float('nan')))
    for task_id in eval_task_ids:
        task_key = f'eval_idm/task_{task_id}/success_rate'
        if task_key in metrics:
            run_logger.info('idm task_%d env=%.2f', task_id, metrics[task_key])
    if idm_only:
        return
    run_logger.info('[ACTOR POLICY] (same success definition)')
    run_logger.info('actor env_success_rate_mean=%.2f', metrics.get('eval/success_rate_mean', float('nan')))
    for task_id in eval_task_ids:
        task_key = f'eval/task_{task_id}/success_rate'
        if task_key in metrics:
            run_logger.info('actor task_%d env=%.2f', task_id, metrics[task_key])
    for prefix, label in (
        ('eval_flow_idm', 'flow+idm'),
        ('eval_flow_actor', 'flow+actor'),
    ):
        mean_key = f'{prefix}/success_rate_mean'
        if mean_key not in metrics:
            continue
        run_logger.info('[%s] env_success_rate_mean=%.2f', label, metrics[mean_key])
        for task_id in eval_task_ids:
            task_key = f'{prefix}/task_{task_id}/success_rate'
            if task_key in metrics:
                run_logger.info('%s task_%d env=%.2f', label, task_id, metrics[task_key])


def _evaluate_env_tasks(
    env,
    dynamics_agent: DynamicsAgent,
    actor_agent: Any,
    actor_config: Any,
    critic_config: Any,
    *,
    critic_agent: Any | None = None,
    task_ids: tuple[int, ...],
    episodes_per_task: int,
    subgoal_override_goal: bool = False,
    idm_only: bool = False,
) -> dict[str, Any]:
    """OGBench-style eval: success is decided **only** by ``info['success']`` (any step). No tolerance diagnostic.

    ``idm_only`` restricts evaluation to the IDM policy (flow subgoal + inverse dynamics)
    and skips the SPI-actor rollout. Use when the actor has not been trained (see ``--train_actor_spi``).
    """
    if not task_ids:
        return {}

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    actor_horizon = int(actor_config['actor_chunk_horizon'])
    idm_horizon = int(critic_config['action_chunk_horizon'])
    metrics: dict[str, Any] = {}

    num_eval = max(0, int(episodes_per_task))
    if num_eval < 1:
        raise ValueError('episodes_per_task (stat eval) must be >= 1')

    # Eval selection is fixed to best_of_n_value (critic-scored best-of-N).
    use_eval_bon = True
    eval_seed_base = int(dynamics_agent.config.get('subgoal_eval_seed', 0))
    if idm_only:
        variants = [('flow', 'idm', 'eval_flow_idm')]
    else:
        variants = [
            ('flow', 'idm', 'eval_flow_idm'),
            ('flow', 'actor', 'eval_flow_actor'),
        ]
    task_successes_by_key: dict[str, list[float]] = {metric_key: [] for _, _, metric_key in variants}

    def _eval_subgoal(obs: np.ndarray, goal: np.ndarray, *, ep_ix: int, source: str) -> np.ndarray:
        del source
        if use_eval_bon:
            rng = jax.random.PRNGKey(eval_seed_base + int(ep_ix))
            pred = dynamics_agent.infer_subgoal_for_eval(
                jnp.asarray(obs, dtype=jnp.float32),
                jnp.asarray(goal, dtype=jnp.float32),
                critic_agent=critic_agent,
                rng=rng,
            )
            return np.asarray(pred, dtype=np.float32).reshape(-1)
        return np.asarray(
            dynamics_agent.infer_subgoal(obs, goal),
            dtype=np.float32,
        ).reshape(-1)

    def _actor_chunk(obs: np.ndarray, goal: np.ndarray, *, ep_ix: int, source: str) -> np.ndarray:
        if bool(subgoal_override_goal):
            pred = np.asarray(goal, dtype=np.float32).reshape(-1)
        else:
            pred = _eval_subgoal(obs, goal, ep_ix=ep_ix, source=source)
        return np.asarray(actor_agent.sample_actions(obs, pred), dtype=np.float32).reshape(actor_horizon, -1)

    def _idm_chunk(obs: np.ndarray, goal: np.ndarray, *, ep_ix: int, source: str) -> np.ndarray:
        if bool(subgoal_override_goal):
            pred = np.asarray(goal, dtype=np.float32).reshape(-1)
        else:
            pred = _eval_subgoal(obs, goal, ep_ix=ep_ix, source=source)
        return _idm_action_chunk(dynamics_agent, obs, pred, idm_horizon)

    for task_id in task_ids:
        episode_successes_by_key: dict[str, list[float]] = {metric_key: [] for _, _, metric_key in variants}

        for ep_ix in range(num_eval):
            for subgoal_source, policy, metric_key in variants:
                ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
                if 'goal' not in info:
                    raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
                obs = np.asarray(ob, dtype=np.float32).reshape(-1)
                goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

                if policy == 'actor':
                    sampler = lambda o, g, source=subgoal_source: _actor_chunk(o, g, ep_ix=ep_ix, source=source)
                else:
                    sampler = lambda o, g, source=subgoal_source: _idm_chunk(o, g, ep_ix=ep_ix, source=source)
                ok_env = rollout_chunked_eval_episode(
                    env,
                    obs,
                    goal,
                    low,
                    high,
                    sample_action_chunk=sampler,
                )
                episode_successes_by_key[metric_key].append(1.0 if ok_env else 0.0)

        for _source, _policy, metric_key in variants:
            task_success_rate = float(np.mean(episode_successes_by_key[metric_key]))
            metrics[f'{metric_key}/task_{task_id}/success_rate'] = task_success_rate
            task_successes_by_key[metric_key].append(task_success_rate)

        primary_idm_key = 'eval_flow_idm'
        idm_task_success_rate = metrics[f'{primary_idm_key}/task_{task_id}/success_rate']
        metrics[f'eval_idm/task_{task_id}/success_rate'] = idm_task_success_rate
        metrics[f'evaluation/idm_task_{task_id}_success'] = idm_task_success_rate
        if not idm_only:
            primary_actor_key = 'eval_flow_actor'
            actor_task_success_rate = metrics[f'{primary_actor_key}/task_{task_id}/success_rate']
            metrics[f'eval/task_{task_id}/success_rate'] = actor_task_success_rate
            metrics[f'evaluation/task_{task_id}_success'] = actor_task_success_rate

    for _source, _policy, metric_key in variants:
        metrics[f'{metric_key}/success_rate_mean'] = float(np.mean(task_successes_by_key[metric_key]))
    metrics['eval_idm/success_rate_mean'] = metrics['eval_flow_idm/success_rate_mean']
    metrics['evaluation/overall_idm_success'] = metrics['eval_idm/success_rate_mean']
    if not idm_only:
        metrics['eval/success_rate_mean'] = metrics['eval_flow_actor/success_rate_mean']
        metrics['evaluation/overall_success'] = metrics['eval/success_rate_mean']
    metrics['eval/num_tasks'] = float(len(task_ids))
    metrics['eval/episodes_per_task'] = float(num_eval)

    return metrics


def main(_):
    impl = _impl_dir()
    resume_run_dir = FLAGS.resume_run_dir.strip()
    resume_step = int(FLAGS.resume_step)
    if resume_step < 0:
        resume_step = int(FLAGS.resume_epoch)
    if resume_step < 0:
        raise ValueError('resume_step must be >= 0.')
    if resume_step > 0 and not resume_run_dir:
        raise ValueError('resume_step > 0 requires resume_run_dir.')
    restoring_ckpt = bool(resume_run_dir and resume_step > 0)

    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    resume_snapshot_path: str | None = None
    if (
        resume_run_dir
        and FLAGS.resume_use_run_snapshot_config
        and not _argv_sets_flag('run_config')
    ):
        resume_snapshot_path = _resolve_resume_snapshot_config_path(os.path.abspath(resume_run_dir))
        if resume_snapshot_path is not None:
            cfg_path = resume_snapshot_path
        else:
            print(
                f'[train] WARN resume_use_run_snapshot_config but no flags.json or config_used.yaml '
                f'in {resume_run_dir!r}; using default run_config: {cfg_path}',
                file=sys.stderr,
            )

    dynamics_updates, critic_updates, actor_updates = {}, {}, {}
    if os.path.isfile(cfg_path):
        dynamics_updates, critic_updates, actor_updates = _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')
    else:
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    dynamics_config, critic_config, actor_config = _prepare_configs(
        dynamics_updates,
        critic_updates,
        actor_updates,
    )

    runs_root = FLAGS.runs_root.strip() or os.path.join(impl, 'runs')
    if resume_run_dir:
        run_dir = os.path.abspath(resume_run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f'resume_run_dir not found: {run_dir}')
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        env_tok = _sanitize_token(FLAGS.env_name)
        run_folder = f'{ts}_seed{FLAGS.seed}_{env_tok}'
        run_dir = os.path.join(runs_root, run_folder)
    ckpt_root = os.path.join(run_dir, 'checkpoints')
    dynamics_ckpt_dir = os.path.join(ckpt_root, 'dynamics')
    critic_ckpt_dir = os.path.join(ckpt_root, 'critic')
    actor_ckpt_dir = os.path.join(ckpt_root, 'actor')
    os.makedirs(dynamics_ckpt_dir, exist_ok=True)
    os.makedirs(critic_ckpt_dir, exist_ok=True)
    os.makedirs(actor_ckpt_dir, exist_ok=True)
    if os.path.isfile(cfg_path) and not resume_run_dir:
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name='train')
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    run_logger, run_log_path = _setup_file_logger(run_dir, resume_step=resume_step if restoring_ckpt else 0)
    run_logger.info('run_dir=%s', run_dir)
    run_logger.info('log_path=%s', run_log_path)
    if resume_snapshot_path is not None:
        run_logger.info('resume hyperparameters from snapshot file: %s', resume_snapshot_path)
    # Force PJRT/CUDA initialisation before MuJoCo/EGL environment creation.
    # In some shells, delaying the first JAX device touch until after env setup
    # can make cuInit fail and silently fall back to CPU.
    jax_devices = jax.devices()
    run_logger.info('jax_backend=%s jax_devices=%s', jax.default_backend(), jax_devices)
    _require_gpu_jax(run_logger)

    env, train_plain, _ = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=critic_config['frame_stack'],
        dataset_dir=FLAGS.dataset_dir,
        render_mode='rgb_array',
    )
    obs_dim_env = int(np.prod(env.observation_space.shape))
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    if not phi_idxs:
        phi_idxs = infer_phi_goal_obs_indices(str(FLAGS.env_name), obs_dim_env)
        critic_config['phi_goal_obs_indices'] = phi_idxs
        dynamics_config['phi_goal_obs_indices'] = phi_idxs
    resolved_dyn_goal_cap = _resolve_max_goal_steps_from_env(dynamics_config, env)
    resolved_critic_goal_cap = _resolve_max_goal_steps_from_env(critic_config, env)
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim
    if resolved_dyn_goal_cap or resolved_critic_goal_cap:
        run_logger.info(
            'resolved max_goal_steps from env max_episode_steps=%d (dynamics=%s critic=%s)',
            _env_max_episode_steps(env),
            dynamics_config.get('max_goal_steps', None),
            critic_config.get('max_goal_steps', None),
        )

    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(
            dict(
                flags=get_flag_dict(),
                dynamics=dynamics_config.to_dict(),
                critic_agent=critic_config.to_dict(),
                actor=actor_config.to_dict(),
            ),
            f,
            indent=2,
        )

    dynamics_dataset = PathHGCDataset(Dataset.create(**train_plain), dynamics_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_config)
    common_valid_starts = _intersect_valid_starts(dynamics_dataset, critic_dataset)
    if int(dynamics_config['dynamics_N']) < int(actor_config['actor_chunk_horizon']):
        raise ValueError(
            f'dynamics_N={int(dynamics_config["dynamics_N"])} must be >= actor_chunk_horizon={int(actor_config["actor_chunk_horizon"])} '
            'for critic-ranked dynamics proposals.'
        )

    np.random.seed(FLAGS.seed)
    ex_idxs = _sample_shared_idxs(common_valid_starts, int(dynamics_config['batch_size']))
    ex_dynamics = dynamics_dataset.sample(len(ex_idxs), idxs=ex_idxs)
    ex_critic = critic_dataset.sample(len(ex_idxs), idxs=ex_idxs)

    dynamics_agent = DynamicsAgent.create(
        FLAGS.seed,
        ex_dynamics['observations'],
        dynamics_config,
        ex_actions=ex_dynamics['actions'],
    )
    critic_agent = _create_critic_agent(FLAGS.seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(FLAGS.seed, ex_dynamics, actor_config)
    if restoring_ckpt:
        dynamics_agent = restore_agent(dynamics_agent, dynamics_ckpt_dir, resume_step)
        critic_agent = restore_agent(critic_agent, critic_ckpt_dir, resume_step)
        # The SPI actor is only checkpointed when trained jointly (--train_actor_spi).
        # When it is fine-tuned separately, its checkpoint may be absent on resume.
        if bool(FLAGS.train_actor_spi) or os.path.isfile(
            os.path.join(actor_ckpt_dir, f'params_{resume_step}.pkl')
        ):
            actor_agent = restore_agent(actor_agent, actor_ckpt_dir, resume_step)

    batch_size = int(dynamics_config['batch_size'])
    spe = _steps_per_epoch(len(common_valid_starts), batch_size)
    measure_timing = bool(FLAGS.measure_timing)
    train_actor_spi = bool(FLAGS.train_actor_spi)
    train_steps = _resolve_total_steps(spe)
    log_every_steps = _resolve_step_interval(
        int(FLAGS.log_every_n_steps), int(FLAGS.log_every_n_epochs), spe
    )
    save_every_steps = _resolve_step_interval(
        int(FLAGS.save_every_n_steps), int(FLAGS.save_every_n_epochs), spe
    )
    eval_every_steps = _resolve_step_interval(
        int(FLAGS.eval_every_n_steps), int(FLAGS.eval_freq), spe, allow_disable=True
    )
    eval_task_ids = (1, 2, 3, 4, 5)
    eval_episodes_per_task = max(1, int(FLAGS.eval_episodes_per_task))
    final_eval_episodes_per_task = max(0, int(FLAGS.final_eval_episodes_per_task))
    final_eval_n_values = parse_int_list(str(FLAGS.final_eval_subgoal_eval_num_samples))
    run_logger.info(
        'shared_valid_starts=%d batch_size=%d steps_per_epoch=%d dyn_h=%d critic_h=%d actor_h=%d',
        len(common_valid_starts),
        batch_size,
        spe,
        int(dynamics_config['subgoal_steps']),
        int(critic_config.get('full_chunk_horizon', 0)),
        int(actor_config.get('actor_chunk_horizon', 0)),
    )
    run_logger.info(
        'run_setup env=%s seed=%d train_steps=%d start_step=%d log_every_steps=%d save_every_steps=%d '
        'legacy_train_epochs=%d steps_per_epoch=%d async_prefetch=%s action_dim=%d train_actor_spi=%s',
        FLAGS.env_name,
        int(FLAGS.seed),
        int(train_steps),
        int(resume_step + 1 if restoring_ckpt else 1),
        int(log_every_steps),
        int(save_every_steps),
        int(FLAGS.train_epochs),
        int(spe),
        bool(FLAGS.async_prefetch),
        action_dim,
        train_actor_spi,
    )
    run_logger.info(
        'dynamics planner=forward_bridge_residual theta_schedule=prefix_progress theta_total=1 progress_alpha=0.8 bridge_gamma_inv=%.4g lambda=%.4g',
        float(dynamics_config.get('bridge_gamma_inv', 0.0)),
        float(dynamics_config.get('dynamics_lambda', 0.0)),
    )
    run_logger.info(
        'subgoal distribution=%s stochastic_loss=%s target_mode=displacement steps=%d samples_U=%d plan_candidates_N=%d total_proposals=%d temperature=%.4g value_style=%s value_expectile=%.4g value_gap_scale=%.4g value_weight_max=%.4g use_mean_for_actor_goal=%s',
        str(dynamics_config.get('subgoal_distribution', '')),
        str(dynamics_config.get('subgoal_stochastic_loss', 'mse')),
        int(dynamics_config.get('subgoal_steps', 0)),
        int(dynamics_config.get('subgoal_num_samples', 1)),
        1,
        int(dynamics_config.get('subgoal_num_samples', 1)) * 1,
        float(dynamics_config.get('subgoal_temperature', 0.0)),
        str(dynamics_config.get('subgoal_value_style', 'exponential')),
        float(dynamics_config.get('subgoal_value_expectile', 0.7)),
        float(dynamics_config.get('subgoal_value_gap_scale', 1.0)),
        float(dynamics_config.get('subgoal_value_weight_max', 0.0)),
        bool(dynamics_config.get('subgoal_use_mean_for_actor_goal', True)),
    )
    run_logger.info(
        'subgoal_flow eval_selection=best_of_n_value eval_num_samples=%d final_eval_n_values=%s eval_include_zero_candidate=False',
        int(dynamics_config.get('subgoal_eval_num_samples', 1)),
        str(final_eval_n_values) if final_eval_n_values else 'off',
    )
    run_logger.info(
        'planner_sampling forward_bridge_mode=%s forward_bridge_use_path_loss=%s path_loss_weight=%.4g rollout_horizon=%d rollout_loss_weight=%.4g',
        str(dynamics_config.get('forward_bridge_mode', '')),
        bool(dynamics_config.get('forward_bridge_use_path_loss', True)),
        float(dynamics_config.get('path_loss_weight', 0.0)),
        int(dynamics_config.get('rollout_horizon', 0)),
        float(dynamics_config.get('rollout_loss_weight', 0.0)),
    )
    run_logger.info(
        'critic_actor type=trl critic_chunk_h=%d action_chunk_h=%d spi_tau=%.4g discount=%.4g',
        int(critic_config.get('full_chunk_horizon', 0)),
        int(critic_config.get('action_chunk_horizon', 0)),
        float(actor_config.get('spi_tau', 0.0)),
        float(critic_config.get('discount', 0.0)),
    )
    run_logger.info(
        'eval eval_every_steps=%d legacy_eval_freq=%d eval_tasks=%s eval_episodes=%d final_eval_episodes=%d '
        'env_max_episode_steps=%d primary_success=any_step_info_success',
        eval_every_steps,
        int(FLAGS.eval_freq),
        ','.join(str(x) for x in eval_task_ids),
        eval_episodes_per_task,
        final_eval_episodes_per_task,
        _env_max_episode_steps(env),
    )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'), resume=restoring_ckpt, flush_every_n=1)
    first_time = time.time()
    last_log = time.time()

    start_step = resume_step + 1 if restoring_ckpt else 1
    step_iter = range(start_step, train_steps + 1)
    if FLAGS.use_tqdm:
        step_iter = tqdm.tqdm(step_iter, smoothing=0.1, dynamic_ncols=True)

    # Async batch prefetch: a single worker thread overlaps host-side numpy
    # slicing for batch N+1 with GPU work for batch N. Single-worker
    # ThreadPoolExecutor preserves the order of np.random calls so the
    # sampling sequence stays deterministic given the seed.
    use_async_prefetch = bool(FLAGS.async_prefetch)
    prefetch_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix='train-prefetch')
        if use_async_prefetch
        else None
    )

    def _submit_prefetch() -> Future:
        return prefetch_pool.submit(
            _prepare_train_batch,
            common_valid_starts,
            batch_size,
            dynamics_dataset,
            critic_dataset,
        )

    next_batch_future: Future | None = _submit_prefetch() if prefetch_pool is not None else None

    data_time = 0.0
    build_time = 0.0
    build_detail_times = {}
    dynamics_time = 0.0
    critic_time = 0.0
    actor_rescore_time = 0.0
    actor_time = 0.0
    interval_compute_time = 0.0
    dynamics_metric_sums = {}
    critic_metric_sums = {}
    actor_metric_sums = {}
    coupling_metric_sums = {}
    last_dynamics_info = None
    last_critic_info = None
    last_actor_info = None
    last_coupling_info = None
    steps_since_log = 0

    for step in step_iter:
        if measure_timing:
            step_start = time.perf_counter()
            t0 = time.perf_counter()
        if next_batch_future is not None:
            dynamics_batch, critic_batch = next_batch_future.result()
            next_batch_future = _submit_prefetch()
        else:
            idxs = _sample_shared_idxs(common_valid_starts, batch_size)
            dynamics_batch = dynamics_dataset.sample(batch_size, idxs=idxs)
            critic_batch = critic_dataset.sample(batch_size, idxs=idxs)
        if measure_timing:
            data_time += time.perf_counter() - t0

        if train_actor_spi:
            if measure_timing:
                t0 = time.perf_counter()
            dynamics_agent, critic_batch, actor_batch, coupling_info, build_detail_info = _build_train_batches(
                dynamics_agent,
                critic_agent,
                dynamics_batch,
                critic_batch,
                actor_config,
            )
            if measure_timing:
                _block_until_ready((critic_batch, actor_batch))
                build_time += time.perf_counter() - t0
                _accumulate_time_sums(build_detail_times, build_detail_info)

        if measure_timing:
            t0 = time.perf_counter()
        dynamics_agent, dynamics_info = dynamics_agent.update(
            dynamics_batch,
            critic_value_params=_extract_critic_value_params(critic_agent),
        )
        if measure_timing:
            _block_until_ready(dynamics_info)
            dynamics_time += time.perf_counter() - t0

        if measure_timing:
            t0 = time.perf_counter()
        critic_agent, critic_info = critic_agent.update(critic_batch)
        if measure_timing:
            _block_until_ready(critic_info)
            critic_time += time.perf_counter() - t0

        if train_actor_spi:
            if measure_timing:
                t0 = time.perf_counter()
            actor_batch_for_update, score_coupling_info = _rescore_actor_batch_for_update(
                actor_batch, critic_agent, actor_config
            )
            coupling_info = dict(coupling_info)
            coupling_info.update(score_coupling_info)
            if measure_timing:
                _block_until_ready(actor_batch_for_update)
                actor_rescore_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            actor_agent, actor_info = actor_agent.update(actor_batch_for_update, critic_agent)
            if measure_timing:
                _block_until_ready(actor_info)
                actor_time += time.perf_counter() - t0
                interval_compute_time += time.perf_counter() - step_start
        else:
            actor_info = None
            coupling_info = None
            if measure_timing:
                interval_compute_time += time.perf_counter() - step_start

        last_dynamics_info = dynamics_info
        last_critic_info = critic_info
        last_actor_info = actor_info
        last_coupling_info = coupling_info
        steps_since_log += 1

        _accumulate_metric_sums(dynamics_metric_sums, dynamics_info)
        _accumulate_metric_sums(critic_metric_sums, critic_info)
        if train_actor_spi:
            _accumulate_metric_sums(actor_metric_sums, actor_info)
            _accumulate_metric_sums(coupling_metric_sums, coupling_info)

        is_final_step = step == train_steps
        do_regular_eval = bool(eval_every_steps > 0 and step % eval_every_steps == 0 and not is_final_step)
        do_final_eval = bool(is_final_step and final_eval_n_values)
        do_log = bool(step % log_every_steps == 0 or is_final_step or do_regular_eval or do_final_eval)

        if do_log and last_dynamics_info is not None and steps_since_log > 0:
            metrics = {}
            metrics.update(_to_host_metrics('train/dynamics', last_dynamics_info))
            metrics.update(_to_host_metrics('train/critic', last_critic_info))
            if train_actor_spi:
                metrics.update(_to_host_metrics('train/actor', last_actor_info))
                metrics.update(_to_host_metrics('train/coupling', last_coupling_info))
            metrics['train/critic/primary_score'] = extract_critic_primary_score(last_critic_info)
            _emit_metric_means(metrics, 'train/dynamics', dynamics_metric_sums, steps_since_log)
            _emit_metric_means(metrics, 'train/critic', critic_metric_sums, steps_since_log)
            if train_actor_spi:
                _emit_metric_means(metrics, 'train/actor', actor_metric_sums, steps_since_log)
                _emit_metric_means(metrics, 'train/coupling', coupling_metric_sums, steps_since_log)
            metrics['train/step'] = float(step)
            metrics['train/steps_per_epoch'] = float(spe)

            did_final_n_sweep = False
            if do_final_eval:
                did_final_n_sweep = True
                eval_episode_count = (
                    final_eval_episodes_per_task if final_eval_episodes_per_task > 0 else eval_episodes_per_task
                )
                with open(os.path.join(run_dir, 'flags.json'), encoding='utf-8') as f:
                    flags_root = json.load(f)
                orig_eval_n = int(dynamics_config.get('subgoal_eval_num_samples', 1))
                last_eval_metrics: dict[str, Any] = {}
                for eval_n in final_eval_n_values:
                    out_path = eval_result_path(run_dir, epoch=step, eval_n=eval_n)
                    if out_path.is_file():
                        run_logger.info('SKIP final N=%d eval (already saved %s)', eval_n, out_path)
                        with open(out_path, encoding='utf-8') as f:
                            saved = json.load(f)
                        last_eval_metrics = {
                            'eval/num_tasks': float(len(eval_task_ids)),
                            'eval/episodes_per_task': float(eval_episode_count),
                            'eval_idm/success_rate_mean': float(saved['idm_success_rate_mean']),
                            'eval/success_rate_mean': float(saved['actor_success_rate_mean']),
                        }
                        for tid in eval_task_ids:
                            idm_tasks = saved.get('idm_task_success_rates', {})
                            actor_tasks = saved.get('actor_task_success_rates', {})
                            if str(tid) in idm_tasks:
                                last_eval_metrics[f'eval_idm/task_{tid}/success_rate'] = float(idm_tasks[str(tid)])
                            if str(tid) in actor_tasks:
                                last_eval_metrics[f'eval/task_{tid}/success_rate'] = float(actor_tasks[str(tid)])
                        continue
                    dynamics_config['subgoal_eval_num_samples'] = eval_n
                    eval_dynamics_agent = dynamics_agent.replace(
                        config=dynamics_agent.config.copy({'subgoal_eval_num_samples': eval_n}),
                    )
                    feval_metrics = _evaluate_env_tasks(
                        env,
                        eval_dynamics_agent,
                        actor_agent,
                        actor_config,
                        critic_config,
                        critic_agent=critic_agent,
                        task_ids=eval_task_ids,
                        episodes_per_task=eval_episode_count,
                        subgoal_override_goal=bool(FLAGS.subgoal_override_goal),
                        idm_only=not train_actor_spi,
                    )
                    saved_path = save_eval_results(
                        run_dir,
                        epoch=step,
                        subgoal_eval_num_samples=eval_n,
                        task_ids=eval_task_ids,
                        episodes_per_task=eval_episode_count,
                        metrics=feval_metrics,
                        fg=flags_root['flags'],
                        root=flags_root,
                    )
                    run_logger.info('=== FINAL EVAL START step=%d eval_n=%d ===', step, eval_n)
                    _log_eval_run_logger(
                        run_logger,
                        step=step,
                        eval_task_ids=eval_task_ids,
                        metrics=feval_metrics,
                        header='final_eval',
                        idm_only=not train_actor_spi,
                    )
                    run_logger.info(
                        '=== FINAL EVAL END step=%d eval_n=%d saved=%s ===',
                        step,
                        eval_n,
                        saved_path,
                    )
                    last_eval_metrics = feval_metrics
                    for key, val in feval_metrics.items():
                        if key.startswith('eval') or key.startswith('evaluation'):
                            metrics[f'feval_n{eval_n}/{key}'] = val
                dynamics_config['subgoal_eval_num_samples'] = orig_eval_n
                if last_eval_metrics:
                    metrics.update(last_eval_metrics)
            elif do_regular_eval:
                eval_episode_count = eval_episodes_per_task
                metrics.update(
                    _evaluate_env_tasks(
                        env,
                        dynamics_agent,
                        actor_agent,
                        actor_config,
                        critic_config,
                        critic_agent=critic_agent,
                        task_ids=eval_task_ids,
                        episodes_per_task=eval_episode_count,
                        subgoal_override_goal=bool(FLAGS.subgoal_override_goal),
                        idm_only=not train_actor_spi,
                    )
                )

            if measure_timing:
                metrics['time/data_interval_sec'] = data_time
                metrics['time/dynamics_update_interval_sec'] = dynamics_time
                metrics['time/critic_update_interval_sec'] = critic_time
                metrics['time/interval_compute_sec'] = interval_compute_time
                metrics['time/data_step_sec'] = data_time / steps_since_log
                metrics['time/dynamics_update_step_sec'] = dynamics_time / steps_since_log
                metrics['time/critic_update_step_sec'] = critic_time / steps_since_log
                if train_actor_spi:
                    metrics['time/build_batches_interval_sec'] = build_time
                    _emit_time_sums(metrics, 'time/build', build_detail_times, steps_since_log)
                    metrics['time/actor_rescore_interval_sec'] = actor_rescore_time
                    metrics['time/actor_update_interval_sec'] = actor_time
                    metrics['time/build_batches_step_sec'] = build_time / steps_since_log
                    metrics['time/actor_rescore_step_sec'] = actor_rescore_time / steps_since_log
                    metrics['time/actor_update_step_sec'] = actor_time / steps_since_log
            metrics['time/wall_sec'] = time.time() - last_log
            metrics['time/total_sec'] = time.time() - first_time
            last_log = time.time()
            if FLAGS.use_wandb:
                import wandb

                wandb.log(metrics, step=step)
            train_logger.log(metrics, step=step)
            run_logger.info(
                'step=%d %s',
                step,
                _format_step_log(metrics),
            )
            if do_regular_eval and not did_final_n_sweep:
                num_tasks = int(metrics.get('eval/num_tasks', 0.0))
                episodes_per_task = int(metrics.get('eval/episodes_per_task', 0.0))
                run_logger.info(
                    '=== EVAL START step=%d num_tasks=%d stat_episodes_per_task=%d ===',
                    step,
                    num_tasks,
                    episodes_per_task,
                )
                _log_eval_run_logger(
                    run_logger,
                    step=step,
                    eval_task_ids=eval_task_ids,
                    metrics=metrics,
                    header='eval',
                    idm_only=not train_actor_spi,
                )
                run_logger.info('=== EVAL END step=%d ===', step)

            data_time = 0.0
            build_time = 0.0
            build_detail_times = {}
            dynamics_time = 0.0
            critic_time = 0.0
            actor_rescore_time = 0.0
            actor_time = 0.0
            interval_compute_time = 0.0
            dynamics_metric_sums = {}
            critic_metric_sums = {}
            actor_metric_sums = {}
            coupling_metric_sums = {}
            steps_since_log = 0

        if save_every_steps > 0 and (step % save_every_steps == 0 or is_final_step):
            save_agent(dynamics_agent, dynamics_ckpt_dir, step)
            save_agent(critic_agent, critic_ckpt_dir, step)
            if train_actor_spi:
                save_agent(actor_agent, actor_ckpt_dir, step)

    if prefetch_pool is not None:
        # Cancel any outstanding prefetch and tear down the worker thread.
        if next_batch_future is not None and not next_batch_future.done():
            next_batch_future.cancel()
        prefetch_pool.shutdown(wait=False, cancel_futures=True)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
