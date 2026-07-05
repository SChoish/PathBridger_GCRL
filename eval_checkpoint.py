#!/usr/bin/env python3
"""Load a training run checkpoint and run the same env eval as training (``main._evaluate_env_tasks``).

Reads ``flags.json`` + ``checkpoints/{dynamics,critic,actor}/params_<epoch>.pkl``.

Example::

    MUJOCO_GL=egl python eval_checkpoint.py \\
        --run_dir=runs/20260422_015908_seed0_antmaze-medium-navigate-v0 \\
        --epoch=1000

IDM env-eval uses ``--idm_action_chunk_horizon`` (default **5**) for ``_idm_action_chunk`` only; the
saved critic YAML may still use a larger ``action_chunk_horizon`` for training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from agents.critic import get_config as get_critic_config, validate_config
from agents.actor import get_actor_config
from agents.dynamics import DynamicsAgent, get_dynamics_config
from main import (
    _create_actor_agent,
    _create_critic_agent,
    _evaluate_env_tasks,
    _intersect_valid_starts,
    _make_critic_dataset,
    _require_matching_frame_stack,
    _sample_shared_idxs,
    _update_config,
)
from utils.datasets import Dataset, PathHGCDataset
from utils.goal_representation import infer_phi_goal_obs_indices, normalize_phi_goal_obs_indices
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    eval_result_path,
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    parse_int_list,
    pick_epoch,
    resolve_actor_checkpoint_dir,
    resolve_critic_checkpoint_dir,
    resolve_dynamics_checkpoint_dir,
    save_eval_results,
)


def _build_configs(root: dict, fg: dict) -> tuple[Any, Any, Any]:
    horizon = int(fg['horizon'])
    dynamics_config = _update_config(get_dynamics_config(), root['dynamics'])
    critic_config = _update_config(get_critic_config(), root['critic_agent'])
    # Saved ``flags.json`` holds the full merged actor dict (not only SPI keys).
    actor_config = _update_config(get_actor_config(), root['actor'])
    dynamics_config['dynamics_N'] = horizon
    dynamics_config['subgoal_steps'] = horizon
    critic_config['full_chunk_horizon'] = horizon
    actor_config['actor_chunk_horizon'] = int(critic_config['action_chunk_horizon'])
    validate_config(critic_config, actor_config)
    bs = int(fg['batch_size'])
    dynamics_config['batch_size'] = bs
    critic_config['batch_size'] = bs
    actor_config['batch_size'] = bs
    _require_matching_frame_stack(dynamics_config, critic_config)
    return dynamics_config, critic_config, actor_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--epoch', type=int, default=1000, help='Checkpoint suffix for dynamics/critic/actor.')
    p.add_argument('--seed', type=int, default=-1, help='Agent RNG seed; -1 uses flags.json flags.seed.')
    p.add_argument('--eval_task_ids', type=str, default='', help='Override e.g. "1,2,3,4,5" (empty = flags, default 1-5).')
    p.add_argument('--eval_episodes_per_task', type=int, default=-1, help='-1 = use flags.')
    p.add_argument(
        '--subgoal_eval_num_samples',
        type=int,
        default=-1,
        help='Override dynamics.subgoal_eval_num_samples for eval only; -1 = checkpoint config.',
    )
    p.add_argument(
        '--subgoal_temperature',
        type=float,
        default=-1.0,
        help='Override dynamics.subgoal_temperature for eval only; -1 = checkpoint config.',
    )
    p.add_argument(
        '--idm_action_chunk_horizon',
        type=int,
        default=5,
        help='Env-eval IDM: env steps per replan (passed as critic_config action_chunk_horizon only for _evaluate_env_tasks).',
    )
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    p.add_argument(
        '--subgoal_override_goal',
        action='store_true',
        help='Ablation: ignore predicted subgoals and use the final goal for both IDM and actor.',
    )
    p.add_argument(
        '--skip_if_saved',
        action='store_true',
        help='Skip eval when run_dir/eval_results/epoch<E>_n<N>.json already exists.',
    )
    p.add_argument(
        '--idm_only',
        action='store_true',
        help='Evaluate only the IDM policy (flow subgoal + inverse dynamics); skip loading and '
        'rolling out the SPI actor. Use before the 50K train_actor_spi.py finetuning stage.',
    )
    args = p.parse_args()

    if str(args.mujoco_gl).strip():
        from rollout.env import configure_mujoco_gl

        configure_mujoco_gl(str(args.mujoco_gl))

    run_dir = Path(args.run_dir).resolve()
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(flags_path)
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    fg = root['flags']
    seed = int(fg['seed']) if int(args.seed) < 0 else int(args.seed)

    dynamics_config, critic_config, actor_config = _build_configs(root, fg)
    if int(args.subgoal_eval_num_samples) > 0:
        dynamics_config['subgoal_eval_num_samples'] = int(args.subgoal_eval_num_samples)
    eval_temperature: float | None = None
    if float(args.subgoal_temperature) >= 0.0:
        eval_temperature = float(args.subgoal_temperature)
        dynamics_config['subgoal_temperature'] = eval_temperature
    env_name = fg['env_name']
    dataset_dir = fg.get('dataset_dir', '')
    env, train_plain, _ = make_env_and_datasets(
        env_name,
        frame_stack=critic_config['frame_stack'],
        dataset_dir=dataset_dir,
    )
    obs_dim_env = int(np.prod(env.observation_space.shape))
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    if not phi_idxs:
        phi_idxs = infer_phi_goal_obs_indices(str(env_name), obs_dim_env)
        critic_config['phi_goal_obs_indices'] = phi_idxs
        dynamics_config['phi_goal_obs_indices'] = phi_idxs
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim

    dynamics_dataset = PathHGCDataset(Dataset.create(**train_plain), dynamics_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_config)
    common = _intersect_valid_starts(dynamics_dataset, critic_dataset)
    bs = int(dynamics_config['batch_size'])
    ex_idxs = _sample_shared_idxs(common, bs)
    ex_dynamics = dynamics_dataset.sample(len(ex_idxs), idxs=ex_idxs)
    ex_critic = critic_dataset.sample(len(ex_idxs), idxs=ex_idxs)

    ex = jnp.asarray(ex_dynamics['observations'], dtype=jnp.float32)
    ex_act = jnp.asarray(ex_dynamics['actions'], dtype=jnp.float32)
    dynamics_agent = DynamicsAgent.create(seed, ex, dynamics_config, ex_actions=ex_act)
    critic_agent = _create_critic_agent(seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(seed, ex_dynamics, actor_config)

    dynamics_dir = resolve_dynamics_checkpoint_dir(run_dir)
    ep = pick_epoch(int(args.epoch), list_checkpoint_suffixes(dynamics_dir))
    dynamics_pkl = dynamics_dir / f'params_{ep}.pkl'
    critic_pkl = resolve_critic_checkpoint_dir(run_dir) / f'params_{ep}.pkl'
    dynamics_agent = load_checkpoint_pkl(dynamics_agent, dynamics_pkl)
    critic_agent = load_checkpoint_pkl(critic_agent, critic_pkl)
    # Auto-enable IDM-only when the actor checkpoint is absent (e.g. runs trained with
    # --train_actor_spi=false), otherwise load and evaluate the SPI actor as usual.
    idm_only = bool(args.idm_only)
    actor_dir = resolve_actor_checkpoint_dir(run_dir, required=False)
    actor_pkl = (actor_dir / f'params_{ep}.pkl') if actor_dir is not None else None
    if not idm_only:
        if actor_pkl is None or not actor_pkl.is_file():
            print(
                f'[eval] actor checkpoint not found (looked for {actor_pkl}); '
                f'falling back to --idm_only.'
            )
            idm_only = True
        else:
            actor_agent = load_checkpoint_pkl(actor_agent, actor_pkl)

    task_ids = parse_int_list(args.eval_task_ids) if str(args.eval_task_ids).strip() else parse_int_list(
        str(fg.get('eval_task_ids', '1,2,3,4,5'))
    )
    ep_task = int(fg['eval_episodes_per_task']) if int(args.eval_episodes_per_task) < 0 else int(args.eval_episodes_per_task)

    critic_eval = copy.deepcopy(critic_config)
    idm_h = int(args.idm_action_chunk_horizon)
    if idm_h < 1:
        p.error('--idm_action_chunk_horizon must be >= 1')
    critic_eval['action_chunk_horizon'] = idm_h

    eval_n = int(dynamics_config.get('subgoal_eval_num_samples', 1))
    saved_path = eval_result_path(
        run_dir,
        epoch=int(args.epoch),
        eval_n=eval_n,
        subgoal_temperature=eval_temperature,
    )
    if bool(args.skip_if_saved) and saved_path.is_file():
        with open(saved_path, encoding='utf-8') as f:
            record = json.load(f)
        print(f'Skip eval (already saved): {saved_path}')
        print(f"eval_idm/success_rate_mean={record.get('idm_success_rate_mean', float('nan')):.4f}")
        print(f"eval/success_rate_mean={record.get('actor_success_rate_mean', float('nan')):.4f}")
        for prefix, value in record.get('four_way_success_rate_means', {}).items():
            print(f'{prefix}/success_rate_mean={float(value):.4f}')
        return

    print(f'Loaded epoch={ep} from {run_dir}')
    print(
        f'eval task_ids={task_ids} episodes_per_task={ep_task} budget=env_max_episode_steps '
        f'idm_action_chunk_horizon={idm_h} '
        f'subgoal_eval_num_samples={dynamics_config.get("subgoal_eval_num_samples", "")} '
        f'subgoal_temperature={dynamics_config.get("subgoal_temperature", "")} '
        f'subgoal_override_goal={bool(args.subgoal_override_goal)} '
        f'(training critic had {int(critic_config["action_chunk_horizon"])})'
    )

    metrics = _evaluate_env_tasks(
        env,
        dynamics_agent,
        actor_agent,
        actor_config,
        critic_eval,
        critic_agent=critic_agent,
        task_ids=task_ids,
        episodes_per_task=ep_task,
        video_episodes_per_task=0,
        video_frame_skip=4,
        video_fps=15,
        subgoal_override_goal=bool(args.subgoal_override_goal),
        idm_only=idm_only,
    )
    print('--- IDM --- (success = any step info["success"])')
    print(f"eval_idm/success_rate_mean={metrics.get('eval_idm/success_rate_mean', float('nan')):.4f}")
    for tid in task_ids:
        k = f'eval_idm/task_{tid}/success_rate'
        if k in metrics:
            print(f'  {k}={metrics[k]:.4f}')
    if not idm_only:
        print('--- Actor ---')
        print(f"eval/success_rate_mean={metrics.get('eval/success_rate_mean', float('nan')):.4f}")
        for tid in task_ids:
            k = f'eval/task_{tid}/success_rate'
            if k in metrics:
                print(f'  {k}={metrics[k]:.4f}')
    for prefix, label in (
        ('eval_flow_idm', 'Flow+IDM'),
        ('eval_flow_actor', 'Flow+Actor'),
    ):
        mean_key = f'{prefix}/success_rate_mean'
        if mean_key not in metrics:
            continue
        print(f'--- {label} ---')
        print(f'{mean_key}={metrics.get(mean_key, float("nan")):.4f}')
        for tid in task_ids:
            k = f'{prefix}/task_{tid}/success_rate'
            if k in metrics:
                print(f'  {k}={metrics[k]:.4f}')

    out_path = save_eval_results(
        run_dir,
        epoch=ep,
        subgoal_eval_num_samples=eval_n,
        task_ids=task_ids,
        episodes_per_task=ep_task,
        metrics=metrics,
        fg=fg,
        root=root,
        subgoal_temperature=eval_temperature,
    )
    print(f'Saved eval results: {out_path}')


if __name__ == '__main__':
    main()
