#!/usr/bin/env python3
"""Real-env rollout: dynamics ``predict_subgoal`` + chunk actor (same path as ``main._evaluate_env_tasks``).

Loads ``checkpoints/dynamics/params_<epoch>.pkl`` and ``checkpoints/actor/params_<epoch>.pkl`` from a training run.

Episode length is taken from the env's own ``TimeLimit`` wrapper (no user-defined chunk budget):
the script runs ``ceil(max_episode_steps / actor_horizon) + 1`` replans, which is enough to walk the
env to either ``info['success']`` or ``truncated``.

Example::

    MUJOCO_GL=egl python rollout/actor.py \\
        --run_dir=runs/... \\
        --checkpoint_epoch=1000 \\
        --task_id=1 \\
        --out_mp4=runs/.../actor_rollout.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from agents.actor import ActorAgent, get_actor_config
from agents.dynamics import DynamicsAgent
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets
from rollout.env import (
    configure_mujoco_gl,
    env_render_rgb_u8,
    format_maze_navigator_log,
    is_manipspace_env,
    load_maze_navigator_snap,
    max_episode_steps_from_wrappers,
    sync_env_state_from_compact_manip_obs,
    sync_env_state_from_obs_vector_aligned,
)
from rollout.episode_runner import make_actor_chunk_fn, run_chunked_episode
from rollout.plot import (
    axis_limits,
    compose_state_subgoal_env_frames,
    maze_navigator_for_xy_plot,
    overlay_rgb_frames_obs2d_panel,
    write_rgb_array_mp4,
)
from utils.run_io import (
    get_trajectory,
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    load_run_flags,
    pick_epoch,
    resolve_actor_checkpoint_dir,
    resolve_dynamics_checkpoint_dir,
)


def _load_actor_config_from_flags(flags_path: Path) -> dict:
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    act = root.get('actor')
    if not isinstance(act, dict):
        raise KeyError(f'{flags_path} must contain an "actor" object.')
    base = get_actor_config()
    for k, v in act.items():
        base[k] = v
    d = dict(base)
    return d


def rollout_dynamics_actor_env(
    env,
    dynamics_agent: DynamicsAgent,
    actor_agent: ActorAgent,
    s0: np.ndarray,
    s_g: np.ndarray,
    max_chunks: int,
    *,
    low: np.ndarray,
    high: np.ndarray,
    actor_horizon: int,
    env_action_dim: int,
    record_env_rgb: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray | None, np.ndarray]:
    """Chunked dynamics-subgoal + actor rollout via the shared runner.

    Success is decided **only** by the env (``info['success']``); no user-defined
    tolerance is applied here.
    """
    if is_manipspace_env(env):
        # ManipSpace uses compact observations, not qpos||qvel. For task-mode rollouts
        # the caller has already reset the env to s0; for offline traj rollouts we can
        # reconstruct a faithful-enough visual/physics state from the compact obs.
        try:
            cur = sync_env_state_from_compact_manip_obs(env, s0)
        except Exception:
            cur = np.asarray(s0, dtype=np.float32).reshape(-1)
    else:
        cur = sync_env_state_from_obs_vector_aligned(env, s0, s_g)
    goal = np.asarray(s_g, dtype=np.float32).reshape(-1)

    base_chunk_fn = make_actor_chunk_fn(
        dynamics_agent,
        actor_agent,
        int(actor_horizon),
        int(env_action_dim),
    )
    hats_per_step: list[np.ndarray] = []

    def _chunk(obs: np.ndarray, g: np.ndarray) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        pred = np.asarray(
            jax.device_get(
                dynamics_agent.infer_subgoal(jnp.asarray(obs, dtype=jnp.float32), jnp.asarray(g, dtype=jnp.float32))
            ),
            dtype=np.float32,
        ).reshape(-1)
        chunk = base_chunk_fn(obs, g)
        for _ in range(int(chunk.shape[0])):
            hats_per_step.append(pred.copy())
        return chunk

    outcome = run_chunked_episode(
        env,
        cur,
        goal,
        low=np.asarray(low, dtype=np.float32).reshape(-1),
        high=np.asarray(high, dtype=np.float32).reshape(-1),
        max_chunks=int(max_chunks),
        sample_action_chunk=_chunk,
        record_rgb=bool(record_env_rgb),
    )
    hats = (
        np.stack(hats_per_step[: outcome.states.shape[0] - 1], axis=0)
        if hats_per_step and outcome.states.shape[0] > 1
        else np.zeros((0, cur.shape[-1]), dtype=np.float32)
    )
    hats_steps_full = (
        np.stack(hats_per_step, axis=0)
        if hats_per_step
        else np.zeros((0, cur.shape[-1]), dtype=np.float32)
    )
    return outcome.states, hats, outcome.n_chunks, outcome.ok_env, outcome.rgb_frames, hats_steps_full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=1000,
        help='Suffix for both dynamics/params_<n>.pkl and actor/params_<n>.pkl (nearest match if missing).',
    )
    p.add_argument('--dynamics_epoch', type=int, default=-1, help='Override dynamics checkpoint suffix (-1 = same as --checkpoint_epoch).')
    p.add_argument('--actor_epoch', type=int, default=-1, help='Override actor checkpoint suffix (-1 = same as --checkpoint_epoch).')
    p.add_argument('--traj_idx', type=int, default=0, help='Offline episode index when --task_id=0.')
    p.add_argument('--task_id', type=int, default=0, help='OGBench task id [1..num_tasks]; 0 uses offline traj start/goal.')
    p.add_argument('--plot_dim0', type=int, default=0)
    p.add_argument('--plot_dim1', type=int, default=1)
    p.add_argument('--navigator_clamp', type=str, choices=('ij', 'oracle', 'union', 'center'), default='ij')
    p.add_argument('--navigator_edge_inset', type=float, default=0.08)
    p.add_argument('--maze_type', type=str, default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_path', type=str, default='rollout_actor.png')
    p.add_argument('--out_mp4', type=str, default='')
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument('--no_mp4', action='store_true')
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    p.add_argument(
        '--render_subgoal_env',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Replace the right-side 2D panel with a second env render of the predicted subgoal '
            '(OGBench manipspace envs only). Default: enabled when the env is manipspace.'
        ),
    )
    p.add_argument(
        '--value_heatmap',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Overlay scalar value V(s, goal) on the right XY panel (checkpoints/critic/).',
    )
    p.add_argument('--value_grid_n', type=int, default=56)
    p.add_argument('--critic_epoch', type=int, default=-1, help='Critic checkpoint suffix; -1 = dynamics epoch used.')
    args = p.parse_args()

    try:
        configure_mujoco_gl(str(args.mujoco_gl))
    except ValueError as e:
        p.error(str(e))

    run_dir = Path(args.run_dir).resolve()
    flags_path = run_dir / 'flags.json'
    dynamics_ckpt_dir = resolve_dynamics_checkpoint_dir(run_dir)
    actor_ckpt_dir = resolve_actor_checkpoint_dir(run_dir, required=True)

    base_ep = int(args.checkpoint_epoch)
    dynamics_ep = int(args.dynamics_epoch) if int(args.dynamics_epoch) >= 0 else base_ep
    actor_ep = int(args.actor_epoch) if int(args.actor_epoch) >= 0 else base_ep
    dynamics_ep = pick_epoch(dynamics_ep, list_checkpoint_suffixes(dynamics_ckpt_dir))
    actor_ep = pick_epoch(actor_ep, list_checkpoint_suffixes(actor_ckpt_dir))

    dynamics_cfg, env_name = load_run_flags(run_dir)
    actor_cfg = _load_actor_config_from_flags(flags_path)

    try:
        navigator = load_maze_navigator_snap(args.maze_type, env_name)
        print(format_maze_navigator_log(navigator, str(args.navigator_clamp), float(args.navigator_edge_inset)))
    except ValueError as nav_err:
        if str(args.maze_type).strip():
            raise
        navigator = None
        print(f'Maze navigator disabled for env_name={env_name!r} ({nav_err.__cause__ or nav_err})')

    env, train_raw, _ = make_env_and_datasets(
        env_name, frame_stack=dynamics_cfg.get('frame_stack'), render_mode='rgb_array',
    )
    tid = int(args.task_id)
    if tid != 0:
        u = env.unwrapped
        n_tasks = int(getattr(u, 'num_tasks', 5))
        if not (1 <= tid <= n_tasks):
            p.error(f'--task_id must be in [1, {n_tasks}] for {env_name!r} (got {tid})')
        ob, info = env.reset(options=dict(task_id=tid, render_goal=False))
        if 'goal' not in info:
            raise RuntimeError(f'{env_name!r} reset did not set info["goal"].')
        s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
        s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
        traj = np.stack([s0, s_g], axis=0)
        print(f'OGBench eval reset: task_id={tid}  obs_dim={s0.shape[-1]}')
    else:
        dataset = Dataset.create(**train_raw)
        traj = get_trajectory(dataset, int(args.traj_idx))
        s0 = np.asarray(traj[0], dtype=np.float32).reshape(-1)
        s_g = np.asarray(traj[-1], dtype=np.float32).reshape(-1)

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    saved_actor_dim = int(actor_cfg.get('action_dim', act_dim))
    actor_cfg['action_dim'] = act_dim
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    dynamics_agent = DynamicsAgent.create(int(args.seed), ex, dynamics_cfg, ex_actions=ex_act)
    dynamics_pkl = dynamics_ckpt_dir / f'params_{dynamics_ep}.pkl'
    dynamics_agent = load_checkpoint_pkl(dynamics_agent, dynamics_pkl)
    print(f'Loaded dynamics {dynamics_pkl}')

    ex_goal = jnp.asarray(s_g.reshape(1, -1), dtype=jnp.float32)
    actor_agent = ActorAgent.create(int(args.seed), ex, actor_cfg, ex_goals=ex_goal)
    actor_pkl = actor_ckpt_dir / f'params_{actor_ep}.pkl'
    actor_agent = load_checkpoint_pkl(actor_agent, actor_pkl)
    print(f'Loaded actor {actor_pkl}')

    from rollout.value_field import value_mesh_for_xy, load_critic_for_run

    ce = int(args.critic_epoch) if int(args.critic_epoch) >= 0 else int(dynamics_ep)
    critic_agent = load_critic_for_run(
        run_dir,
        ce,
        env,
        train_raw,
        seed=int(args.seed),
    )
    print(f'Loaded critic for actor inference/value heatmap (epoch suffix {ce})')

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    env_action_dim = int(low.shape[-1])
    actor_horizon = int(actor_cfg['actor_chunk_horizon'])
    if saved_actor_dim != env_action_dim:
        print(
            f'Corrected flags actor action_dim={saved_actor_dim} to env action dim={env_action_dim} '
            'before loading the actor checkpoint.'
        )
    env_max_steps = max_episode_steps_from_wrappers(env)
    if env_max_steps is None:
        raise RuntimeError(
            f'{env_name!r} has no TimeLimit wrapper; cannot derive replan budget from the env. '
            'Add a TimeLimit wrapper or hard-code the budget here.'
        )
    max_chunks = (int(env_max_steps) + int(actor_horizon) - 1) // int(actor_horizon) + 1
    print(
        f'Actor conditioning at inference: predicted dynamics subgoal (spi_goals). '
        f'env_max_steps={int(env_max_steps)} actor_horizon={actor_horizon} → max_chunks={int(max_chunks)}'
    )
    roll, hats, n_chunks, reached, env_frames, hats_per_step = rollout_dynamics_actor_env(
        env,
        dynamics_agent,
        actor_agent,
        s0,
        s_g,
        int(max_chunks),
        low=low,
        high=high,
        actor_horizon=actor_horizon,
        env_action_dim=env_action_dim,
        record_env_rgb=True,
    )
    n_trans = max(0, int(roll.shape[0]) - 1)
    print(
        f'Actor rollout: {n_chunks} replan rounds, actor_horizon={actor_horizon} → '
        f'{roll.shape[0]} obs ({n_trans} transitions), env_info_success={reached}'
    )

    d0, d1 = int(args.plot_dim0), int(args.plot_dim1)
    plot_nav = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)

    heat_mesh = None
    heat_vmin = heat_vmax = None
    if bool(args.value_heatmap):
        xlim, ylim = axis_limits(traj, roll, hats, d0, d1, s_g, s0, navigator=plot_nav, seg=None)
        tpl = np.asarray(roll[0], dtype=np.float32).reshape(-1)
        XX, YY, ZZ, heat_vmin, heat_vmax = value_mesh_for_xy(
            critic_agent,
            tpl,
            np.asarray(s_g, dtype=np.float32).reshape(-1),
            int(d0),
            int(d1),
            xlim,
            ylim,
            grid_n=int(args.value_grid_n),
        )
        heat_mesh = (XX, YY, ZZ)

    use_subgoal_env_panel = (
        bool(is_manipspace_env(env)) if args.render_subgoal_env is None else bool(args.render_subgoal_env)
    )
    if use_subgoal_env_panel and not is_manipspace_env(env):
        print('Warning: --render_subgoal_env requested but env is not OGBench manipspace; falling back to 2D panel.')
        use_subgoal_env_panel = False

    if (not args.no_mp4) and env_frames is not None and env_frames.size > 0:
        mp4_out = Path(args.out_mp4.strip()) if str(args.out_mp4).strip() else Path(args.out_path).with_suffix('.mp4')
        mp4_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if use_subgoal_env_panel:
                if int(env_frames.shape[0]) <= 1 or int(hats_per_step.shape[0]) == 0:
                    raise RuntimeError('No subgoal-aligned frames to render.')
                step_env_frames = env_frames[1:]
                T_steps = min(int(step_env_frames.shape[0]), int(hats_per_step.shape[0]))
                step_env_frames = step_env_frames[:T_steps]

                sub_env, _, _ = make_env_and_datasets(
                    env_name, frame_stack=dynamics_cfg.get('frame_stack'), render_mode='rgb_array',
                )
                if tid != 0:
                    sub_env.reset(options=dict(task_id=tid, render_goal=False))
                else:
                    sub_env.reset()
                subgoal_frames: list[np.ndarray] = []
                for t in range(T_steps):
                    sync_env_state_from_compact_manip_obs(sub_env, hats_per_step[t])
                    fr = env_render_rgb_u8(sub_env)
                    if fr is None:
                        raise RuntimeError(f'Failed to render subgoal frame at step {t}.')
                    subgoal_frames.append(fr)
                sub_frames_np = np.stack(subgoal_frames, axis=0)
                try:
                    sub_env.close()
                except Exception:
                    pass

                frames = compose_state_subgoal_env_frames(step_env_frames, sub_frames_np, output_scale=1.1)
                caption = ['left: actor env state', 'right: env @ predicted subgoal']
                write_rgb_array_mp4(frames, mp4_out, float(args.fps), caption_lines=caption)
            else:
                frames = overlay_rgb_frames_obs2d_panel(
                    env_frames,
                    traj,
                    roll,
                    hats,
                    None,
                    s0,
                    s_g,
                    d0,
                    d1,
                    plot_nav,
                    env_name=env_name,
                    chunk_hat_stride=actor_horizon,
                    value_heatmap=heat_mesh,
                    value_heatmap_vmin=heat_vmin,
                    value_heatmap_vmax=heat_vmax,
                )
                write_rgb_array_mp4(frames, mp4_out, float(args.fps))
            print(f'Wrote MP4 {mp4_out.resolve()}')
        except Exception as e:
            print(f'Warning: MP4 failed ({e!r}). pip install imageio imageio-ffmpeg pillow')
    elif not args.no_mp4:
        print('Warning: no RGB frames; check render_mode')


if __name__ == '__main__':
    main()

