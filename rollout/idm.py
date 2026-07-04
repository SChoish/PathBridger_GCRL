#!/usr/bin/env python3
"""Real-environment rollout: dynamics planner + inverse dynamics (IDM).

This script is **separate** from ``rollout/subgoal.py`` (pure state-space trajectory plots).
Each replan: ``predict_subgoal`` → ``plan`` or stochastic ``sample_plan`` (see
``--planner_noise_scale``) → IDM actions for up to ``--action_chunk_horizon`` env steps
(capped at ``dynamics_N``), then repeat.

Headless RGB: if ``DISPLAY`` is unset, ``MUJOCO_GL`` defaults to ``egl`` (see ``--mujoco_gl``).

Example::

    cd <douri repo root>
    python rollout/idm.py \\
        --run_dir=runs/... \\
        --checkpoint_epoch=1000 \\
        --max_steps=1000 \\
        --out_path=idm_rollout.png

``--idm_checkpoint`` is optional: when omitted, weights are read from ``idm_net`` inside the dynamics
``params_<epoch>.pkl``. Use a standalone IDM pickle only for older runs without ``idm_net``.

OGBench-style evaluation goals: pass ``--task_id`` in ``[1, num_tasks]`` (typically 5). The script
calls ``env.reset(options=dict(task_id=..., render_goal=False))`` and uses the returned observation
as ``s0`` and ``info['goal']`` as ``s_g``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.dynamics import DynamicsAgent
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    load_run_flags,
    pick_epoch,
    resolve_dynamics_checkpoint_dir,
)
from rollout.env import (
    configure_mujoco_gl,
    env_render_rgb_u8,
    format_maze_navigator_log,
    is_manipspace_env,
    load_maze_navigator_snap,
    make_xy_clamper,
    sync_env_state_from_compact_manip_obs,
    sync_env_state_from_obs_vector_aligned,
)
from rollout.episode_runner import run_chunked_episode
from rollout.plot import (
    axis_limits,
    compose_state_subgoal_env_frames,
    maze_navigator_for_xy_plot,
    overlay_rgb_frames_obs2d_panel,
    write_rgb_array_mp4,
)
from utils.inverse_dynamics import InverseDynamicsMLP
from rollout.maze_navigator import MazeNavigatorMap


def rollout_dynamics_idm_env(
    env,
    agent: DynamicsAgent,
    idm_model: InverseDynamicsMLP,
    idm_params,
    s0: np.ndarray,
    s_g: np.ndarray,
    max_chunks: int,
    navigator: MazeNavigatorMap | None = None,
    clamp_dim0: int = 0,
    clamp_dim1: int = 1,
    navigator_clamp_mode: str = 'ij',
    navigator_edge_inset: float = 0.08,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_chunk_horizon: int = 5,
    record_env_rgb: bool = True,
    planner_noise_scale: float = 0.0,
    planner_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray | None, np.ndarray, np.ndarray]:
    """Chunked dynamics bridge + inverse dynamics in the real environment.

    Success is decided **only** by the env (``info['success']`` in the shared
    runner); no user-defined tolerance is applied here. ``navigator`` /
    ``xy_clamper`` are visualization-only and do not affect success.
    """
    g_np = np.asarray(s_g, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    xy_clamper = make_xy_clamper(
        g_np, navigator, clamp_dim0, clamp_dim1, navigator_clamp_mode, navigator_edge_inset
    )

    if is_manipspace_env(env):
        # OGBench manipspace envs use compact (28-dim) observations that are NOT qpos||qvel,
        # so we cannot replay physics from ``s0`` here. The caller already set the env to ``s0``
        # via ``env.reset(options=dict(task_id=...))`` and there is no ``set_goal`` analogue, so
        # we trust the reset state and only normalize ``s0``'s dtype.
        cur = np.asarray(s0, dtype=np.float32).reshape(-1)
    else:
        cur = sync_env_state_from_obs_vector_aligned(env, s0, s_g)
    d = int(cur.shape[-1])

    @jax.jit
    def _idm_actions(p, o_stack: jnp.ndarray, on_stack: jnp.ndarray) -> jnp.ndarray:
        return idm_model.apply({'params': p}, o_stack, on_stack)

    plan_rng = jax.random.PRNGKey(int(planner_seed))
    use_stoch_plan = float(planner_noise_scale) > 0.0

    hats_per_step: list[np.ndarray] = []
    plan_trajs_per_step: list[np.ndarray] = [np.zeros((0, d), dtype=np.float32)]

    def _idm_chunk(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        nonlocal plan_rng
        s = jnp.asarray(obs, dtype=jnp.float32)
        g = jnp.asarray(goal, dtype=jnp.float32)
        hat = agent.infer_subgoal(s, g)
        hat_np = np.asarray(jax.device_get(hat), dtype=np.float32).reshape(-1)
        hat_plot_np = xy_clamper(hat_np.copy())

        if use_stoch_plan:
            plan_rng, sk = jax.random.split(plan_rng)
            out = agent.sample_plan(s, hat, sk, noise_scale=float(planner_noise_scale))
        else:
            out = agent.plan(s, hat)
        chunk_traj_raw = np.asarray(jax.device_get(out['trajectory']), dtype=np.float32)
        if chunk_traj_raw.shape[0] < 2:
            return np.zeros((0, low.shape[-1]), dtype=np.float32)
        chunk_traj_plot = np.stack(
            [xy_clamper(chunk_traj_raw[i].copy()) for i in range(chunk_traj_raw.shape[0])]
        )
        plan_seg = np.asarray(chunk_traj_plot[1:], dtype=np.float32)

        o_prev = jnp.asarray(chunk_traj_raw[:-1], dtype=jnp.float32)
        o_next = jnp.asarray(chunk_traj_raw[1:], dtype=jnp.float32)
        actions = np.asarray(
            jax.device_get(
                agent._idm_actions_from_trajectories(
                    jnp.asarray(chunk_traj_raw[None, ...], dtype=jnp.float32),
                    int(action_chunk_horizon),
                )
            ),
            dtype=np.float32,
        )[0]

        n_exec = min(int(actions.shape[0]), max(1, int(action_chunk_horizon)))
        for _ in range(n_exec):
            hats_per_step.append(hat_plot_np.copy())
            plan_trajs_per_step.append(plan_seg.copy())
        return actions[:n_exec]

    outcome = run_chunked_episode(
        env,
        cur,
        g_np,
        low=low,
        high=high,
        max_chunks=int(max_chunks),
        sample_action_chunk=_idm_chunk,
        record_rgb=bool(record_env_rgb),
    )

    hats = (
        np.stack(hats_per_step[: outcome.states.shape[0] - 1], axis=0)
        if hats_per_step and outcome.states.shape[0] > 1
        else np.zeros((0, d), dtype=np.float32)
    )
    if outcome.states.shape[0] > 0:
        pkr_slice = plan_trajs_per_step[: outcome.states.shape[0]]
        try:
            pkr = np.stack(pkr_slice, axis=0)
        except ValueError:
            # First entry is a zero-length stub (no plan available before the first replan); the
            # rest are constant-length ``plan_seg``. Stacking heterogeneous shapes fails: pad the
            # stub by repeating the first real plan, so callers always get a regular array.
            non_stub = next((p for p in pkr_slice if p.shape[0] > 0), None)
            if non_stub is None:
                pkr = np.zeros((len(pkr_slice), 0, d), dtype=np.float32)
            else:
                fixed = [non_stub if p.shape[0] == 0 else p for p in pkr_slice]
                pkr = np.stack(fixed, axis=0)
    else:
        pkr = np.zeros((0, 0, d), dtype=np.float32)
    # Per-step subgoal series (length = total executed env steps). Distinct from ``hats``,
    # which is truncated to one entry per replan-chunk for the existing 2D panel overlay.
    hats_steps_full = (
        np.stack(hats_per_step, axis=0)
        if hats_per_step
        else np.zeros((0, d), dtype=np.float32)
    )
    return outcome.states, hats, outcome.n_chunks, outcome.ok_env, outcome.rgb_frames, pkr, hats_steps_full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=1000,
        help='Dynamics params_<n>.pkl suffix (default 1000). If missing, nearest available is used.',
    )
    p.add_argument(
        '--task_id',
        type=int,
        required=True,
        help='OGBench eval task id in [1, num_tasks] via env.reset(options=dict(task_id=...)).',
    )
    p.add_argument('--max_steps', type=int, default=1000, help='Max replan chunks (default 1000).')
    p.add_argument(
        '--idm_checkpoint',
        type=str,
        default='',
        help='Standalone inverse dynamics params_*.pkl. Empty → use idm_net inside dynamics checkpoint.',
    )
    p.add_argument('--action_chunk_horizon', type=int, default=5)
    p.add_argument(
        '--planner_noise_scale',
        type=float,
        default=0.0,
        help='If > 0, replan uses stochastic sample_plan(s, hat, rng, noise_scale=...) instead of plan().',
    )
    p.add_argument(
        '--planner_seed',
        type=int,
        default=-1,
        help='JAX PRNG seed for sample_plan splits; -1 means use --seed.',
    )
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    p.add_argument('--plot_dim0', type=int, default=0)
    p.add_argument('--plot_dim1', type=int, default=1)
    p.add_argument('--navigator_clamp', type=str, choices=('ij', 'oracle', 'union', 'center'), default='ij')
    p.add_argument('--navigator_edge_inset', type=float, default=0.08)
    p.add_argument('--maze_type', type=str, default='')
    p.add_argument('--clamp_dim0', type=int, default=-1)
    p.add_argument('--clamp_dim1', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_path', type=str, default='rollout_idm.png')
    p.add_argument('--out_mp4', type=str, default='')
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument('--no_mp4', action='store_true')
    p.add_argument(
        '--value_heatmap',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Overlay scalar value V(s, goal) on the right XY panel (requires checkpoints/critic/).',
    )
    p.add_argument(
        '--render_subgoal_env',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Replace the right-side 2D panel with a second env render of the predicted subgoal '
            '(OGBench manipspace envs only). Default: enabled when the env is manipspace.'
        ),
    )
    p.add_argument('--value_grid_n', type=int, default=56, help='Square grid resolution for value heatmap.')
    p.add_argument(
        '--critic_epoch',
        type=int,
        default=-1,
        help='Critic params_<n>.pkl suffix; -1 = same suffix as --checkpoint_epoch.',
    )
    args = p.parse_args()

    try:
        configure_mujoco_gl(str(args.mujoco_gl))
    except ValueError as e:
        p.error(str(e))

    if int(args.action_chunk_horizon) < 1:
        p.error('--action_chunk_horizon must be >= 1')
    planner_seed = int(args.seed) if int(args.planner_seed) < 0 else int(args.planner_seed)

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = resolve_dynamics_checkpoint_dir(run_dir)
    ckpt_epoch = pick_epoch(int(args.checkpoint_epoch), list_checkpoint_suffixes(ckpt_dir))

    cfg, env_name = load_run_flags(run_dir)
    try:
        navigator = load_maze_navigator_snap(args.maze_type, env_name)
        print(format_maze_navigator_log(navigator, str(args.navigator_clamp), float(args.navigator_edge_inset)))
    except ValueError as nav_err:
        # Non-maze envs (e.g. OGBench manipspace) have no maze metadata; navigator stays disabled.
        if str(args.maze_type).strip():
            raise
        navigator = None
        print(f'Maze navigator disabled for env_name={env_name!r} ({nav_err.__cause__ or nav_err})')

    env, train_raw, _ = make_env_and_datasets(
        env_name, frame_stack=cfg.get('frame_stack'), render_mode='rgb_array',
    )
    tid = int(args.task_id)
    u = env.unwrapped
    n_tasks = int(getattr(u, 'num_tasks', 5))
    if not (1 <= tid <= n_tasks):
        p.error(f'--task_id must be in [1, {n_tasks}] for {env_name!r} (got {tid})')
    ob, info = env.reset(options=dict(task_id=tid, render_goal=False))
    if 'goal' not in info:
        raise RuntimeError(
            f'{env_name!r} reset(task_id=...) did not set info["goal"]; cannot run IDM rollout in task mode.'
        )
    s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
    s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
    traj = np.stack([s0, s_g], axis=0)
    print(f'OGBench eval reset: task_id={tid}  obs_dim={s0.shape[-1]}  goal_dim={s_g.shape[-1]}')

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = DynamicsAgent.create(args.seed, ex, cfg, ex_actions=ex_act)
    pkl_path = ckpt_dir / f'params_{ckpt_epoch}.pkl'
    agent = load_checkpoint_pkl(agent, pkl_path)
    dynamics_N = int(agent.config['dynamics_N'])
    print(
        f'Loaded dynamics {pkl_path}  dynamics_N={dynamics_N}  '
        f'planner_noise_scale={float(args.planner_noise_scale)}  planner_seed={planner_seed}'
    )

    idm_ck_stripped = str(args.idm_checkpoint).strip()
    if not idm_ck_stripped:
        ptree = agent.network.params
        # Flax ``nn.scan`` / combined modules may prefix keys with ``modules_``.
        if 'idm_net' in ptree:
            idm_params = ptree['idm_net']
        elif 'modules_idm_net' in ptree:
            idm_params = ptree['modules_idm_net']
        else:
            raise FileNotFoundError(
                'Dynamics checkpoint has no idm_net / modules_idm_net '
                '(train with embedded IDM or pass --idm_checkpoint).'
            )
        idm_model = InverseDynamicsMLP(
            obs_dim=int(s0.shape[-1]),
            action_dim=int(agent.config['idm_action_dim']),
            hidden_dims=tuple(int(x) for x in agent.config['idm_hidden_dims']),
        )
        print('Using idm_net from dynamics checkpoint')
    else:
        idm_ckpt_path = Path(idm_ck_stripped).resolve()
        if not idm_ckpt_path.is_file():
            raise FileNotFoundError(f'IDM checkpoint not found: {idm_ckpt_path}')
        with open(idm_ckpt_path, 'rb') as f:
            idm_ck = pickle.load(f)
        idm_model = InverseDynamicsMLP(
            obs_dim=int(idm_ck['obs_dim']),
            action_dim=int(idm_ck['action_dim']),
            hidden_dims=tuple(idm_ck['hidden_dims']),
        )
        idm_params = idm_ck['params']
        if int(idm_ck['obs_dim']) != int(s0.shape[-1]):
            raise ValueError(f'IDM obs_dim mismatch vs trajectory obs dim={int(s0.shape[-1])}')
        print(f'Loaded standalone IDM {idm_ckpt_path} (epoch={idm_ck.get("epoch")})')
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    c0 = args.clamp_dim0 if args.clamp_dim0 >= 0 else args.plot_dim0
    c1 = args.clamp_dim1 if args.clamp_dim1 >= 0 else args.plot_dim1
    nav_kw = dict(
        navigator=navigator,
        clamp_dim0=c0,
        clamp_dim1=c1,
        navigator_clamp_mode=str(args.navigator_clamp),
        navigator_edge_inset=float(args.navigator_edge_inset),
    )

    from rollout.value_field import value_mesh_for_xy, load_critic_for_run

    ce = int(args.critic_epoch) if int(args.critic_epoch) >= 0 else int(ckpt_epoch)
    critic_agent = load_critic_for_run(
        run_dir,
        ce,
        env,
        train_raw,
        seed=int(args.seed),
    )
    print(f'Loaded critic for IDM inference/value heatmap (epoch suffix {ce})')

    roll, hats, n_chunks, reached, env_frames, frame_plan_trajs, hats_per_step = rollout_dynamics_idm_env(
        env,
        agent,
        idm_model,
        idm_params,
        s0,
        s_g,
        int(args.max_steps),
        **nav_kw,
        action_low=low,
        action_high=high,
        action_chunk_horizon=int(args.action_chunk_horizon),
        planner_noise_scale=float(args.planner_noise_scale),
        planner_seed=planner_seed,
    )
    n_trans = max(0, int(roll.shape[0]) - 1)
    action_chunk_horizon = int(args.action_chunk_horizon)
    steps_per_replan = min(action_chunk_horizon, dynamics_N)
    print(
        f'IDM rollout: {n_chunks} replans × up to {steps_per_replan} env steps/replan → '
        f'{roll.shape[0]} obs ({n_trans} transitions), env_info_success={reached}'
    )

    d0, d1 = args.plot_dim0, args.plot_dim1
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
                # Render the predicted subgoal in a second env instance, one frame per executed step.
                # ``env_frames`` is step-aligned with one extra initial frame (``T+1`` total),
                # ``hats_per_step`` has T entries (one hat per env step). Drop the initial env
                # frame so the two panels share the same length.
                if int(env_frames.shape[0]) <= 1 or int(hats_per_step.shape[0]) == 0:
                    raise RuntimeError('No subgoal-aligned frames to render.')
                step_env_frames = env_frames[1:]
                T_steps = min(int(step_env_frames.shape[0]), int(hats_per_step.shape[0]))
                step_env_frames = step_env_frames[:T_steps]

                sub_env, _, _ = make_env_and_datasets(
                    env_name, frame_stack=cfg.get('frame_stack'), render_mode='rgb_array',
                )
                sub_env.reset(options=dict(task_id=tid, render_goal=False))
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
                caption = ['left: env state', 'right: env @ predicted subgoal']
                write_rgb_array_mp4(frames, mp4_out, float(args.fps), caption_lines=caption)
            else:
                _pf = min(action_chunk_horizon, dynamics_N)
                frames = overlay_rgb_frames_obs2d_panel(
                    env_frames,
                    traj,
                    roll,
                    hats,
                    frame_plan_trajs,
                    s0,
                    s_g,
                    d0,
                    d1,
                    plot_nav,
                    env_name=env_name,
                    chunk_hat_stride=_pf,
                    value_heatmap=heat_mesh,
                    value_heatmap_vmin=heat_vmin,
                    value_heatmap_vmax=heat_vmax,
                )
                write_rgb_array_mp4(frames, mp4_out, float(args.fps))
            print(f'Wrote MP4 {mp4_out.resolve()}')
        except Exception as e:
            print(f'Warning: MP4 failed ({e!r}). pip install imageio imageio-ffmpeg')
    elif not args.no_mp4:
        print('Warning: no RGB frames; check render_mode')


if __name__ == '__main__':
    main()
