#!/usr/bin/env python3
"""ManipSpace ``*-play-v0`` (cube/puzzle) **virtual state-space rollout** — no env.step.

Per task this script does the manip analogue of ``rollout/subgoal.py`` (maze):

1. ``env.reset(options=task_id, render_goal=False)`` to fix ``s_g = info['goal']`` and
   capture initial mocap (goal markers) snapshot.
2. Iterate, starting from ``s = s0``, repeating up to ``--max_chunks`` times::

       hat = predict_subgoal(s, s_g)
       traj = plan(s, hat)      # reverse chain (N+1 states)
       walk trajectory[1:K+1]  (K = ``--action_chunk_horizon``; 0 ⇒ ``dynamics_N``)
       s ← trajectory[K]

   For each appended virtual state ``s_t`` we ``sync_env_state_from_compact_manip_obs``
   into the **state-render env** and re-paste the captured mocap so goal markers stay put;
   ``hat`` is recorded per step (constant across one chunk) and rendered separately on a
   second env via the shared ``_render_subgoal_frames`` helper.
3. Stop early (unless ``--no_stop_on_success``) when the manip env itself reports
   success on the synthesized state — i.e. after ``sync_env_state_from_compact_manip_obs``
   we call ``env.unwrapped._compute_successes()`` (cube: all blocks within 4 cm of mocap
   targets; puzzle: button states match targets) and break on ``all(successes)``.
4. Compose ``left=state(synthesized)`` / ``right=predicted subgoal`` MP4 and write one per task.

Output dir: ``<run_dir>/rollouts_manip_<env_slug>_ep<EPOCH>_state/``. Task summary CSV is
written alongside the MP4s.

Execute: ``python -m rollout.manip_play_state_rollout --run_dir=...``
GPU: ``export JAX_PLATFORMS=cuda`` before running; defaults to CPU.
"""

from __future__ import annotations

import csv
import os
from typing import Any

if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import contextlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.dynamics import DynamicsAgent
from rollout.common import manip_play_family, slug_from_env
from rollout.env import (
    apply_snapshot_manip_mocap,
    configure_mujoco_gl,
    env_render_rgb_u8,
    snapshot_manip_mocap,
    sync_env_state_from_compact_manip_obs,
)
from rollout.manip_play_rollouts import _reset_task_env, _write_state_subgoal_mp4
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    load_run_flags,
    parse_int_list,
    pick_epoch,
    resolve_dynamics_checkpoint_dir,
)


def _env_state_is_success(env) -> bool:
    """Read manip env ``_compute_successes`` after a state-set. Task mode only."""
    u = getattr(env, 'unwrapped', env)
    fn = getattr(u, '_compute_successes', None)
    if fn is None:
        return False
    try:
        succ = fn()
    except Exception:
        return False
    if not isinstance(succ, (list, tuple)) or len(succ) == 0:
        return False
    return bool(all(bool(x) for x in succ))


@contextlib.contextmanager
def _exclusive_out_dir_lock(out_dir: Path):
    try:
        import fcntl as _fcntl
    except ImportError:
        yield
        return
    lock_path = out_dir / '.manip_play_state_rollout.lock'
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = open(lock_path, 'a', encoding='utf-8')
    try:
        _fcntl.flock(fp.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError as e:
        fp.close()
        raise SystemExit(
            f'이미 다른 manip_play_state_rollout 가 {out_dir} 를 쓰는 중입니다 ({lock_path}). '
            f'끝날 때까지 기다리거나 ``--out_dir`` 로 다른 디렉터리를 쓰세요.'
        ) from e
    try:
        yield
    finally:
        try:
            _fcntl.flock(fp.fileno(), _fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            fp.close()
        except OSError:
            pass


def _path_rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


_ROLLOUT_SUMMARY_FIELDS: tuple[str, ...] = (
    'task_id',
    'env_name',
    'family',
    'checkpoint_epoch',
    'max_chunks',
    'action_chunk_horizon',
    'dynamics_N',
    'state_steps',
    'stop_on_success',
    'env_success',
    'state_mp4',
    'state_mp4_frames',
)


def _write_summary_csv(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = out_dir / 'rollout_task_summary.csv'
    if not rows:
        return path
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(_ROLLOUT_SUMMARY_FIELDS), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return path


def _virtual_rollout_with_render(
    *,
    env,
    agent: DynamicsAgent,
    s0: np.ndarray,
    s_g: np.ndarray,
    max_chunks: int,
    action_chunk_horizon: int,
    stop_on_success: bool,
    mocap: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, list[np.ndarray], int, bool]:
    """Open-loop virtual state rollout. Returns
    ``(state_frames, subgoals_per_step, n_state_steps, env_success)``.

    ``state_frames`` always starts with the rendered ``s0`` frame, then one frame per
    walked virtual state. ``subgoals_per_step`` has one entry per walked step (constant
    across a chunk). ``env_success`` is True iff the manip env's
    ``_compute_successes`` ever reports ``all(...)`` on the synthesized state.
    """
    fr0 = env_render_rgb_u8(env)
    if fr0 is None:
        raise RuntimeError('Failed to render initial state frame for env.reset().')
    state_frames: list[np.ndarray] = [fr0]
    subgoals_per_step: list[np.ndarray] = []

    dynamics_N = int(agent.config['dynamics_N'])
    K_req = dynamics_N if int(action_chunk_horizon) <= 0 else int(action_chunk_horizon)
    if K_req < 1:
        raise ValueError('--action_chunk_horizon must be >= 0 (0 = dynamics_N).')

    s = jnp.asarray(s0, dtype=jnp.float32)
    g = jnp.asarray(s_g, dtype=jnp.float32)

    total_steps = 0
    env_success = False
    for _chunk in range(int(max_chunks)):
        hat = agent.predict_subgoal(s, g)
        hat_np = np.asarray(jax.device_get(hat), dtype=np.float32).reshape(-1)
        out = agent.plan(s, hat)
        traj = np.asarray(jax.device_get(out['trajectory']), dtype=np.float32)
        if traj.ndim == 3:
            traj = traj[0]
        n_bridge = int(traj.shape[0]) - 1
        k = min(K_req, n_bridge)
        if k < 1:
            break

        for j in range(k):
            sn = np.asarray(traj[j + 1], dtype=np.float32).reshape(-1)
            sync_env_state_from_compact_manip_obs(env, sn)
            if mocap is not None:
                apply_snapshot_manip_mocap(env, mocap[0], mocap[1])
            fr = env_render_rgb_u8(env)
            if fr is None:
                raise RuntimeError(f'Failed to render virtual state frame at step {total_steps}.')
            state_frames.append(fr)
            subgoals_per_step.append(hat_np.copy())
            total_steps += 1
            s = jnp.asarray(sn, dtype=jnp.float32)
            if _env_state_is_success(env):
                env_success = True
                if stop_on_success:
                    break
        if env_success and stop_on_success:
            break

    return (
        np.stack(state_frames, axis=0),
        subgoals_per_step,
        int(total_steps),
        bool(env_success),
    )


def _run_one_task(
    run_dir: Path,
    task_id: int,
    ckpt_epoch: int,
    out_task_dir: Path,
    out_dir: Path,
    *,
    mujoco_gl: str,
    seed: int,
    max_chunks: int,
    action_chunk_horizon: int,
    stop_on_success: bool,
    fps: float,
    min_mp4_seconds: float,
    show_goal_panel: bool = True,
) -> dict[str, Any]:
    configure_mujoco_gl(mujoco_gl)
    cfg, env_name = load_run_flags(run_dir)
    family = manip_play_family(env_name)

    env, _, _ = make_env_and_datasets(
        env_name,
        frame_stack=cfg.get('frame_stack'),
        render_mode='rgb_array',
    )
    u = env.unwrapped
    n_tasks = int(getattr(u, 'num_tasks', 5))
    if not (1 <= int(task_id) <= n_tasks):
        raise ValueError(f'task_id must be in [1, {n_tasks}]')

    s0, s_g, goal_rendered = _reset_task_env(env, int(task_id), show_goal_panel=show_goal_panel)
    mocap = snapshot_manip_mocap(env)

    dyn_dir = resolve_dynamics_checkpoint_dir(run_dir)
    ckpt_epoch = pick_epoch(int(ckpt_epoch), list_checkpoint_suffixes(dyn_dir))

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = DynamicsAgent.create(int(seed), ex, cfg, ex_actions=ex_act)
    dyn_pkl = dyn_dir / f'params_{ckpt_epoch}.pkl'
    agent = load_checkpoint_pkl(agent, dyn_pkl)
    dynamics_N = int(agent.config['dynamics_N'])
    K_eff = dynamics_N if int(action_chunk_horizon) <= 0 else int(action_chunk_horizon)

    state_frames, subgoals_per_step, n_steps, env_success = _virtual_rollout_with_render(
        env=env,
        agent=agent,
        s0=s0,
        s_g=s_g,
        max_chunks=int(max_chunks),
        action_chunk_horizon=int(action_chunk_horizon),
        stop_on_success=bool(stop_on_success),
        mocap=mocap,
    )

    out_tag = '_success' if env_success else '_fail'
    mp4_path = out_task_dir / f'state_env_rgb{out_tag}.mp4'
    mp4_rel = ''
    mp4_frames = 0
    if state_frames.size > 0 and subgoals_per_step:
        mocap_list = (
            [mocap] * len(subgoals_per_step) if mocap is not None else None
        )
        mp4_frames = _write_state_subgoal_mp4(
            env_name=env_name,
            frame_stack=cfg.get('frame_stack'),
            task_id=int(task_id),
            state_frames=state_frames,
            subgoals_per_step=subgoals_per_step,
            path=mp4_path,
            fps=float(fps),
            min_mp4_seconds=float(min_mp4_seconds),
            mocap_snapshots=mocap_list,
            s_g=s_g,
            goal_rendered=goal_rendered,
            show_goal_panel=show_goal_panel,
        )
        mp4_rel = _path_rel_to(out_dir, mp4_path)
        print(
            f'[task {task_id}] wrote {mp4_path}  state_steps={n_steps} K={K_eff} '
            f'dynamics_N={dynamics_N} max_chunks={max_chunks} env_success={env_success} '
            f'stop_on_success={stop_on_success} mp4_frames={mp4_frames}'
        )
    else:
        print(
            f'[task {task_id}] state rollout produced no frames (steps={n_steps}). '
            'Did plan() return empty trajectory?'
        )

    try:
        env.close()
    except Exception:
        pass

    return {
        'task_id': int(task_id),
        'env_name': str(env_name),
        'family': str(family),
        'checkpoint_epoch': int(ckpt_epoch),
        'max_chunks': int(max_chunks),
        'action_chunk_horizon': int(action_chunk_horizon),
        'dynamics_N': int(dynamics_N),
        'state_steps': int(n_steps),
        'stop_on_success': int(bool(stop_on_success)),
        'env_success': int(bool(env_success)),
        'state_mp4': mp4_rel,
        'state_mp4_frames': int(mp4_frames),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--checkpoint_epoch', type=int, default=1000)
    p.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='Default: <run_dir>/rollouts_manip_<env_slug>_ep<EPOCH>_state.',
    )
    p.add_argument('--task_ids', type=str, default='1,2,3,4,5')
    p.add_argument('--mujoco_gl', type=str, default='osmesa')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument(
        '--max_chunks',
        type=int,
        default=40,
        help='Outer replans (each calls predict_subgoal + plan once). Default 40.',
    )
    p.add_argument(
        '--action_chunk_horizon',
        type=int,
        default=0,
        help=(
            'States to walk per plan() (trajectory[1:K+1]). 0 = dynamics_N (full bridge). '
            'K=1 = single-step replan (next_step only). Capped by dynamics_N.'
        ),
    )
    p.add_argument(
        '--no_stop_on_success',
        action='store_true',
        help=(
            'Walk the full --max_chunks budget even if the env reports success on a '
            'synthesized state. Default: stop on first env success.'
        ),
    )
    p.add_argument(
        '--min_mp4_seconds',
        type=float,
        default=0.0,
        help='If >0, pad MP4 by repeating last frame until at least this duration.',
    )
    p.add_argument(
        '--show_goal_panel',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Append a third MP4 panel showing the task goal configuration (default: on).',
    )
    p.add_argument(
        '--no_exclusive_lock',
        action='store_true',
        help='기본 배타 락(out_dir/.manip_play_state_rollout.lock)을 쓰지 않음 (디버그용).',
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt = int(args.checkpoint_epoch)
    _, env_nm = load_run_flags(run_dir)
    slug = slug_from_env(env_nm)
    out_arg = str(args.out_dir).strip()
    out_dir = (
        Path(out_arg).resolve()
        if out_arg
        else run_dir / f'rollouts_manip_{slug}_ep{ckpt}_state'
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tids = parse_int_list(str(args.task_ids))
    if not tids:
        raise SystemExit('empty --task_ids')

    rows: list[dict[str, Any]] = []
    lock_cm = contextlib.nullcontext() if bool(args.no_exclusive_lock) else _exclusive_out_dir_lock(out_dir)
    with lock_cm:
        for tid in tids:
            sub = out_dir / f'task{tid}'
            sub.mkdir(parents=True, exist_ok=True)
            rows.append(
                _run_one_task(
                    run_dir,
                    int(tid),
                    ckpt,
                    sub,
                    out_dir,
                    mujoco_gl=str(args.mujoco_gl),
                    seed=int(args.seed),
                    max_chunks=int(args.max_chunks),
                    action_chunk_horizon=int(args.action_chunk_horizon),
                    stop_on_success=not bool(args.no_stop_on_success),
                    fps=float(args.fps),
                    min_mp4_seconds=float(args.min_mp4_seconds),
                    show_goal_panel=bool(args.show_goal_panel),
                )
            )
        summary_path = _write_summary_csv(out_dir, rows)
    print(f'done out_dir={out_dir} wrote {summary_path}')


__all__ = ['main']


if __name__ == '__main__':
    main()
