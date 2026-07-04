#!/usr/bin/env python3
"""OGBench **ManipSpace ``*-play-v0``** (cube / puzzle 등) IDM/actor rollout + 영상.

`flags.json` 의 `env_name` 만 보고 cube/puzzle을 자동 인식한다. 한 task마다
IDM rollout과 actor rollout을 모두 실행하며, 각각 success/fail MP4 +
`rollout_task_summary.csv` 를 남긴다 (state-space open-loop plot은 그리지 않음).

성공 판정은 항상 env의 ``info['success']`` (any step) 만으로 결정된다 — 사용자가
정의한 거리 임계치 같은 임의 기준은 사용하지 않는다.

기본 출력: ``<run_dir>/rollouts_manip_<env_slug>_ep<EPOCH>/``

같은 ``out_dir`` 에는 파일 락 (``.manip_play_rollouts.lock``) 으로 한 번에 하나의
프로세스만 실행되게 한다.

실행: ``python -m rollout.manip_play_rollouts --run_dir=...``

JAX: 오프라인 롤아웃은 기본 CPU (``JAX_PLATFORMS`` 미설정 시 여기서 ``cpu`` 로 고정).
GPU로 돌리려면 실행 전 ``export JAX_PLATFORMS=cuda``.
"""

from __future__ import annotations

import csv
import os
from typing import Any

if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import contextlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.actor import ActorAgent, get_actor_config
from agents.dynamics import DynamicsAgent
from rollout.common import align_action_to_env, manip_play_family, slug_from_env
from rollout.env import (
    apply_snapshot_manip_mocap,
    configure_mujoco_gl,
    env_render_rgb_u8,
    max_episode_steps_from_wrappers,
    snapshot_manip_mocap,
    sync_env_state_from_compact_manip_obs,
)
from rollout.episode_runner import (
    run_chunked_episode,
)
from rollout.plot import compose_state_subgoal_env_frames, write_rgb_array_mp4
from rollout.value_field import load_critic_for_run
from main import _idm_action_chunk
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    load_run_flags,
    parse_int_list,
    pick_epoch,
    resolve_actor_checkpoint_dir,
    resolve_dynamics_checkpoint_dir,
)


def _eval_subgoal_selection(cfg: dict[str, Any]) -> str:
    return str(cfg.get('subgoal_eval_selection', 'zero_noise')).lower()


def _uses_eval_subgoal(cfg: dict[str, Any]) -> bool:
    return _eval_subgoal_selection(cfg) in ('best_of_n_value', 'best_of_n_goal_l2', 'sample')


def _predict_subgoal_for_rollout(
    agent: DynamicsAgent,
    obs: np.ndarray,
    goal: np.ndarray,
    *,
    critic_agent: Any | None,
    cfg: dict[str, Any],
    ep_ix: int,
) -> np.ndarray:
    if _uses_eval_subgoal(cfg):
        seed_base = int(cfg.get('subgoal_eval_seed', 0))
        rng = jax.random.PRNGKey(seed_base + int(ep_ix))
        pred = agent.infer_subgoal_for_eval(
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(goal, dtype=jnp.float32),
            critic_agent=critic_agent,
            rng=rng,
        )
        return np.asarray(pred, dtype=np.float32).reshape(-1)
    return np.asarray(agent.infer_subgoal(obs, goal), dtype=np.float32).reshape(-1)


def _chunk_budget_for_full_episode(env, chunk_h: int) -> int:
    ms = max_episode_steps_from_wrappers(env)
    ch = max(1, int(chunk_h))
    if ms is None:
        raise ValueError('Rollout env must expose max_episode_steps.')
    need = (int(ms) + ch - 1) // ch
    return max(1, int(need))


def _load_eval_rollout_limits(run_dir: Path) -> tuple[int, int]:
    flags_path = run_dir / 'flags.json'
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    critic = root.get('critic_agent') if isinstance(root.get('critic_agent'), dict) else {}
    actor = root.get('actor') if isinstance(root.get('actor'), dict) else {}
    idm_h = max(1, int(critic.get('action_chunk_horizon', 5)))
    act_h = max(1, int(actor.get('actor_chunk_horizon', idm_h)))
    return idm_h, act_h


def _load_actor_cfg(flags_path: Path) -> dict:
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    act = root.get('actor')
    if not isinstance(act, dict):
        raise KeyError(f'{flags_path} must contain an "actor" object.')
    base = get_actor_config()
    for k, v in act.items():
        base[k] = v
    return dict(base)


def _pad_rgb_frames_min_duration(frames: np.ndarray, fps: float, min_seconds: float) -> np.ndarray:
    if frames is None or frames.size == 0 or min_seconds <= 0 or fps <= 0:
        return frames
    n = int(frames.shape[0])
    need = int(np.ceil(float(min_seconds) * float(fps))) - n
    if need <= 0:
        return frames
    tail = np.repeat(np.asarray(frames[-1:], dtype=np.uint8), need, axis=0)
    return np.concatenate([frames, tail], axis=0)


@contextlib.contextmanager
def _exclusive_out_dir_lock(out_dir: Path):
    """같은 ``out_dir`` 에 롤아웃이 동시에 두 개 뜨지 않게 비차단 배타 락 (POSIX)."""
    try:
        import fcntl as _fcntl
    except ImportError:
        yield
        return
    lock_path = out_dir / '.manip_play_rollouts.lock'
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = open(lock_path, 'a', encoding='utf-8')
    try:
        _fcntl.flock(fp.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except BlockingIOError as e:
        fp.close()
        raise SystemExit(
            f'이미 다른 manip_play_rollouts 가 {out_dir} 를 쓰는 중입니다 ({lock_path}). '
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
    'actor_checkpoint_epoch',
    'eval_budget',
    'idm_horizon',
    'actor_horizon',
    'eval_ep_ix',
    'idm_chunks',
    'actor_chunks',
    'idm_env_success',
    'actor_env_success',
    'idm_mp4',
    'actor_mp4',
    'idm_mp4_frames',
    'actor_mp4_frames',
)


def _write_rollout_task_summary_csv(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """태스크별 성공/지표를 ``rollout_task_summary.csv`` 로 저장 (``out_dir`` 기준 상대 경로 열)."""
    path = out_dir / 'rollout_task_summary.csv'
    if not rows:
        return path
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(_ROLLOUT_SUMMARY_FIELDS), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    return path


def _render_goal_reference_frames(
    env_name: str,
    frame_stack: int | None,
    task_id: int,
    s_g: np.ndarray,
    n_frames: int,
    *,
    goal_rendered: np.ndarray | None = None,
) -> np.ndarray:
    """Render the task goal ``s_g`` as a constant reference panel (repeated ``n_frames`` times)."""
    if n_frames < 1:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    if goal_rendered is not None:
        frame = np.asarray(goal_rendered, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f'goal_rendered must be (H,W,3), got {frame.shape}')
        return np.stack([frame] * int(n_frames), axis=0)

    sub_env, _, _ = make_env_and_datasets(env_name, frame_stack=frame_stack, render_mode='rgb_array')
    sub_env.reset(options=dict(task_id=int(task_id), render_goal=False))
    sync_env_state_from_compact_manip_obs(sub_env, np.asarray(s_g, dtype=np.float32))
    try:
        fr = env_render_rgb_u8(sub_env)
        if fr is None:
            raise RuntimeError('Failed to render goal reference frame.')
        frame = np.asarray(fr, dtype=np.uint8)
    finally:
        try:
            sub_env.close()
        except Exception:
            pass
    return np.stack([frame] * int(n_frames), axis=0)


def _render_subgoal_frames(
    env_name: str,
    frame_stack: int | None,
    task_id: int,
    subgoals: np.ndarray,
    n_frames: int,
    mocap_snapshots: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Render predicted subgoals in a second ManipSpace env, one frame per executed env step.

    When ``mocap_snapshots`` is provided (one entry per frame, from the main rollout env),
    mocap poses are pasted after ``sync_env_state_from_compact_manip_obs`` so goal markers
    match the left panel (fixes independent ``permute_blocks`` RNG on the render env).
    """
    if n_frames < 1:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    sub_env, _, _ = make_env_and_datasets(env_name, frame_stack=frame_stack, render_mode='rgb_array')
    sub_env.reset(options=dict(task_id=int(task_id), render_goal=False))
    frames: list[np.ndarray] = []
    try:
        if subgoals.size == 0:
            raise RuntimeError('No predicted subgoals were recorded for subgoal render composition.')
        for t in range(int(n_frames)):
            sg = subgoals[min(t, int(subgoals.shape[0]) - 1)]
            sync_env_state_from_compact_manip_obs(sub_env, sg)
            if mocap_snapshots is not None and t < len(mocap_snapshots):
                mp, mq = mocap_snapshots[t]
                apply_snapshot_manip_mocap(sub_env, mp, mq)
            fr = env_render_rgb_u8(sub_env)
            if fr is None:
                raise RuntimeError(f'Failed to render subgoal frame at step {t}.')
            frames.append(fr)
    finally:
        try:
            sub_env.close()
        except Exception:
            pass
    return np.stack(frames, axis=0)


def _write_state_subgoal_mp4(
    *,
    env_name: str,
    frame_stack: int | None,
    task_id: int,
    state_frames: np.ndarray,
    subgoals_per_step: list[np.ndarray],
    path: Path,
    fps: float,
    min_mp4_seconds: float,
    mocap_snapshots: list[tuple[np.ndarray, np.ndarray]] | None = None,
    s_g: np.ndarray | None = None,
    goal_rendered: np.ndarray | None = None,
    show_goal_panel: bool = False,
    show_caption: bool = False,
) -> int:
    """Write left=actual env, right=predicted-subgoal env MP4 and return frame count."""
    # ``run_chunked_episode`` records one initial frame plus one frame per executed env step.
    step_frames = np.asarray(state_frames[1:], dtype=np.uint8)
    if step_frames.size == 0:
        step_frames = np.asarray(state_frames, dtype=np.uint8)
    subgoals = (
        np.stack(subgoals_per_step, axis=0).astype(np.float32)
        if subgoals_per_step
        else np.zeros((0, 0), dtype=np.float32)
    )
    n = min(int(step_frames.shape[0]), int(subgoals.shape[0]))
    if n <= 0:
        raise RuntimeError('No aligned state/subgoal frames to write.')
    step_frames = step_frames[:n]
    mocap_use = None
    if mocap_snapshots is not None and len(mocap_snapshots) > 0:
        mocap_use = mocap_snapshots[:n]
        if len(mocap_use) != n:
            raise RuntimeError(
                f'mocap_snapshots length {len(mocap_use)} != aligned subgoal frames {n} '
                f'(state_frames={int(state_frames.shape[0])}, subgoals={len(subgoals_per_step)})'
            )
    subgoal_frames = _render_subgoal_frames(
        env_name, frame_stack, task_id, subgoals[:n], n, mocap_snapshots=mocap_use
    )
    goal_frames = None
    if show_goal_panel and s_g is not None:
        goal_frames = _render_goal_reference_frames(
            env_name,
            frame_stack,
            task_id,
            np.asarray(s_g, dtype=np.float32),
            n,
            goal_rendered=goal_rendered,
        )
    composed = compose_state_subgoal_env_frames(
        step_frames,
        subgoal_frames,
        goal_frames=goal_frames,
        output_scale=1.1,
        label_left='state',
        label_right='predicted subgoal',
        label_goal='goal' if goal_frames is not None else None,
    )
    frames = _pad_rgb_frames_min_duration(composed, float(fps), float(min_mp4_seconds))
    caption_lines = None
    if show_caption:
        caption_lines = ['left: env state', 'middle: env @ predicted subgoal']
        if goal_frames is not None:
            caption_lines.append('right: task goal (target configuration)')
    write_rgb_array_mp4(
        frames,
        path,
        float(fps),
        caption_lines=caption_lines,
    )
    return int(frames.shape[0])


def _reset_task_env(
    env,
    task_id: int,
    *,
    show_goal_panel: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=bool(show_goal_panel)))
    if 'goal' not in info:
        raise RuntimeError('reset did not set info["goal"]')
    s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
    s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
    goal_rendered = info.get('goal_rendered') if show_goal_panel else None
    if goal_rendered is not None:
        goal_rendered = np.asarray(goal_rendered, dtype=np.uint8)
    return s0, s_g, goal_rendered


def _run_one_task(
    run_dir: Path,
    task_id: int,
    ckpt_epoch: int,
    out_task_dir: Path,
    out_dir: Path,
    *,
    mujoco_gl: str,
    seed: int,
    idm_max_chunks: int,
    actor_max_chunks: int,
    fps: float,
    min_mp4_seconds: float,
    show_goal_panel: bool = True,
    idm_horizon: int | None = None,
    idm_only: bool = False,
    subgoal_eval_num_samples: int = -1,
    eval_ep_ix: int = 0,
    until_success: bool = False,
    max_eval_ep_ix: int = 25,
    show_caption: bool = False,
) -> dict[str, Any]:
    configure_mujoco_gl(mujoco_gl)
    cfg, env_name = load_run_flags(run_dir)
    if int(subgoal_eval_num_samples) > 0:
        cfg['subgoal_eval_num_samples'] = int(subgoal_eval_num_samples)
    family = manip_play_family(env_name)
    idm_h, act_h = _load_eval_rollout_limits(run_dir)
    if idm_horizon is not None and int(idm_horizon) > 0:
        idm_h = int(idm_horizon)

    env, train_raw, _ = make_env_and_datasets(
        env_name,
        frame_stack=cfg.get('frame_stack'),
        render_mode='rgb_array',
    )
    u = env.unwrapped
    n_tasks = int(getattr(u, 'num_tasks', 5))
    if not (1 <= int(task_id) <= n_tasks):
        raise ValueError(f'task_id must be in [1, {n_tasks}]')

    env, train_raw, _ = make_env_and_datasets(
        env_name,
        frame_stack=cfg.get('frame_stack'),
        render_mode='rgb_array',
    )
    u = env.unwrapped
    n_tasks = int(getattr(u, 'num_tasks', 5))
    if not (1 <= int(task_id) <= n_tasks):
        raise ValueError(f'task_id must be in [1, {n_tasks}]')

    dyn_dir = resolve_dynamics_checkpoint_dir(run_dir)
    ckpt_epoch = pick_epoch(int(ckpt_epoch), list_checkpoint_suffixes(dyn_dir))
    act_dir = resolve_actor_checkpoint_dir(run_dir, required=True)
    act_ep = pick_epoch(int(ckpt_epoch), list_checkpoint_suffixes(act_dir))

    ex = jnp.zeros((1, int(np.prod(env.observation_space.shape))), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = DynamicsAgent.create(int(seed), ex, cfg, ex_actions=ex_act)
    dyn_pkl = dyn_dir / f'params_{ckpt_epoch}.pkl'
    agent = load_checkpoint_pkl(agent, dyn_pkl)

    critic_agent = None
    if _uses_eval_subgoal(cfg):
        critic_agent = load_critic_for_run(
            run_dir, int(ckpt_epoch), env, train_raw, seed=int(seed),
        )

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    def _run_idm_actor_rollouts(ep_ix: int, *, phase: str = 'both') -> dict[str, Any]:
        run_idm = phase in ('both', 'idm')
        run_actor = phase in ('both', 'actor')
        idm_outcome = None
        idm_mp4_rel = ''
        idm_mp4_frames = 0
        idm_subgoals_per_step: list[np.ndarray] = []

        if run_idm:
            s0, s_g, goal_rendered = _reset_task_env(env, int(task_id), show_goal_panel=show_goal_panel)
            idm_mocap_snapshots: list[tuple[np.ndarray, np.ndarray]] = []

            def _idm_post_step_mocap(e: Any) -> None:
                snap = snapshot_manip_mocap(e)
                if snap is not None:
                    idm_mocap_snapshots.append(snap)

            def _idm_chunk_with_subgoal_trace(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
                pred = _predict_subgoal_for_rollout(
                    agent, obs, goal, critic_agent=critic_agent, cfg=cfg, ep_ix=ep_ix,
                )
                chunk = _idm_action_chunk(agent, np.asarray(obs, dtype=np.float32).reshape(-1), pred, int(idm_h))
                for _ in range(int(chunk.shape[0])):
                    idm_subgoals_per_step.append(pred.copy())
                return chunk

            idm_outcome = run_chunked_episode(
                env,
                s0,
                s_g,
                low=low,
                high=high,
                max_chunks=int(idm_chunks),
                sample_action_chunk=_idm_chunk_with_subgoal_trace,
                post_step_hook=_idm_post_step_mocap,
                record_rgb=True,
            )
            idm_out_tag = 'success' if idm_outcome.ok_env else 'fail'
            mp4_idm = out_task_dir / f'idm_env_rgb_{idm_out_tag}.mp4'
            if idm_outcome.rgb_frames is not None and idm_outcome.rgb_frames.size > 0:
                idm_mp4_frames = _write_state_subgoal_mp4(
                    env_name=env_name,
                    frame_stack=cfg.get('frame_stack'),
                    task_id=int(task_id),
                    state_frames=idm_outcome.rgb_frames,
                    subgoals_per_step=idm_subgoals_per_step,
                    path=mp4_idm,
                    fps=float(fps),
                    min_mp4_seconds=float(min_mp4_seconds),
                    mocap_snapshots=idm_mocap_snapshots if idm_mocap_snapshots else None,
                    s_g=s_g,
                    goal_rendered=goal_rendered,
                    show_goal_panel=show_goal_panel,
                    show_caption=show_caption,
                )
                idm_mp4_rel = _path_rel_to(out_dir, mp4_idm)
                print(
                    f'[task {task_id}] wrote {mp4_idm}  raw_frames={idm_outcome.rgb_frames.shape[0]} '
                    f'mp4_frames={idm_mp4_frames} chunks={idm_outcome.n_chunks} idm_horizon={idm_h} '
                    f'eval_ep_ix={ep_ix} eval_budget=env_max_episode_steps env_info_success={idm_outcome.ok_env}'
                )
            else:
                print(f'[task {task_id}] IDM: no RGB frames states={idm_outcome.states.shape[0]} eval_ep_ix={ep_ix}')

            if idm_only or phase == 'idm':
                return {
                    'task_id': int(task_id),
                    'env_name': str(env_name),
                    'family': str(family),
                    'checkpoint_epoch': int(ckpt_epoch),
                    'actor_checkpoint_epoch': int(act_ep),
                    'eval_budget': 'env_max_episode_steps',
                    'idm_horizon': int(idm_h),
                    'actor_horizon': int(act_h),
                    'eval_ep_ix': int(ep_ix),
                    'idm_chunks': int(idm_outcome.n_chunks),
                    'actor_chunks': 0,
                    'idm_env_success': int(bool(idm_outcome.ok_env)),
                    'actor_env_success': 0,
                    'idm_mp4': idm_mp4_rel,
                    'actor_mp4': '',
                    'idm_mp4_frames': idm_mp4_frames,
                    'actor_mp4_frames': 0,
                }

        actor_outcome = None
        actor_mp4_rel = ''
        actor_mp4_frames = 0
        if run_actor:
            flags_path = run_dir / 'flags.json'
            actor_cfg = _load_actor_cfg(flags_path)
            actor_cfg['action_dim'] = act_dim
            ex_goal = jnp.asarray(
                _reset_task_env(env, int(task_id), show_goal_panel=show_goal_panel)[1].reshape(1, -1),
                dtype=jnp.float32,
            )
            actor_agent = ActorAgent.create(int(seed), ex, actor_cfg, ex_goals=ex_goal)
            actor_pkl = act_dir / f'params_{act_ep}.pkl'
            actor_agent = load_checkpoint_pkl(actor_agent, actor_pkl)

            actor_chunks = (
                int(actor_max_chunks)
                if int(actor_max_chunks) >= 0
                else _chunk_budget_for_full_episode(env, int(act_h))
            )
            s0, s_g, goal_rendered = _reset_task_env(env, int(task_id), show_goal_panel=show_goal_panel)
            actor_subgoals_per_step: list[np.ndarray] = []
            actor_mocap_snapshots: list[tuple[np.ndarray, np.ndarray]] = []

            def _actor_post_step_mocap(e: Any) -> None:
                snap = snapshot_manip_mocap(e)
                if snap is not None:
                    actor_mocap_snapshots.append(snap)

            def _actor_chunk_with_subgoal_trace(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
                pred = _predict_subgoal_for_rollout(
                    agent, obs, goal, critic_agent=critic_agent, cfg=cfg, ep_ix=ep_ix,
                )
                chunk = np.asarray(
                    actor_agent.sample_actions(
                        jnp.asarray(obs, dtype=jnp.float32),
                        jnp.asarray(pred, dtype=jnp.float32),
                    ),
                    dtype=np.float32,
                ).reshape(int(act_h), -1)
                if not chunk.flags.writeable:
                    chunk = chunk.copy()
                for i in range(int(chunk.shape[0])):
                    chunk[i] = align_action_to_env(chunk[i], int(act_dim))
                    actor_subgoals_per_step.append(pred.copy())
                return chunk

            actor_outcome = run_chunked_episode(
                env,
                s0,
                s_g,
                low=low,
                high=high,
                max_chunks=int(actor_chunks),
                sample_action_chunk=_actor_chunk_with_subgoal_trace,
                post_step_hook=_actor_post_step_mocap,
                record_rgb=True,
            )
            ac_out_tag = 'success' if actor_outcome.ok_env else 'fail'
            mp4_ac = out_task_dir / f'actor_env_rgb_{ac_out_tag}.mp4'
            if actor_outcome.rgb_frames is not None and actor_outcome.rgb_frames.size > 0:
                actor_mp4_frames = _write_state_subgoal_mp4(
                    env_name=env_name,
                    frame_stack=cfg.get('frame_stack'),
                    task_id=int(task_id),
                    state_frames=actor_outcome.rgb_frames,
                    subgoals_per_step=actor_subgoals_per_step,
                    path=mp4_ac,
                    fps=float(fps),
                    min_mp4_seconds=float(min_mp4_seconds),
                    mocap_snapshots=actor_mocap_snapshots if actor_mocap_snapshots else None,
                    s_g=s_g,
                    goal_rendered=goal_rendered,
                    show_goal_panel=show_goal_panel,
                    show_caption=show_caption,
                )
                actor_mp4_rel = _path_rel_to(out_dir, mp4_ac)
                print(
                    f'[task {task_id}] wrote {mp4_ac}  raw_frames={actor_outcome.rgb_frames.shape[0]} '
                    f'mp4_frames={actor_mp4_frames} chunks={actor_outcome.n_chunks} actor_horizon={act_h} '
                    f'eval_ep_ix={ep_ix} eval_budget=env_max_episode_steps env_info_success={actor_outcome.ok_env}'
                )
            else:
                print(f'[task {task_id}] actor: no RGB frames states={actor_outcome.states.shape[0]} eval_ep_ix={ep_ix}')

            if phase == 'actor':
                return {
                    'task_id': int(task_id),
                    'env_name': str(env_name),
                    'family': str(family),
                    'checkpoint_epoch': int(ckpt_epoch),
                    'actor_checkpoint_epoch': int(act_ep),
                    'eval_budget': 'env_max_episode_steps',
                    'idm_horizon': int(idm_h),
                    'actor_horizon': int(act_h),
                    'eval_ep_ix': int(ep_ix),
                    'idm_chunks': 0,
                    'actor_chunks': int(actor_outcome.n_chunks),
                    'idm_env_success': 0,
                    'actor_env_success': int(bool(actor_outcome.ok_env)),
                    'idm_mp4': '',
                    'actor_mp4': actor_mp4_rel,
                    'idm_mp4_frames': 0,
                    'actor_mp4_frames': actor_mp4_frames,
                }

        assert idm_outcome is not None and actor_outcome is not None
        return {
            'task_id': int(task_id),
            'env_name': str(env_name),
            'family': str(family),
            'checkpoint_epoch': int(ckpt_epoch),
            'actor_checkpoint_epoch': int(act_ep),
            'eval_budget': 'env_max_episode_steps',
            'idm_horizon': int(idm_h),
            'actor_horizon': int(act_h),
            'eval_ep_ix': int(ep_ix),
            'idm_chunks': int(idm_outcome.n_chunks),
            'actor_chunks': int(actor_outcome.n_chunks),
            'idm_env_success': int(bool(idm_outcome.ok_env)),
            'actor_env_success': int(bool(actor_outcome.ok_env)),
            'idm_mp4': idm_mp4_rel,
            'actor_mp4': actor_mp4_rel,
            'idm_mp4_frames': idm_mp4_frames,
            'actor_mp4_frames': actor_mp4_frames,
        }

    idm_chunks = (
        int(idm_max_chunks)
        if int(idm_max_chunks) >= 0
        else _chunk_budget_for_full_episode(env, int(idm_h))
    )

    if bool(until_success):
        idm_row: dict[str, Any] | None = None
        actor_row: dict[str, Any] | None = None
        last_row: dict[str, Any] | None = None
        for ep_ix in range(max(1, int(max_eval_ep_ix))):
            row = _run_idm_actor_rollouts(int(ep_ix), phase='idm')
            last_row = row
            if bool(row.get('idm_env_success')):
                idm_row = row
                break
        for ep_ix in range(max(1, int(max_eval_ep_ix))):
            row = _run_idm_actor_rollouts(int(ep_ix), phase='actor')
            last_row = row
            if bool(row.get('actor_env_success')):
                actor_row = row
                break
        if idm_row is None or actor_row is None:
            if last_row is None:
                raise RuntimeError(f'task {task_id}: until_success found no rollout rows')
            return last_row
        merged = dict(idm_row)
        merged.update({
            'actor_env_success': actor_row.get('actor_env_success', 0),
            'actor_mp4': actor_row.get('actor_mp4', ''),
            'actor_mp4_frames': actor_row.get('actor_mp4_frames', 0),
            'actor_chunks': actor_row.get('actor_chunks', 0),
        })
        idm_ep = idm_row.get('eval_ep_ix')
        actor_ep = actor_row.get('eval_ep_ix')
        merged['eval_ep_ix'] = (
            int(idm_ep) if idm_ep == actor_ep else f'idm={idm_ep},actor={actor_ep}'
        )
        return merged

    return _run_idm_actor_rollouts(int(eval_ep_ix))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--checkpoint_epoch', type=int, default=1000)
    p.add_argument(
        '--out_dir',
        type=str,
        default='',
        help='Default: <run_dir>/rollouts_manip_<env_slug>_ep<EPOCH>.',
    )
    p.add_argument('--task_ids', type=str, default='1,2,3,4,5', help='Comma-separated; e.g. "1" for single task.')
    p.add_argument('--mujoco_gl', type=str, default='osmesa')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fps', type=float, default=30.0, help='MP4 playback FPS (default 30).')
    p.add_argument(
        '--idm_horizon',
        type=int,
        default=-1,
        help='IDM action-chunk length (default: flags critic action_chunk_horizon).',
    )
    p.add_argument(
        '--idm_only',
        action='store_true',
        help='Run IDM rollout only (skip actor).',
    )
    p.add_argument(
        '--idm_max_chunks',
        type=int,
        default=-1,
        help='Outer replans; -1 -> ceil(env max episode steps / IDM horizon).',
    )
    p.add_argument(
        '--actor_max_chunks',
        type=int,
        default=-1,
        help='Outer replans; -1 -> ceil(env max episode steps / actor horizon).',
    )
    p.add_argument(
        '--min_mp4_seconds',
        type=float,
        default=0.0,
        help='If >0, pad MP4 by repeating the last frame until at least this duration (0 = actual length only).',
    )
    p.add_argument(
        '--show_goal_panel',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Append a third MP4 panel showing the task goal configuration (default: on).',
    )
    p.add_argument(
        '--subgoal_eval_num_samples',
        type=int,
        default=-1,
        help='Override dynamics.subgoal_eval_num_samples for infer_subgoal (-1 = flags.json).',
    )
    p.add_argument(
        '--eval_ep_ix',
        type=int,
        default=0,
        help='Episode index for eval subgoal RNG (matches main eval: seed_base + ep_ix).',
    )
    p.add_argument(
        '--until_success',
        action='store_true',
        help='Try eval_ep_ix=0..max_eval_ep_ix-1 until IDM or actor succeeds.',
    )
    p.add_argument(
        '--max_eval_ep_ix',
        type=int,
        default=25,
        help='Upper bound for --until_success episode search.',
    )
    p.add_argument(
        '--show_caption',
        action='store_true',
        help='Burn English panel labels into the bottom of MP4s (default: off).',
    )
    p.add_argument(
        '--no_exclusive_lock',
        action='store_true',
        help='기본 배타 락(out_dir/.manip_play_rollouts.lock)을 쓰지 않음 (디버그용; 중복 실행 주의).',
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt = int(args.checkpoint_epoch)
    out_arg = str(args.out_dir).strip()
    _, env_nm = load_run_flags(run_dir)
    slug = slug_from_env(env_nm)
    out_dir = (
        Path(out_arg).resolve()
        if out_arg
        else run_dir / f'rollouts_manip_{slug}_ep{ckpt}'
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
                    idm_max_chunks=int(args.idm_max_chunks),
                    actor_max_chunks=int(args.actor_max_chunks),
                    fps=float(args.fps),
                    min_mp4_seconds=float(args.min_mp4_seconds),
                    show_goal_panel=bool(args.show_goal_panel),
                    idm_horizon=int(args.idm_horizon) if int(args.idm_horizon) > 0 else None,
                    idm_only=bool(args.idm_only),
                    subgoal_eval_num_samples=int(args.subgoal_eval_num_samples),
                    eval_ep_ix=int(args.eval_ep_ix),
                    until_success=bool(args.until_success),
                    max_eval_ep_ix=int(args.max_eval_ep_ix),
                    show_caption=bool(args.show_caption),
                )
            )
        summary_path = _write_rollout_task_summary_csv(out_dir, rows)
    print(f'done out_dir={out_dir} wrote {summary_path}')


__all__ = ['main']


if __name__ == '__main__':
    main()
