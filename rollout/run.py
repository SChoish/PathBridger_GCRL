#!/usr/bin/env python3
"""Unified rollout launcher.

Picks the right rollout path from ``flags.json`` (manip play vs maze) and runs
IDM and/or actor episodes through the shared :func:`rollout.episode_runner.run_chunked_episode`.
``--mode subgoal`` (state-space open-loop + xy plot) is maze-only; manip play
runs IDM and actor only since cube/puzzle have no state-space plot.

Examples::

    PYTHONPATH=. python -m rollout.run --run_dir=runs/... --mode all --task_ids=1,2,3
    PYTHONPATH=. python -m rollout.run --run_dir=runs/... --mode actor --task_ids=1 --mujoco_gl=egl
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Iterator, Sequence

from rollout.common import classify_rollout_env, slug_from_env
from utils.run_io import load_run_flags, parse_int_list

_MODES = ('subgoal', 'idm', 'actor')


@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    old = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


def _run_module_main(module_name: str, argv: Sequence[str]) -> None:
    module = __import__(module_name, fromlist=['main'])
    with _temporary_argv(argv):
        module.main()


def _selected_modes(mode: str) -> tuple[str, ...]:
    m = str(mode).strip().lower()
    if m == 'all':
        return _MODES
    if m not in _MODES:
        raise ValueError(f'Unknown rollout mode {mode!r}; expected all or one of {_MODES}')
    return (m,)


def _append_bool_flag(argv: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        argv.append(flag)


def _run_manip(args: argparse.Namespace, passthrough: Sequence[str]) -> None:
    from rollout import manip_play_rollouts

    argv = [
        'rollout.manip_play_rollouts',
        f'--run_dir={args.run_dir}',
        f'--checkpoint_epoch={int(args.checkpoint_epoch)}',
        f'--task_ids={args.task_ids}',
        f'--mujoco_gl={args.mujoco_gl or "osmesa"}',
        f'--seed={int(args.seed)}',
        f'--fps={float(args.fps)}',
    ]
    if args.out_dir:
        argv.append(f'--out_dir={args.out_dir}')
    if args.max_chunks >= 0:
        argv.append(f'--idm_max_chunks={int(args.max_chunks)}')
        argv.append(f'--actor_max_chunks={int(args.max_chunks)}')
    argv.extend(passthrough)
    with _temporary_argv(argv):
        manip_play_rollouts.main()


def _run_maze_like(args: argparse.Namespace, env_name: str, passthrough: Sequence[str]) -> None:
    task_ids = parse_int_list(str(args.task_ids))
    if not task_ids:
        raise SystemExit('empty --task_ids')

    ckpt = int(args.checkpoint_epoch)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else Path(args.run_dir).resolve() / f'rollouts_{slug_from_env(env_name)}_ep{ckpt}'
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        task_dir = out_dir / f'task{int(task_id)}'
        task_dir.mkdir(parents=True, exist_ok=True)
        for mode in _selected_modes(args.mode):
            if mode == 'subgoal':
                argv = [
                    'rollout.subgoal',
                    f'--run_dir={args.run_dir}',
                    f'--checkpoint_epoch={ckpt}',
                    f'--task_id={int(task_id)}',
                    f'--max_steps={int(args.max_steps)}',
                    f'--out_path={task_dir / "subgoal.png"}',
                    f'--fps={float(args.fps)}',
                ]
                _append_bool_flag(argv, bool(args.no_mp4), '--no_mp4')
                argv.extend(passthrough)
                _run_module_main('rollout.subgoal', argv)
            elif mode == 'idm':
                argv = [
                    'rollout.idm',
                    f'--run_dir={args.run_dir}',
                    f'--checkpoint_epoch={ckpt}',
                    f'--task_id={int(task_id)}',
                    f'--max_steps={int(args.max_steps)}',
                    f'--out_path={task_dir / "idm.png"}',
                    f'--out_mp4={task_dir / "idm.mp4"}',
                    f'--fps={float(args.fps)}',
                ]
                if args.mujoco_gl:
                    argv.append(f'--mujoco_gl={args.mujoco_gl}')
                if int(args.action_chunk_horizon) >= 0:
                    argv.append(f'--action_chunk_horizon={int(args.action_chunk_horizon)}')
                _append_bool_flag(argv, bool(args.no_mp4), '--no_mp4')
                argv.extend(passthrough)
                _run_module_main('rollout.idm', argv)
            elif mode == 'actor':
                argv = [
                    'rollout.actor',
                    f'--run_dir={args.run_dir}',
                    f'--checkpoint_epoch={ckpt}',
                    f'--task_id={int(task_id)}',
                    f'--out_path={task_dir / "actor.png"}',
                    f'--out_mp4={task_dir / "actor.mp4"}',
                    f'--fps={float(args.fps)}',
                ]
                if args.mujoco_gl:
                    argv.append(f'--mujoco_gl={args.mujoco_gl}')
                _append_bool_flag(argv, bool(args.no_mp4), '--no_mp4')
                argv.extend(passthrough)
                _run_module_main('rollout.actor', argv)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--checkpoint_epoch', type=int, default=1000)
    p.add_argument('--mode', type=str, choices=('all', *_MODES), default='all')
    p.add_argument('--task_ids', type=str, default='1,2,3,4,5')
    p.add_argument('--out_dir', type=str, default='')
    p.add_argument('--mujoco_gl', type=str, default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument('--max_steps', type=int, default=1000)
    p.add_argument(
        '--max_chunks',
        type=int,
        default=-1,
        help='Manip-only override for IDM/actor outer replans; <0 lets the env TimeLimit decide '
        '(maze actor always derives the budget from the env).',
    )
    p.add_argument(
        '--action_chunk_horizon',
        type=int,
        default=-1,
        help='Optional IDM action chunk horizon override; -1 keeps the rollout.idm default.',
    )
    p.add_argument('--no_mp4', action='store_true')
    args, passthrough = p.parse_known_args()

    _cfg, env_name = load_run_flags(Path(args.run_dir).resolve())
    family = classify_rollout_env(env_name)
    if family == 'manip_play':
        if args.mode not in ('all', 'idm', 'actor'):
            raise SystemExit(
                'Manip play rollout supports --mode=all|idm|actor (state-space subgoal plot is not produced).'
            )
        _run_manip(args, passthrough)
    else:
        _run_maze_like(args, env_name, passthrough)


if __name__ == '__main__':
    main()
