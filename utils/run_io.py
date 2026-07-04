"""Shared checkpoint, parsing, and evaluation-result I/O helpers for production runs."""

from __future__ import annotations

import csv
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flax


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


# --- parsing ------------------------------------------------------------------


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse ``"1,2,3"`` → ``(1, 2, 3)``. Returns ``()`` for empty strings."""
    items = [item.strip() for item in str(text).split(',') if item.strip()]
    return tuple(int(item) for item in items)


# --- evaluation result files --------------------------------------------------

def eval_result_path(
    run_dir: Path | str,
    *,
    epoch: int,
    eval_n: int,
    subgoal_temperature: float | None = None,
) -> Path:
    if subgoal_temperature is None:
        return Path(run_dir) / 'eval_results' / f'epoch{int(epoch)}_n{int(eval_n)}.json'
    temp_tag = _temp_tag(float(subgoal_temperature))
    return Path(run_dir) / 'eval_results' / f'epoch{int(epoch)}_t{temp_tag}_n{int(eval_n)}.json'


def _temp_tag(temperature: float) -> str:
    if abs(temperature - round(temperature)) < 1e-9:
        return str(int(round(temperature)))
    return format(temperature, 'g').replace('.', 'p')


def save_eval_results(
    run_dir: Path | str,
    *,
    epoch: int,
    subgoal_eval_num_samples: int,
    task_ids: tuple[int, ...] | list[int],
    episodes_per_task: int,
    metrics: dict[str, Any],
    fg: dict[str, Any],
    root: dict[str, Any],
    subgoal_temperature: float | None = None,
) -> Path:
    run_dir = Path(run_dir)
    eval_n = int(subgoal_eval_num_samples)
    def _task_rates(prefix: str) -> dict[str, float]:
        return {
            str(tid): float(metrics[f'{prefix}/task_{tid}/success_rate'])
            for tid in task_ids
            if f'{prefix}/task_{tid}/success_rate' in metrics
        }

    idm_tasks = _task_rates('eval_idm')
    actor_tasks = _task_rates('eval')
    four_way_prefixes = (
        'eval_flow_idm',
        'eval_flow_actor',
    )
    four_way_means = {
        prefix: float(metrics[f'{prefix}/success_rate_mean'])
        for prefix in four_way_prefixes
        if f'{prefix}/success_rate_mean' in metrics
    }
    four_way_tasks = {
        prefix: _task_rates(prefix)
        for prefix in four_way_prefixes
        if any(f'{prefix}/task_{tid}/success_rate' in metrics for tid in task_ids)
    }
    dyn = root.get('dynamics', {})
    record: dict[str, Any] = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'run_dir': str(run_dir.resolve()),
        'run_group': str(fg.get('run_group', '')),
        'env_name': str(fg.get('env_name', '')),
        'epoch': int(epoch),
        'subgoal_eval_num_samples': eval_n,
        'subgoal_num_samples_train': int(dyn.get('subgoal_num_samples', 0)),
        'subgoal_value_gap_scale': float(dyn.get('subgoal_value_gap_scale', 0.0)),
        'subgoal_value_weight_max': float(dyn.get('subgoal_value_weight_max', 0.0)),
        'subgoal_temperature': float(
            subgoal_temperature if subgoal_temperature is not None else dyn.get('subgoal_temperature', 1.0)
        ),
        'eval_episodes_per_task': int(episodes_per_task),
        'eval_budget': 'env_max_episode_steps',
        'eval_task_ids': [int(t) for t in task_ids],
        'idm_success_rate_mean': float(metrics.get('eval_idm/success_rate_mean', float('nan'))),
        'actor_success_rate_mean': float(metrics.get('eval/success_rate_mean', float('nan'))),
        'idm_task_success_rates': idm_tasks,
        'actor_task_success_rates': actor_tasks,
        'four_way_success_rate_means': four_way_means,
        'four_way_task_success_rates': four_way_tasks,
    }
    out_dir = run_dir / 'eval_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_result_path(
        run_dir,
        epoch=epoch,
        eval_n=eval_n,
        subgoal_temperature=subgoal_temperature,
    )
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
        f.write('\n')

    csv_path = out_dir / 'all.csv'
    row = {
        'timestamp': record['timestamp'],
        'epoch': record['epoch'],
        'eval_n': eval_n,
        'idm_mean': record['idm_success_rate_mean'],
        'actor_mean': record['actor_success_rate_mean'],
        'idm_tasks': ','.join(f'{k}:{v:.4f}' for k, v in sorted(idm_tasks.items())),
        'actor_tasks': ','.join(f'{k}:{v:.4f}' for k, v in sorted(actor_tasks.items())),
    }
    for prefix in four_way_prefixes:
        row[f'{prefix}_mean'] = four_way_means.get(prefix, '')
        row[f'{prefix}_tasks'] = ','.join(
            f'{k}:{v:.4f}' for k, v in sorted(four_way_tasks.get(prefix, {}).items())
        )
    write_header = not csv_path.is_file()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return json_path


__all__ = [
    'list_checkpoint_suffixes',
    'pick_epoch',
    'load_checkpoint_pkl',
    'resolve_dynamics_checkpoint_dir',
    'resolve_critic_checkpoint_dir',
    'resolve_actor_checkpoint_dir',
    'parse_int_list',
    'eval_result_path',
    'save_eval_results',
]
