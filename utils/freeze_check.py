"""Helpers to verify frozen modules during actor SPI finetuning."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


def _leaves(tree: Any) -> list[np.ndarray]:
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def summarize_param_diff(before: Any, after: Any) -> dict[str, float]:
    """Return max/mean absolute parameter drift and changed leaf counts."""
    before_leaves = _leaves(before)
    after_leaves = _leaves(after)
    if len(before_leaves) != len(after_leaves):
        raise ValueError(f'pytree leaf count mismatch: {len(before_leaves)} vs {len(after_leaves)}')

    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    changed = 0
    non_empty = 0
    for before_arr, after_arr in zip(before_leaves, after_leaves):
        if before_arr.shape != after_arr.shape:
            raise ValueError(f'leaf shape mismatch: {before_arr.shape} vs {after_arr.shape}')
        if before_arr.size == 0:
            continue
        non_empty += 1
        diff = np.abs(after_arr - before_arr)
        leaf_max = float(np.max(diff))
        max_abs = max(max_abs, leaf_max)
        sum_abs += float(np.sum(diff))
        count += int(diff.size)
        if leaf_max > 0.0:
            changed += 1

    return {
        'max_abs': max_abs,
        'mean_abs': (sum_abs / count) if count else 0.0,
        'num_changed_leaves': float(changed),
        'num_leaves': float(non_empty),
    }


def assert_frozen(before: Any, after: Any, *, name: str, tol: float = 1e-6) -> dict[str, float]:
    """Raise if a frozen module moved more than ``tol``."""
    summary = summarize_param_diff(before, after)
    if summary['max_abs'] > float(tol):
        raise RuntimeError(
            f'[freeze_check] {name} params changed '
            f'(max_abs={summary["max_abs"]:.3e} > tol={tol:.1e}).'
        )
    return summary


def assert_trained(before: Any, after: Any, *, name: str, min_abs: float = 0.0) -> dict[str, float]:
    """Raise if a trainable module did not move."""
    summary = summarize_param_diff(before, after)
    if summary['max_abs'] <= float(min_abs):
        raise RuntimeError(
            f'[freeze_check] {name} params did not change '
            f'(max_abs={summary["max_abs"]:.3e}).'
        )
    return summary
