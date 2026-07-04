"""Theta schedules for the linear-SDE dynamics bridge.

A *theta schedule* defines the per-step OU rate ``theta_i`` of the forward
linear-SDE. Each schedule here is a pure function returning ``theta_fwd``
indexed by forward state time ``i = 0, ..., N - 1``; downstream bridge math
(``utils.dynamics._linear_dynamics_arrays``) consumes that array and
constructs all derived quantities.

Currently supported schedules
-----------------------------
* ``linear_beta`` (default)
    Diffusion-style schedule

        theta_diffusion_n = beta_min / N + (beta_max - beta_min) * n / N^2,  n = 1..N

    in diffusion-time index ``n`` (large ``n`` = noisy end). The forward
    state-time array is the reverse of this diffusion-indexed array.

* ``prefix_progress``
    Calibrates the hard linear-SDE bridge marginal interpolation so that the
    actor-visible prefix already covers a meaningful fraction of the subgoal
    displacement. The target progress curve is

        c_i = (i / K) ** progress_alpha,    c_0 = 0,  c_K = 1.

    For the hard bridge,  beta_i = sinh(Theta_i) / sinh(theta_total)  with
    Theta_i = sum_{l<i} theta_l, hence

        Theta_i = asinh(c_i * sinh(theta_total)),
        theta_i = Theta_{i+1} - Theta_i.

The dispatcher :func:`compute_theta_fwd` is the single entry point used by
:func:`utils.dynamics.make_dynamics_schedule` and
:func:`utils.dynamics.forward_bridge_coefficients`.
"""

from __future__ import annotations

import jax.numpy as jnp


VALID_THETA_SCHEDULES = ('linear_beta', 'prefix_progress')


def canonical_theta_schedule(theta_schedule: str) -> str:
    """Return the canonical lower-cased schedule name; raise on unknown."""
    mode = str(theta_schedule).lower()
    if mode not in VALID_THETA_SCHEDULES:
        raise ValueError(
            f'theta_schedule must be one of {VALID_THETA_SCHEDULES}, got {theta_schedule!r}.'
        )
    return mode


def schedule_id(theta_schedule: str) -> float:
    """Float metric id for logging / metadata. Linear-beta = 0, prefix-progress = 1."""
    return 1.0 if canonical_theta_schedule(theta_schedule) == 'prefix_progress' else 0.0


# ---------------------------------------------------------------------------
# linear_beta
# ---------------------------------------------------------------------------


def linear_beta_theta_diffusion(N: int, beta_min: float, beta_max: float) -> jnp.ndarray:
    """Diffusion-indexed linear-beta theta, shape ``(N,)``.

    Indexed by diffusion-time ``n = 1, ..., N`` (large ``n`` = noisy
    start). The forward state-time array is :func:`linear_beta_theta_fwd`.
    """
    steps = jnp.arange(1, N + 1, dtype=jnp.float32)
    return beta_min / float(N) + (beta_max - beta_min) * steps / float(N * N)


def linear_beta_theta_fwd(N: int, beta_min: float, beta_max: float) -> jnp.ndarray:
    """Forward state-time linear-beta theta, shape ``(N,)``.

    Equal to :func:`linear_beta_theta_diffusion` reversed: ``theta_fwd[i] =
    theta_diffusion[N - 1 - i]``.
    """
    return linear_beta_theta_diffusion(N, beta_min, beta_max)[::-1]


# ---------------------------------------------------------------------------
# prefix_progress
# ---------------------------------------------------------------------------


def desired_prefix_progress(N: int, progress_alpha: float = 0.8) -> jnp.ndarray:
    """Desired hard-bridge marginal progress ``c_i = (i / N) ** alpha``.

    Returns a ``(N + 1,)`` array indexed by forward state time ``i``, with
    ``c_0 = 0`` and ``c_N = 1`` pinned exactly.
    """
    if N < 1:
        raise ValueError(f'N must be >= 1, got {N}.')
    if float(progress_alpha) <= 0.0:
        raise ValueError(f'progress_alpha must be > 0, got {progress_alpha!r}.')
    idx = jnp.arange(N + 1, dtype=jnp.float32)
    c = (idx / float(N)) ** float(progress_alpha)
    return c.at[0].set(0.0).at[-1].set(1.0)


def prefix_progress_theta_fwd(
    N: int,
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
) -> jnp.ndarray:
    """Forward state-time prefix-calibrated theta, shape ``(N,)``.

    See module docstring for derivation. Result is clamped to be strictly
    positive (analytically it is, but ``jnp.maximum`` guards floating-point
    underflow at very small alpha / large N).
    """
    if N < 1:
        raise ValueError(f'N must be >= 1, got {N}.')
    if float(theta_total) <= 0.0:
        raise ValueError(f'theta_total must be > 0, got {theta_total!r}.')

    c = desired_prefix_progress(N, progress_alpha=progress_alpha)
    total = jnp.asarray(theta_total, dtype=jnp.float32)
    Theta = jnp.arcsinh(c * jnp.sinh(total))
    theta_fwd = Theta[1:] - Theta[:-1]
    return jnp.maximum(theta_fwd, 1e-12)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def compute_theta_fwd(
    N: int,
    *,
    theta_schedule: str = 'linear_beta',
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
) -> jnp.ndarray:
    """Single entry point: return ``theta_fwd`` for the requested schedule.

    Schedule-specific arguments are ignored when not used by the selected
    schedule (e.g., ``theta_total`` / ``progress_alpha`` are ignored for
    ``linear_beta``).
    """
    mode = canonical_theta_schedule(theta_schedule)
    if mode == 'linear_beta':
        return linear_beta_theta_fwd(N, beta_min, beta_max)
    return prefix_progress_theta_fwd(
        N,
        theta_total=theta_total,
        progress_alpha=progress_alpha,
    )


def compute_progress_target_fwd(
    N: int,
    *,
    theta_schedule: str = 'linear_beta',
    progress_alpha: float = 0.8,
) -> jnp.ndarray:
    """Desired progress curve ``c_i`` for the schedule, shape ``(N + 1,)``.

    Returns ``NaN``-filled for ``linear_beta`` (no calibration target),
    ``(i / N) ** progress_alpha`` for ``prefix_progress``.
    """
    mode = canonical_theta_schedule(theta_schedule)
    if mode == 'linear_beta':
        return jnp.full((N + 1,), jnp.nan, dtype=jnp.float32)
    return desired_prefix_progress(N, progress_alpha=progress_alpha)


__all__ = [
    'VALID_THETA_SCHEDULES',
    'canonical_theta_schedule',
    'schedule_id',
    'linear_beta_theta_diffusion',
    'linear_beta_theta_fwd',
    'desired_prefix_progress',
    'prefix_progress_theta_fwd',
    'compute_theta_fwd',
    'compute_progress_target_fwd',
]
