"""Linear-SDE bridge dynamics schedule and math helpers.

The dynamics model uses a single exact linear-SDE bridge with state-time
consistent posterior coefficients used by training, reverse sampling, and
forward-bridge planning.

Indexing conventions
--------------------
* Diffusion steps n = 0 (clean endpoint x_0) to N (noisy start x_T).
* Per-step arrays ``theta``, ``g2``, ``step_var``:
  shape (N,), index k corresponds to step n = k + 1.
* Linear-SDE forward state time uses i = N - n internally; schedule arrays are
  also exposed in diffusion n-indexing so agent code can index
  ``bridge_w[n]`` and ``bridge_var[n]`` directly.
"""

import jax
import jax.numpy as jnp

from utils.theta_schedules import (
    canonical_theta_schedule,
    compute_progress_target_fwd,
    compute_theta_fwd,
    linear_beta_theta_diffusion,
    prefix_progress_theta_fwd,
    schedule_id,
)


def _linear_dynamics_arrays(theta_fwd, g2_fwd, step_var_fwd, gamma_inv=0.0):
    """Exact linear-SDE bridge arrays in forward state time.

    This uses the exact per-step discretization

        r_{i+1} = exp(theta_i) r_i + eta_i,
        eta_i ~ N(0, exp(2 theta_i) step_var_i I).

    The returned arrays are converted to diffusion index ``n`` via
    ``i = N - n`` so calls ``bridge_w[n]`` / ``bridge_var[n]`` keep the
    linear-SDE bridge.
    """
    theta_fwd = jnp.asarray(theta_fwd, dtype=jnp.float32)
    g2_fwd = jnp.asarray(g2_fwd, dtype=jnp.float32)
    step_var_fwd = jnp.asarray(step_var_fwd, dtype=jnp.float32)
    N = int(theta_fwd.shape[0])
    A = jnp.exp(theta_fwd)
    q2 = step_var_fwd * A ** 2

    # P_i = Var[r_i | r_0 = 0].
    Ps = [jnp.asarray(0.0, dtype=jnp.float32)]
    for i in range(N):
        Ps.append(A[i] ** 2 * Ps[-1] + q2[i])
    P = jnp.stack(Ps)  # (N+1,)

    # Phi_{i:K} = prod_{l=i}^{K-1} A_l.
    Phis = [None] * (N + 1)
    Phis[N] = jnp.asarray(1.0, dtype=jnp.float32)
    for i in range(N - 1, -1, -1):
        Phis[i] = A[i] * Phis[i + 1]
    Phi = jnp.stack(Phis)

    # Omega_{i:K} = Var[r_K | r_i] with r_i fixed.
    Oms = [None] * (N + 1)
    Oms[N] = jnp.asarray(0.0, dtype=jnp.float32)
    for i in range(N - 1, -1, -1):
        Oms[i] = q2[i] * Phi[i + 1] ** 2 + Oms[i + 1]
    Omega = jnp.stack(Oms)

    gamma_inv_arr = jnp.asarray(gamma_inv, dtype=jnp.float32)
    denom = jnp.maximum(P[-1] + gamma_inv_arr, 1e-12)

    beta = P * Phi / denom
    bridge_var = P * (Omega + gamma_inv_arr) / denom
    bridge_var = jnp.maximum(bridge_var, 0.0)

    # Hard endpoint bridge should pin endpoints exactly.
    if float(gamma_inv) == 0.0:
        beta = beta.at[0].set(0.0).at[-1].set(1.0)
        bridge_var = bridge_var.at[0].set(0.0).at[-1].set(0.0)

    # Diffusion index n corresponds to forward index i = N - n.
    return dict(
        theta_fwd=theta_fwd,
        g2_fwd=g2_fwd,
        step_var_fwd=step_var_fwd,
        theta_diffusion=theta_fwd[::-1],
        g2_diffusion=g2_fwd[::-1],
        step_var_diffusion=step_var_fwd[::-1],
        dynamics_A_fwd=A,
        dynamics_A=A[::-1],
        dynamics_q2_fwd=q2,
        dynamics_q2=q2[::-1],
        dynamics_P_fwd=P,
        dynamics_P=P[::-1],
        dynamics_phi_iK_fwd=Phi,
        dynamics_phi_iK=Phi[::-1],
        dynamics_omega_iK_fwd=Omega,
        dynamics_omega_iK=Omega[::-1],
        dynamics_beta_fwd=beta,
        dynamics_bridge_w=beta[::-1],
        dynamics_bridge_var_fwd=bridge_var,
        dynamics_bridge_var=bridge_var[::-1],
    )

def make_dynamics_schedule(
    N: int,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    lambda_: float = 1.0,
    bridge_gamma_inv: float = 0.0,
    theta_schedule: str = 'linear_beta',
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
):
    """Precompute all linear-SDE dynamics schedule quantities.

    Args:
        N: number of diffusion steps.
        beta_min, beta_max, lambda_: linear-beta OU schedule parameters
            (used only when ``theta_schedule == 'linear_beta'``).
        bridge_gamma_inv: endpoint precision offset used directly in bridge
            denominators. ``0.0`` is the hard endpoint bridge.
        theta_schedule: ``'linear_beta'`` (default diffusion-style schedule) or ``'prefix_progress'``
            (calibrates the hard-bridge marginal interpolation so that the
            actor-visible prefix already reaches a meaningful fraction of the
            subgoal displacement).
        theta_total: total cumulative rate ``Theta_K`` for the prefix-progress
            schedule. Ignored when ``theta_schedule == 'linear_beta'``.
        progress_alpha: exponent on ``i / K`` defining the desired marginal
            progress curve ``c_i = (i / K) ** progress_alpha`` for the
            prefix-progress schedule. Ignored otherwise.
    """
    gamma_inv = float(bridge_gamma_inv)
    if gamma_inv < 0.0:
        raise ValueError(f'bridge_gamma_inv must be >= 0, got {bridge_gamma_inv!r}.')

    schedule_mode = canonical_theta_schedule(theta_schedule)

    theta_fwd = compute_theta_fwd(
        N,
        theta_schedule=schedule_mode,
        beta_min=beta_min,
        beta_max=beta_max,
        theta_total=theta_total,
        progress_alpha=progress_alpha,
    )
    g2_fwd = 2.0 * lambda_ ** 2 * theta_fwd
    step_var_fwd = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * theta_fwd))
    progress_target_fwd = compute_progress_target_fwd(
        N, theta_schedule=schedule_mode, progress_alpha=progress_alpha,
    )

    gamma_inv_arr = jnp.asarray(gamma_inv, dtype=jnp.float32)

    dynamics = _linear_dynamics_arrays(theta_fwd, g2_fwd, step_var_fwd, gamma_inv=gamma_inv)
    # Arrays indexed by diffusion n correspond to forward state-time step i = N - n.
    theta = dynamics['theta_diffusion']
    g2 = dynamics['g2_diffusion']
    step_var = dynamics['step_var_diffusion']
    bridge_w = dynamics['dynamics_bridge_w']
    bridge_var = dynamics['dynamics_bridge_var']

    # Diagnostic cumulative arrays in diffusion index n.
    bar_theta = jnp.concatenate([jnp.zeros(1), jnp.cumsum(theta)])  # (N+1,)
    bar_sigma2 = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * bar_theta))  # (N+1,)
    bar_theta_nN = bar_theta[-1] - bar_theta  # (N+1,)
    bar_sigma2_nN = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * bar_theta_nN))  # (N+1,)
    bar_sigma2_N = bar_sigma2[-1]  # scalar

    sched_id = jnp.asarray(schedule_id(schedule_mode), dtype=jnp.float32)

    out = dict(
        theta=theta, g2=g2, step_var=step_var,
        bar_theta=bar_theta, bar_sigma2=bar_sigma2,
        bar_theta_nN=bar_theta_nN, bar_sigma2_nN=bar_sigma2_nN,
        bar_sigma2_N=bar_sigma2_N,
        bridge_var=bridge_var, bridge_w=bridge_w,
        gamma_inv=gamma_inv_arr,
        theta_schedule_id=sched_id,
        theta_total=jnp.asarray(theta_total, dtype=jnp.float32),
        progress_alpha=jnp.asarray(progress_alpha, dtype=jnp.float32),
        progress_target_fwd=progress_target_fwd,
    )
    out.update(dynamics)
    return out


def bridge_sample(x_0, x_T, n, schedule, rng):
    """Sample x_n from the forward bridge q(x_n | x_0, x_T).

    Args:
        x_0: Target endpoint, shape (B, D).
        x_T: Current state, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N-1}.
        schedule: Output of ``make_dynamics_schedule``.
        rng: JAX PRNG key.

    Returns:
        x_n of shape (B, D).
    """
    w = schedule['bridge_w'][n][..., None]
    var = schedule['bridge_var'][n][..., None]
    mean = w * x_0 + (1.0 - w) * x_T
    return mean + jnp.sqrt(jnp.maximum(var, 1e-12)) * jax.random.normal(rng, x_0.shape)


def posterior_moments(x_n, x_0, x_T, n, schedule):
    """Analytic linear-dynamics posterior moments for ``x_{n-1} | x_n, x_0, x_T``.

    Combines the one-step reverse ``N(mu_back, sigma_back^2)`` with the bridge
    marginal at step ``n-1`` ``N(bridge_mean_{n-1}, bridge_var_{n-1})`` via Bayes.

    Valid for ``n in {1, ..., N}``. At ``n = 1`` the bridge variance at step 0
    is zero, so the posterior mean collapses to ``x_0`` and the posterior
    variance vanishes (the residual scale automatically zeroes at the
    terminal endpoint in ``exact_residual_model_mean``).

    Args:
        x_n: Current bridge state, shape (B, D).
        x_0: Target endpoint, shape (B, D).
        x_T: Current observation, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N}.
        schedule: Output of ``make_dynamics_schedule``.

    Returns:
        mean: shape (B, D).
        var:  shape (B, 1) -- one-step posterior variance ``Var[x_{n-1}|...]``.
    """
    k = n - 1  # 0-based index into (N,) arrays
    theta_n = schedule['theta'][k][..., None]
    svar = schedule['step_var'][k][..., None]

    exp_t = jnp.exp(theta_n)
    mu_back = exp_t * x_n - (exp_t - 1.0) * x_T
    sig_back2 = svar * exp_t ** 2

    nm1 = n - 1  # step n-1 for (N+1,) arrays
    bvar = schedule['bridge_var'][nm1][..., None]
    bw = schedule['bridge_w'][nm1][..., None]
    bmean = bw * x_0 + (1.0 - bw) * x_T

    denom = sig_back2 + bvar + 1e-12
    mean = (bvar * mu_back + sig_back2 * bmean) / denom
    var = jnp.maximum((sig_back2 * bvar) / denom, 0.0)
    return mean, var


def posterior_mean(x_n, x_0, x_T, n, schedule):
    """Analytic linear-dynamics posterior mean  ``E[x_{n-1} | x_n, x_0, x_T]``.

    Thin wrapper around :func:`posterior_moments` that drops the variance.
    Numerical output is bit-identical to the previous implementation.
    """
    mean, _ = posterior_moments(x_n, x_0, x_T, n, schedule)
    return mean


def exact_residual_model_mean(
    x_n,
    x_0,
    x_T,
    eps_pred,
    n,
    schedule,
    residual_scale: float = 1.0,
):
    """Exact bridge posterior mean plus a learned data residual.

    ``mu = posterior_mean(x_n, x_0, x_T, n) +
            residual_scale * sqrt(max(post_var, 0)) * eps_pred``.

    This is the ``exact_residual`` ablation: the analytic one-step bridge
    posterior is used as the *base* transition, and only a data-dependent
    residual (from ``ResidualNet``) is learned. Valid for all ``n in {1,..,N}``;
    no boundary special case is required because the exact moments already
    handle the terminal step (and ``post_var`` -> 0 at ``n=1``, which makes
    the residual vanish automatically).

    Returns:
        mu: shape (B, D).
        mu_base: shape (B, D), exact posterior mean.
        post_var: shape (B, 1), exact posterior variance.
    """
    mu_base, post_var = posterior_moments(x_n, x_0, x_T, n, schedule)
    scale = jnp.asarray(residual_scale, dtype=jnp.float32)
    std = scale * jnp.sqrt(jnp.maximum(post_var, 0.0))
    mu = mu_base + std * eps_pred
    return mu, mu_base, post_var

def reverse_std(n, schedule):
    """Return sqrt(g_n^2) for reverse sampling.

    Args:
        n: Step indices, shape (B,), values in {1, ..., N}.
        schedule: Output of ``make_dynamics_schedule``.

    Returns:
        std of shape (B, 1).
    """
    k = n - 1
    g2_n = schedule['g2'][k][..., None]
    return jnp.sqrt(jnp.maximum(g2_n, 1e-12))


def sample_from_reverse_mean(mu, n, schedule, rng, noise_scale=1.0):
    """Sample x_{n-1} ~ N(mu, noise_scale^2 * g_n^2 I).

    Args:
        mu: Reverse-step mean, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N}.
        schedule: Output of ``make_dynamics_schedule``.
        rng: JAX PRNG key.
        noise_scale: Optional temperature multiplier (scalar or 0-dim array).

    Returns:
        Sampled x_{n-1}, shape (B, D).
    """
    std = reverse_std(n, schedule)
    noise = jax.random.normal(rng, mu.shape)
    ns = jnp.asarray(noise_scale, dtype=jnp.float32)
    return mu + ns * std * noise



def forward_bridge_coefficients(
    K: int,
    *,
    beta_min: float,
    beta_max: float,
    lambda_: float,
    bridge_gamma_inv: float = 0.0,
    theta_schedule: str = 'linear_beta',
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
):
    """Closed-form forward bridge marginals for the linear dynamics bridge.

    When ``theta_schedule == 'linear_beta'`` the ascending raw array
    ``beta_min/K + (beta_max - beta_min) * (k+1) / K^2``  is passed *directly*
    as ``theta_fwd`` to ``_linear_dynamics_arrays``. ``bridge_gamma_inv`` is the
    same finite-gamma denominator offset used by :func:`make_dynamics_schedule`.
    """
    if K < 1:
        raise ValueError(f'K must be >= 1, got {K}.')
    K_int = int(K)
    gamma_inv = float(bridge_gamma_inv)
    if gamma_inv < 0.0:
        raise ValueError(f'bridge_gamma_inv must be >= 0, got {gamma_inv!r}.')
    mode = canonical_theta_schedule(theta_schedule)

    if mode == 'linear_beta':
        theta_fwd = linear_beta_theta_diffusion(K_int, beta_min, beta_max)
    else:
        theta_fwd = prefix_progress_theta_fwd(
            K_int,
            theta_total=theta_total,
            progress_alpha=progress_alpha,
        )

    g2_fwd = 2.0 * float(lambda_) ** 2 * theta_fwd
    step_var_fwd = float(lambda_) ** 2 * (1.0 - jnp.exp(-2.0 * theta_fwd))
    arr = _linear_dynamics_arrays(theta_fwd, g2_fwd, step_var_fwd, gamma_inv=gamma_inv)
    b = arr['dynamics_beta_fwd']
    std = jnp.sqrt(jnp.maximum(arr['dynamics_bridge_var_fwd'], 0.0))
    a = 1.0 - b
    a = a.at[0].set(1.0).at[-1].set(0.0)
    b = b.at[0].set(0.0).at[-1].set(1.0)
    std = std.at[0].set(0.0).at[-1].set(0.0)
    return a, b, std
