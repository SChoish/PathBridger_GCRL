"""Linear-SDE bridge dynamics schedule and math helpers.

The dynamics model uses a single exact linear-SDE bridge with state-time
consistent coefficients used by training and forward-bridge planning.

Theta schedule
--------------
The per-step OU rate ``theta_i`` comes from the *prefix-progress* schedule: it
calibrates the hard linear-SDE bridge marginal interpolation so that the
actor-visible prefix already covers a meaningful fraction of the subgoal
displacement. The target progress curve is

    c_i = (i / K) ** progress_alpha,    c_0 = 0,  c_K = 1.

For the hard bridge,  beta_i = sinh(Theta_i) / sinh(theta_total)  with
Theta_i = sum_{l<i} theta_l, hence

    Theta_i = asinh(c_i * sinh(theta_total)),
    theta_i = Theta_{i+1} - Theta_i.

``theta_total`` and ``progress_alpha`` are fixed to ``1.0`` and ``0.8`` in the
dynamics agent; they remain arguments here so the schedule stays testable.

Indexing conventions
--------------------
* Diffusion steps n = 0 (clean endpoint x_0) to N (noisy start x_T).
* Per-step arrays ``theta``, ``g2``, ``step_var``:
  shape (N,), index k corresponds to step n = k + 1.
* Linear-SDE forward state time uses i = N - n internally; schedule arrays are
  also exposed in diffusion n-indexing so agent code can index
  ``bridge_w[n]`` and ``bridge_var[n]`` directly.
"""

import jax.numpy as jnp


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

    See module docstring for the derivation. Result is clamped to be strictly
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
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
):
    """Precompute all linear-SDE dynamics schedule quantities.

    The theta schedule is the prefix-progress schedule: it calibrates the
    hard-bridge marginal interpolation so that the actor-visible prefix already
    reaches a meaningful fraction of the subgoal displacement.

    Args:
        N: number of diffusion steps.
        beta_min, beta_max: accepted for call-site compatibility; unused by the
            prefix-progress schedule.
        lambda_: linear-SDE diffusion coefficient.
        bridge_gamma_inv: endpoint precision offset used directly in bridge
            denominators. ``0.0`` is the hard endpoint bridge.
        theta_total: total cumulative rate ``Theta_K``.
        progress_alpha: exponent on ``i / K`` defining the desired marginal
            progress curve ``c_i = (i / K) ** progress_alpha``.
    """
    del beta_min, beta_max  # unused: kept only for call-site compatibility.
    gamma_inv = float(bridge_gamma_inv)
    if gamma_inv < 0.0:
        raise ValueError(f'bridge_gamma_inv must be >= 0, got {bridge_gamma_inv!r}.')

    theta_fwd = prefix_progress_theta_fwd(
        N,
        theta_total=theta_total,
        progress_alpha=progress_alpha,
    )
    g2_fwd = 2.0 * lambda_ ** 2 * theta_fwd
    step_var_fwd = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * theta_fwd))
    progress_target_fwd = desired_prefix_progress(N, progress_alpha=progress_alpha)

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

    out = dict(
        theta=theta, g2=g2, step_var=step_var,
        bar_theta=bar_theta, bar_sigma2=bar_sigma2,
        bar_theta_nN=bar_theta_nN, bar_sigma2_nN=bar_sigma2_nN,
        bar_sigma2_N=bar_sigma2_N,
        bridge_var=bridge_var, bridge_w=bridge_w,
        gamma_inv=gamma_inv_arr,
        theta_total=jnp.asarray(theta_total, dtype=jnp.float32),
        progress_alpha=jnp.asarray(progress_alpha, dtype=jnp.float32),
        progress_target_fwd=progress_target_fwd,
    )
    out.update(dynamics)
    return out


def forward_bridge_coefficients(
    K: int,
    *,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    lambda_: float,
    bridge_gamma_inv: float = 0.0,
    theta_total: float = 1.0,
    progress_alpha: float = 0.8,
):
    """Closed-form forward bridge marginals for the linear dynamics bridge.

    Uses the prefix-progress theta schedule. ``bridge_gamma_inv`` is the same
    finite-gamma denominator offset used by :func:`make_dynamics_schedule`.
    ``beta_min`` / ``beta_max`` are accepted for call-site compatibility but
    unused by the prefix-progress schedule.
    """
    del beta_min, beta_max  # unused: kept only for call-site compatibility.
    if K < 1:
        raise ValueError(f'K must be >= 1, got {K}.')
    K_int = int(K)
    gamma_inv = float(bridge_gamma_inv)
    if gamma_inv < 0.0:
        raise ValueError(f'bridge_gamma_inv must be >= 0, got {gamma_inv!r}.')

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
