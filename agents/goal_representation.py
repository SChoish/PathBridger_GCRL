"""Goal representation helpers shared by goal-conditioned networks.

``phi`` / ``auto`` / ``goal_phi`` are defined only for the OGBench families used by
the production configs (distinguished by ``env_name``):

  - **puzzle**: binary pressed state per button (compact obs one-hot channel).
  - **cube**: concatenated ``(x, y, z)`` per cube — same layout as
    ``CubeEnv.compute_oracle_observation`` (scaled center-relative positions).
  - **antmaze** / **humanoidmaze**: planar goal ``(x, y)`` — indices ``(0, 1)``.

Any other ``env_name`` under ``phi`` raises ``ValueError``. ``phi_goal_obs_indices``
in YAML is optional: when omitted or empty, training fills it from ``env_name`` and
the env observation dimension (maze → ``(0, 1)``; ManipSpace puzzle/cube layouts
from ``obs_dim``). At runtime, ``goal_representation(..., 'phi', ...)`` uses the
configured tuple, falling back to the same inference when needed.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

_MANIP_ARM_JOINT_DIM = 6
_MANIP_HEAD_DIM = 2 * _MANIP_ARM_JOINT_DIM + 3 + 1 + 1 + 1 + 1
_MANIP_CUBE_STRIDE = 3 + 4 + 1 + 1
_MANIP_BUTTON_STRIDE = 2 + 1 + 1

# Planar achieved-goal channels in OGBench maze goal observations.
_MAZE_GOAL_XY_INDICES = (0, 1)


def _env_goal_phi_kind(env_name: str | None) -> str:
    """Return canonical phi family."""

    if env_name is None or not str(env_name).strip():
        raise ValueError(
            "goal_representation='phi' requires a non-empty env_name "
            "(e.g. FLAGS.env_name from yaml env_name)."
        )
    name = str(env_name).lower()
    if 'humanoidmaze' in name:
        return 'humanoidmaze'
    if 'antmaze' in name:
        return 'antmaze'
    if 'puzzle' in name:
        return 'puzzle'
    if 'cube' in name:
        return 'cube'
    raise ValueError(
        f"goal_representation='phi': unsupported env_name={env_name!r}. "
        "Expected a name containing one of: 'humanoidmaze', 'antmaze', 'puzzle', "
        "or 'cube'."
    )


def manip_cube_pos_indices(obs_dim: int) -> tuple[int, ...]:
    """Return compact-obs indices for every cube's ``(x,y,z)`` (oracle-aligned)."""

    dim = int(obs_dim)
    rem = dim - _MANIP_HEAD_DIM
    if rem < _MANIP_CUBE_STRIDE or rem % _MANIP_CUBE_STRIDE != 0:
        return ()
    idxs: list[int] = []
    for start in range(_MANIP_HEAD_DIM, dim, _MANIP_CUBE_STRIDE):
        idxs.extend((start, start + 1, start + 2))
    return tuple(idxs)


def manip_button_state_indices(obs_dim: int) -> tuple[int, ...]:
    """Return per-button binary-state channels for ManipSpace puzzle obs."""

    dim = int(obs_dim)
    rem = dim - _MANIP_HEAD_DIM
    if rem <= 0 or rem % _MANIP_BUTTON_STRIDE != 0:
        return ()
    n_buttons = rem // _MANIP_BUTTON_STRIDE
    idxs: list[int] = []
    for i in range(n_buttons):
        start = _MANIP_HEAD_DIM + i * _MANIP_BUTTON_STRIDE
        idxs.append(start + 1)
    return tuple(idxs)


def infer_phi_goal_obs_indices(env_name: str | None, obs_dim: int | None = None) -> tuple[int, ...]:
    """Default ``phi_goal_obs_indices`` when YAML omits the field.

    Maze envs always return ``(0, 1)`` (planar goal). Puzzle/cube need ``obs_dim``
    to validate the compact layout.
    """

    if env_name is None or not str(env_name).strip():
        return ()
    try:
        kind = _env_goal_phi_kind(env_name)
    except ValueError:
        return ()
    if kind in ('antmaze', 'humanoidmaze'):
        return _MAZE_GOAL_XY_INDICES
    if obs_dim is None:
        return ()
    dim = int(obs_dim)
    if kind == 'puzzle':
        return manip_button_state_indices(dim)
    if kind == 'cube':
        return manip_cube_pos_indices(dim)
    raise AssertionError(f'unreachable phi kind={kind!r}')


def normalize_phi_goal_obs_indices(raw: object) -> tuple[int, ...]:
    """Parse YAML / CLI values into a tuple of non-negative ints (may be empty)."""

    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    raise TypeError(f'phi_goal_obs_indices must be a list/tuple of ints, got {type(raw).__name__}')


def assert_phi_goal_obs_indices(
    obs_dim: int,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...],
    *,
    where: str,
    env_name: str | None = None,
) -> None:
    """Validate that ``obs_dim`` and indices are compatible with ``phi``."""

    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        return
    dim = int(obs_dim)
    idxs = normalize_phi_goal_obs_indices(phi_goal_obs_indices)
    if not idxs:
        idxs = infer_phi_goal_obs_indices(env_name, dim)
    if not idxs:
        try:
            _env_goal_phi_kind(env_name)
        except ValueError as e:
            raise ValueError(f'{where}: {e}') from e
        raise ValueError(
            f'{where}: goal_representation={mode_l!r} for env={env_name!r}: '
            f'could not infer phi_goal_obs_indices for obs_dim={dim}.'
        )
    bad = [idx for idx in idxs if int(idx) < 0 or int(idx) >= dim]
    if bad:
        raise ValueError(f'{where}: phi_goal_obs_indices out of bounds for obs_dim={dim}: {bad}')


def goal_representation(
    goals: jnp.ndarray | None,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...] = (),
    *,
    env_name: str | None = None,
) -> jnp.ndarray | None:
    """Map a full goal state to the configured goal representation."""

    if goals is None:
        return None
    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return goals
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        raise ValueError(
            f"Unknown goal_representation={mode!r}; expected 'full' or 'phi'."
        )

    obs_dim = int(goals.shape[-1])
    idxs = normalize_phi_goal_obs_indices(phi_goal_obs_indices)
    if not idxs:
        idxs = infer_phi_goal_obs_indices(env_name, obs_dim)
    if not idxs:
        _env_goal_phi_kind(env_name)
        raise ValueError(
            f"goal_representation='phi' for env={env_name!r}: could not infer "
            f'phi_goal_obs_indices for obs_dim={obs_dim}.'
        )
    return jnp.take(goals, jnp.asarray(idxs, dtype=jnp.int32), axis=-1)
