"""Goal representation helpers shared by goal-conditioned networks.

``phi`` / ``auto`` / ``goal_phi`` are defined only for these OGBench families
(distinguished by ``env_name``):

  - **puzzle**: binary pressed state per button (compact obs one-hot channel).
  - **cube**: concatenated ``(x, y, z)`` per cube — same layout as
    ``CubeEnv.compute_oracle_observation`` (scaled center-relative positions).
  - **scene**: same order as ``SceneEnv.compute_oracle_observation``: all cubes'
    scaled ``(x,y,z)``, then one scalar **button state** per button (from argmax
    of the one-hot block in compact obs), then ``drawer_pos * drawer_scaler`` and
    ``window_pos * window_scaler`` (the first channel of each tail pair in
    ``compute_observation``). Cube / button counts and one-hot width are inferred
    from ``obs_dim`` (must factor uniquely).
  - **antmaze** / **humanoidmaze**: planar goal ``(x, y)`` — indices ``(0, 1)``.

Any other ``env_name`` under ``phi`` raises ``ValueError``. ``phi_goal_obs_indices``
in YAML is optional: when omitted or empty, training fills it from ``env_name`` and
the env observation dimension (maze → ``(0, 1)``; ManipSpace puzzle/cube layouts
from ``obs_dim``). At runtime, ``goal_representation(..., 'phi', ...)`` still
derives channels from ``env_name`` and goal shape; the stored tuple is mainly for
config parity.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

_MANIP_ARM_JOINT_DIM = 6
_MANIP_HEAD_DIM = 2 * _MANIP_ARM_JOINT_DIM + 3 + 1 + 1 + 1 + 1
_MANIP_CUBE_STRIDE = 3 + 4 + 1 + 1
_MANIP_BUTTON_STRIDE = 2 + 1 + 1

# Scene compact tail: drawer (scaled pos, vel), window (scaled pos, vel).
_SCENE_TAIL_DIM = 4

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
    if 'scene' in name:
        return 'scene'
    if 'cube' in name:
        return 'cube'
    raise ValueError(
        f"goal_representation='phi': unsupported env_name={env_name!r}. "
        "Expected a name containing one of: 'humanoidmaze', 'antmaze', 'puzzle', "
        "'scene', or 'cube'."
    )


def _parse_scene_compact_layout(obs_dim: int) -> tuple[int, int, int]:
    """Infer ``(num_cubes, num_buttons, num_button_states)`` from compact ``obs_dim``.

    After the fixed ManipSpace head, ``SceneEnv.compute_observation`` packs
    ``n_cubes`` blocks of stride 9, then ``n_buttons`` blocks of
    ``(one_hot[S] + pos + vel)`` with ``S = num_button_states``, then a tail of
    length ``_SCENE_TAIL_DIM``. OGBench scene defaults use ``S >= 2``.
    """

    dim = int(obs_dim)
    if dim < _MANIP_HEAD_DIM + _SCENE_TAIL_DIM:
        raise ValueError(
            f'Scene compact obs_dim={dim} is too small '
            f'(need at least head={_MANIP_HEAD_DIM} + tail={_SCENE_TAIL_DIM}).'
        )
    mid = dim - _MANIP_HEAD_DIM - _SCENE_TAIL_DIM
    if mid < 0:
        raise ValueError(f'Scene compact obs_dim={dim} has negative mid-body length.')

    solutions: list[tuple[int, int, int]] = []
    max_cubes = mid // _MANIP_CUBE_STRIDE
    for nc in range(0, max_cubes + 1):
        rest = mid - _MANIP_CUBE_STRIDE * nc
        if rest < 0:
            continue
        for S in range(2, 16):
            bstride = S + 2
            if rest % bstride != 0:
                continue
            nb = rest // bstride
            if nb < 1:
                continue
            solutions.append((nc, nb, S))

    if not solutions:
        raise ValueError(
            f'Scene obs_dim={dim}: cannot factor mid={mid} into '
            f'n_cubes * {_MANIP_CUBE_STRIDE} + n_buttons * (S+2) with S>=2.'
        )
    if len(solutions) > 1:
        raise ValueError(
            f'Scene obs_dim={dim}: ambiguous layout; mid={mid} matches multiple '
            f'(n_cubes, n_buttons, S) tuples: {solutions}.'
        )
    return solutions[0]


def scene_oracle_phi_from_goals(goals: jnp.ndarray, obs_dim: int) -> jnp.ndarray:
    """``phi`` vector matching ``SceneEnv.compute_oracle_observation`` ordering."""

    nc, nb, S = _parse_scene_compact_layout(obs_dim)
    dim = int(obs_dim)
    head = _MANIP_HEAD_DIM
    parts: list[jnp.ndarray] = []

    cur = head
    if nc > 0:
        cube_idx: list[int] = []
        for _ in range(nc):
            cube_idx.extend((cur, cur + 1, cur + 2))
            cur += _MANIP_CUBE_STRIDE
        parts.append(jnp.take(goals, jnp.asarray(cube_idx, dtype=jnp.int32), axis=-1))

    bstride = S + 2
    if nb > 0:
        scalars = []
        for _ in range(nb):
            block = goals[..., cur : cur + bstride]
            oh = block[..., :S]
            scalars.append(jnp.argmax(oh, axis=-1).astype(jnp.float32))
            cur += bstride
        parts.append(jnp.stack(scalars, axis=-1))

    expected_cur = dim - _SCENE_TAIL_DIM
    if cur != expected_cur:
        raise ValueError(
            f'Scene layout parse inconsistency: expected cursor {expected_cur} after '
            f'cubes/buttons, got {cur} for obs_dim={dim}.'
        )

    tail_idx = jnp.asarray([dim - 4, dim - 2], dtype=jnp.int32)
    parts.append(jnp.take(goals, tail_idx, axis=-1))

    if not parts:
        raise ValueError(f'Scene phi: empty parts for obs_dim={dim}.')
    if len(parts) == 1:
        return parts[0]
    return jnp.concatenate(parts, axis=-1)


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

    ``goal_representation(..., 'phi', ...)`` already picks channels from ``env_name``
    and the goal tensor; this tuple keeps critic/dynamics configs self-describing.

    Maze envs always return ``(0, 1)`` (planar goal). Puzzle/cube need ``obs_dim`` to
    validate the compact layout. Scene returns ``()`` (φ is assembled in
    ``scene_oracle_phi_from_goals``).
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
    if kind == 'scene':
        return ()
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
    """Validate that ``obs_dim`` is compatible with ``phi`` for this ``env_name``."""

    _ = phi_goal_obs_indices  # ignored: phi channels are fixed per env family

    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        return
    dim = int(obs_dim)
    try:
        kind = _env_goal_phi_kind(env_name)
    except ValueError as e:
        raise ValueError(f'{where}: {e}') from e

    if kind == 'puzzle':
        if not manip_button_state_indices(dim):
            raise ValueError(
                f'{where}: goal_representation={mode_l!r} for env={env_name!r}: '
                f'obs_dim={dim} is not compatible with the puzzle layout '
                f'(head={_MANIP_HEAD_DIM}, button_stride={_MANIP_BUTTON_STRIDE}).'
            )
        return
    if kind == 'cube':
        if not manip_cube_pos_indices(dim):
            raise ValueError(
                f'{where}: goal_representation={mode_l!r} for env={env_name!r}: '
                f'obs_dim={dim} is not compatible with the ManipSpace cube compact layout '
                f'(head={_MANIP_HEAD_DIM}, cube_stride={_MANIP_CUBE_STRIDE}).'
            )
        return
    if kind == 'scene':
        try:
            _parse_scene_compact_layout(dim)
        except ValueError as e:
            raise ValueError(f'{where}: scene layout: {e}') from e
        return
    if kind in ('antmaze', 'humanoidmaze'):
        if dim < 2:
            raise ValueError(
                f'{where}: goal_representation={mode_l!r} for env={env_name!r}: '
                f'obs_dim={dim} must be >= 2 for maze (x, y) goal channels.'
            )
        return
    raise AssertionError(f'unreachable phi kind={kind!r}')


def goal_representation(
    goals: jnp.ndarray | None,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...] = (),
    *,
    env_name: str | None = None,
) -> jnp.ndarray | None:
    """Map a full goal state to the configured goal representation."""

    _ = phi_goal_obs_indices  # ignored: phi channels are fixed per env family

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
    kind = _env_goal_phi_kind(env_name)

    if kind == 'puzzle':
        idxs = manip_button_state_indices(obs_dim)
        if not idxs:
            raise ValueError(
                f"goal_representation='phi' for env={env_name!r}: obs_dim={obs_dim} "
                f"is not compatible with the puzzle layout "
                f"(head={_MANIP_HEAD_DIM}, button_stride={_MANIP_BUTTON_STRIDE})."
            )
        return jnp.take(goals, jnp.asarray(idxs, dtype=jnp.int32), axis=-1)

    if kind == 'cube':
        idxs = manip_cube_pos_indices(obs_dim)
        if not idxs:
            raise ValueError(
                f"goal_representation='phi' for env={env_name!r}: obs_dim={obs_dim} "
                f"is not compatible with the ManipSpace cube layout "
                f"(head={_MANIP_HEAD_DIM}, cube_stride={_MANIP_CUBE_STRIDE})."
            )
        return jnp.take(goals, jnp.asarray(idxs, dtype=jnp.int32), axis=-1)

    if kind == 'scene':
        return scene_oracle_phi_from_goals(goals, obs_dim)

    if kind in ('antmaze', 'humanoidmaze'):
        if obs_dim < 2:
            raise ValueError(
                f"goal_representation='phi' for env={env_name!r}: obs_dim={obs_dim} must be >= 2 "
                'for maze (x, y) goal.'
            )
        take = jnp.asarray(_MAZE_GOAL_XY_INDICES, dtype=jnp.int32)
        return jnp.take(goals, take, axis=-1)

    raise AssertionError(f'unreachable phi kind={kind!r}')
