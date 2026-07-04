"""Environment / maze helpers for dynamics rollout scripts (state sync, xy snap, navigator setup)."""

from __future__ import annotations

from collections.abc import Callable

import os

import numpy as np

from rollout.maze_navigator import MazeNavigatorMap


def configure_mujoco_gl(mujoco_gl: str) -> None:
    """Set ``MUJOCO_GL`` before creating MuJoCo-backed envs (headless ``rgb_array`` rollouts).

    If ``mujoco_gl`` is non-empty, it must be one of ``egl``, ``osmesa``, ``glfw``.
    If empty and ``DISPLAY`` is unset, default to ``egl`` so ``env.render()`` does not require X11/GLFW.
    """
    s = (mujoco_gl or '').strip().lower()
    if s:
        if s not in ('egl', 'osmesa', 'glfw'):
            raise ValueError(f'Invalid mujoco_gl={mujoco_gl!r} (use egl, osmesa, glfw, or empty for auto)')
        os.environ['MUJOCO_GL'] = s
    elif not (os.environ.get('DISPLAY') or '').strip():
        os.environ.setdefault('MUJOCO_GL', 'egl')


def sync_env_state_from_obs_vector(env, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
    """Set MuJoCo state from flat observation (qpos‖qvel) and maze goal xy from ``goal_obs[:2]``."""
    u = env.unwrapped
    ob = np.asarray(obs, dtype=np.float64).reshape(-1)
    nq, nv = int(u.model.nq), int(u.model.nv)
    need = nq + nv
    if ob.shape[0] < need:
        raise ValueError(f'Observation dim {ob.shape[0]} < nq+nv={need} (cannot replay physics).')
    u.set_state(ob[:nq].copy(), ob[nq:need].copy())
    if hasattr(u, 'set_goal'):
        g_xy = np.asarray(goal_obs[:2], dtype=np.float64).reshape(2)
        u.set_goal(goal_xy=g_xy)
    return np.asarray(u.get_ob(), dtype=np.float32)


_MANIP_XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float64)
_MANIP_XYZ_SCALER = 10.0
_MANIP_GRIPPER_SCALER = 3.0
_MANIP_RIGHT_DRIVER_LIMIT = 0.8


def is_manipspace_env(env) -> bool:
    """Heuristic: True for OGBench manipspace envs (cube/scene/puzzle) that emit compact obs."""
    u = getattr(env, 'unwrapped', env)
    return (
        hasattr(u, 'compute_ob_info')
        and hasattr(u, '_gripper_opening_joint_id')
        and hasattr(u, '_data')
        and hasattr(u, '_model')
    )


def snapshot_manip_mocap(env) -> tuple[np.ndarray, np.ndarray] | None:
    """Copy MuJoCo mocap buffers (goal markers, etc.) for ManipSpace envs; else None."""
    u = getattr(env, 'unwrapped', env)
    if not is_manipspace_env(u):
        return None
    d = u._data
    return (np.asarray(d.mocap_pos, dtype=np.float64).copy(), np.asarray(d.mocap_quat, dtype=np.float64).copy())


def apply_snapshot_manip_mocap(dst_env, mocap_pos: np.ndarray, mocap_quat: np.ndarray) -> bool:
    """Paste mocap pose into ``dst_env`` and run ``mj_forward`` so renders match the source episode."""
    import mujoco

    du = getattr(dst_env, 'unwrapped', dst_env)
    if not is_manipspace_env(du):
        return False
    dp, dq = du._data.mocap_pos, du._data.mocap_quat
    if dp.shape != mocap_pos.shape or dq.shape != mocap_quat.shape:
        return False
    dp[:] = np.asarray(mocap_pos, dtype=np.float64).reshape(dp.shape)
    dq[:] = np.asarray(mocap_quat, dtype=np.float64).reshape(dq.shape)
    mujoco.mj_forward(du._model, du._data)
    return True


_MANIP_BUTTON_SCALER = 120.0


def sync_env_state_from_compact_manip_obs(env, obs: np.ndarray) -> np.ndarray:
    """Decode an OGBench manipspace compact observation into ``qpos``/``qvel`` and apply via ``set_state``.

    Cube-family obs layout (per ``ogbench/manipspace/envs/manipspace_env.py::compute_observation``)::

        [joint_pos (J),
         joint_vel (J),
         (effector_pos - xyz_center) * 10 (3),
         cos(effector_yaw), sin(effector_yaw),
         gripper_opening * 3, gripper_contact,
         per-cube i: (block_pos - xyz_center) * 10 (3), block_quat (4),
                     cos(block_yaw), sin(block_yaw)]

    Puzzle-family obs (``ogbench/manipspace/envs/puzzle_env.py::compute_observation``) shares the
    head and trails ``per-button: [one_hot_state(n_states), button_pos*120, button_vel]`` instead.
    For puzzle envs we additionally restore ``_cur_button_states`` (argmax of one-hot) and call
    ``PuzzleEnv.set_state(qpos, qvel, button_states)`` which requires the third arg.

    ``effector_pos``, ``effector_yaw``, and the per-cube ``cos/sin yaw`` channels are derivable from
    arm joints and the cube quat respectively, so we ignore them when reconstructing state.

    Gripper: only ``gripper_opening`` is recoverable from the obs. We invert
    ``opening = clip(qpos[right_driver_joint] / 0.8, 0, 1)`` and mirror the value onto
    ``left_driver_joint`` (mimicked by an equality constraint in the model). The other 6 robotiq
    follower/coupler/spring joints are left at zero, which gives a slightly cartoonish gripper but
    keeps cube + arm pose faithful — sufficient for subgoal visualization.

    Velocities of the cube and gripper are not in the compact obs, so they are zeroed.
    """
    u = getattr(env, 'unwrapped', env)
    if not is_manipspace_env(env):
        raise ValueError('sync_env_state_from_compact_manip_obs: env is not an OGBench manipspace env.')
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)

    ob_info = u.compute_ob_info()
    J = int(np.asarray(ob_info['proprio/joint_pos']).shape[0])
    n_cubes = int(getattr(u, '_num_cubes', 0))
    n_buttons = int(getattr(u, '_num_buttons', 0))
    n_button_states = int(getattr(u, '_num_button_states', 2))

    if n_buttons > 0:
        b0_pos_dim = int(np.asarray(ob_info['privileged/button_0_pos']).shape[0])
        b0_vel_dim = int(np.asarray(ob_info['privileged/button_0_vel']).shape[0])
    else:
        b0_pos_dim = b0_vel_dim = 0

    head_dim = 2 * J + 3 + 1 + 1 + 1 + 1
    cube_block = 3 + 4 + 1 + 1
    button_block = n_button_states + b0_pos_dim + b0_vel_dim
    needed = head_dim + n_cubes * cube_block + n_buttons * button_block
    if obs.shape[0] < needed:
        raise ValueError(
            f'compact manip obs too short: dim={obs.shape[0]} < required {needed} '
            f'for J={J}, n_cubes={n_cubes}, n_buttons={n_buttons}.'
        )

    p = 0
    arm_qpos = obs[p:p + J]
    p += J
    arm_qvel = obs[p:p + J]
    p += J
    p += 3 + 1 + 1  # skip effector_pos / cos/sin yaw (FK).
    grip_open_scaled = float(obs[p])
    p += 1
    p += 1  # skip gripper_contact (sensor).

    cube_pos_quat: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_cubes):
        c_pos_scaled = obs[p:p + 3]
        p += 3
        c_quat = obs[p:p + 4]
        p += 4
        p += 1 + 1  # skip cos/sin yaw.
        cube_pos_quat.append((c_pos_scaled, c_quat))

    button_states = np.zeros((n_buttons,), dtype=np.int64)
    button_pos: list[np.ndarray] = []
    for i in range(n_buttons):
        oh = obs[p:p + n_button_states]
        p += n_button_states
        bp = obs[p:p + b0_pos_dim]
        p += b0_pos_dim
        p += b0_vel_dim  # button_vel ignored (qvel zeroed below).
        button_states[i] = int(np.argmax(oh))
        button_pos.append(np.asarray(bp, dtype=np.float64) / _MANIP_BUTTON_SCALER)

    qpos = np.array(u._data.qpos, dtype=np.float64).copy()
    qvel = np.zeros_like(np.asarray(u._data.qvel, dtype=np.float64))
    qpos[0:J] = arm_qpos
    qvel[0:J] = arm_qvel

    grip_open = float(np.clip(grip_open_scaled / _MANIP_GRIPPER_SCALER, 0.0, 1.0))
    grip_qpos = grip_open * _MANIP_RIGHT_DRIVER_LIMIT
    right_driver_id = int(getattr(u, '_gripper_opening_joint_id'))
    qpos[int(u._model.jnt_qposadr[right_driver_id])] = grip_qpos
    try:
        ldj_id = u._model.joint('ur5e/robotiq/left_driver_joint').id
        qpos[int(u._model.jnt_qposadr[ldj_id])] = grip_qpos
    except Exception:
        pass

    for i, (c_pos_scaled, c_quat) in enumerate(cube_pos_quat):
        try:
            jid = u._model.joint(f'object_joint_{i}').id
        except Exception:
            jid = u._model.joint('object_joint_0').id
        qadr = int(u._model.jnt_qposadr[jid])
        qpos[qadr:qadr + 3] = np.asarray(c_pos_scaled, dtype=np.float64) / _MANIP_XYZ_SCALER + _MANIP_XYZ_CENTER
        q = np.asarray(c_quat, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        qpos[qadr + 3:qadr + 7] = q / qn if qn > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])

    for i, bp in enumerate(button_pos):
        try:
            jid = u._model.joint(f'buttonbox_joint_{i}').id
        except Exception:
            continue
        qadr = int(u._model.jnt_qposadr[jid])
        qpos[qadr:qadr + bp.shape[0]] = bp

    if n_buttons > 0:
        u.set_state(qpos, qvel, button_states)
    else:
        u.set_state(qpos, qvel)
    return np.asarray(u.compute_observation(), dtype=np.float32).reshape(-1)


def sync_env_state_from_obs_vector_aligned(env, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
    """Update physics like :func:`sync_env_state_from_obs_vector` and return an obs matching ``env``'s space.

    :class:`utils.env_utils.FrameStackWrapper` keeps a deque of base observations; mutating the base env without
    going through ``reset`` / ``step`` leaves stale frames. When that wrapper is detected, refill the deque with
    the current base ``get_ob()`` repeated ``num_stack`` times (same protocol as ``FrameStackWrapper.reset``).
    """
    sync_env_state_from_obs_vector(env, obs, goal_obs)
    if hasattr(env, 'frames') and hasattr(env, 'num_stack') and hasattr(env, 'get_observation'):
        ob0 = np.asarray(env.unwrapped.get_ob(), dtype=np.float32).reshape(-1)
        env.frames.clear()
        for _ in range(int(env.num_stack)):
            env.frames.append(ob0)
        return np.asarray(env.get_observation(), dtype=np.float32).reshape(-1)
    return np.asarray(env.unwrapped.get_ob(), dtype=np.float32).reshape(-1)


def make_xy_clamper(
    goal_obs: np.ndarray,
    navigator: MazeNavigatorMap | None,
    clamp_dim0: int,
    clamp_dim1: int,
    navigator_clamp_mode: str,
    navigator_edge_inset: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that optionally snaps ``(clamp_dim0, clamp_dim1)`` xy using ``navigator``."""
    if navigator is None:

        def _identity(vec: np.ndarray) -> np.ndarray:
            return vec

        return _identity

    g_np = np.asarray(goal_obs, dtype=np.float32)

    def _clamp(vec: np.ndarray) -> np.ndarray:
        kw = {'mode': navigator_clamp_mode, 'edge_inset': float(navigator_edge_inset)}
        if navigator_clamp_mode == 'oracle':
            kw['goal_obs'] = g_np
        return navigator.clamp_obs_xy(vec, clamp_dim0, clamp_dim1, **kw)

    return _clamp


def load_maze_navigator_snap(maze_type: str, env_name: str) -> MazeNavigatorMap:
    """Build a :class:`MazeNavigatorMap` when ``--navigator snap`` is enabled."""
    mt = maze_type.strip()
    if mt:
        return MazeNavigatorMap.from_maze_type_embedded(mt)
    try:
        return MazeNavigatorMap.from_env_name(env_name)
    except ValueError as ex:
        raise ValueError(
            f'Could not infer maze type from env_name={env_name!r} ({ex}). '
            'Pass --maze_type= one of arena|medium|large|giant|teleport.'
        ) from ex


def format_maze_navigator_log(
    navigator: MazeNavigatorMap,
    navigator_clamp: str,
    navigator_edge_inset: float,
) -> str:
    ei = float(navigator_edge_inset)
    box_half = 0.5 * float(navigator.maze_unit) * max(0.0, min(1.0, 1.0 - ei))
    return (
        f'Navigator snap enabled (source={navigator.source}, maze_type={navigator.maze_type}, '
        f'clamp={navigator_clamp}, edge_inset={ei}, '
        f'box_half={box_half:.3f}, free cells={len(navigator.free_xy)})'
    )


def max_episode_steps_from_wrappers(env) -> int | None:
    """Return ``TimeLimit._max_episode_steps`` if present on ``env`` or a nested wrapper."""
    w = env
    for _ in range(32):
        m = getattr(w, '_max_episode_steps', None)
        if m is not None:
            return int(m)
        w = getattr(w, 'env', None)
        if w is None:
            break
    return None


def env_render_rgb_u8(env) -> np.ndarray | None:
    """Return a single RGB uint8 frame from ``env.render()``, or None if unavailable."""
    try:
        fr = env.render()
    except Exception:
        return None
    if fr is None:
        return None
    x = np.asarray(fr)
    if x.ndim != 3 or x.shape[-1] < 3:
        return None
    x = x[..., :3]
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(x)
