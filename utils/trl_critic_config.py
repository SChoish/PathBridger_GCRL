"""TRL critic_agent presets for sweep YAML generation (gap10 baseline)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MAZE_ACTOR_GOAL_SAMPLING: dict[str, Any] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 1.0,
    'actor_p_randomgoal': 0.0,
    'actor_geom_sample': False,
}

PUZZLE_ACTOR_GOAL_SAMPLING: dict[str, Any] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 0.5,
    'actor_p_randomgoal': 0.5,
    'actor_geom_sample': True,
}

# TRL-only critic. Structural/numeric constants are hardcoded in agents/critic.py;
# only env-varying knobs (discount, value_distance_weight_power) and goal-sampling
# probabilities remain here. full_chunk_horizon is synced from the top-level horizon.
TRL_CRITIC_COMMON: dict[str, Any] = {
    'full_chunk_horizon': 25,
}

STANDARD_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
}

LONG_HORIZON_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 0.0,
    'value_p_randomgoal': 0.0,
}

# gap10 baseline (write_trl_gap10_g099_sweep_yaml.py Table 5).
TRL_ENV_SPECS: dict[str, dict[str, Any]] = {
    'amm': {
        'env_name': 'antmaze-medium-navigate-v0',
        'stem': 'antmaze_medium',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'aml': {
        'env_name': 'antmaze-large-navigate-v0',
        'stem': 'antmaze_large',
        'regime': 'standard',
        'discount': 0.995,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'amg': {
        'env_name': 'antmaze-giant-navigate-v0',
        'stem': 'antmaze_giant',
        'regime': 'standard',
        'discount': 0.999,
        'value_distance_weight_power': 0.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'p3': {
        'env_name': 'puzzle-3x3-play-v0',
        'stem': 'puzzle_3x3',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 0.5,
        'batch_size': 1024,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
    },
    'p4': {
        'env_name': 'puzzle-4x4-play-v0',
        'stem': 'puzzle_4x4',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 2.0,
        'batch_size': 1024,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
    },
    'cs': {
        'env_name': 'cube-single-play-v0',
        'stem': 'cube_single',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 0.7,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'cd': {
        'env_name': 'cube-double-play-v0',
        'stem': 'cube_double',
        'regime': 'standard',
        'discount': 0.99,
        'value_distance_weight_power': 1.0,
        'batch_size': 1024,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
    'ct': {
        'env_name': 'cube-triple-play-v0',
        'stem': 'cube_triple',
        'regime': 'standard',
        'discount': 0.995,
        'value_distance_weight_power': 1.0,
        'batch_size': 4096,
        'actor_goal_sampling': MAZE_ACTOR_GOAL_SAMPLING,
    },
}

ENV_ORDER = ['amm', 'aml', 'amg', 'p3', 'p4', 'cs', 'cd', 'ct']

# OGBench TimeLimit (``max_episode_steps``) per environment family. This mirrors
# the values registered by the installed ``ogbench`` package so that the static
# helper below agrees with the live-env budget used during evaluation. Evaluation
# always rolls out to the environment's max episode length; ``eval_max_chunks``
# is simply that budget expressed in action chunks.
ENV_MAX_EPISODE_STEPS: dict[str, int] = {
    'antmaze-medium': 1000,
    'antmaze-large': 1000,
    'antmaze-giant': 1000,
    'antmaze-teleport': 1000,
    'humanoidmaze-medium': 2000,
    'humanoidmaze-large': 2000,
    'humanoidmaze-giant': 4000,
    'humanoidmaze-teleport': 2000,
    'pointmaze-medium': 1000,
    'pointmaze-large': 1000,
    'pointmaze-giant': 1000,
    'pointmaze-teleport': 1000,
    'antsoccer-arena': 1000,
    'antsoccer-medium': 1000,
    'cube-single': 200,
    'cube-double': 500,
    'cube-triple': 1000,
    'cube-quadruple': 1000,
    'cube-octuple': 1500,
    'scene': 750,
    'puzzle-3x3': 500,
    'puzzle-4x4': 500,
    'puzzle-4x5': 1000,
    'puzzle-4x6': 1000,
}

# Short aliases used throughout the sweep configs / eval scripts.
ENV_ALIAS_TO_FAMILY: dict[str, str] = {
    'amm': 'antmaze-medium',
    'aml': 'antmaze-large',
    'amg': 'antmaze-giant',
    'hmm': 'humanoidmaze-medium',
    'hml': 'humanoidmaze-large',
    'hmg': 'humanoidmaze-giant',
    'p3': 'puzzle-3x3',
    'p4': 'puzzle-4x4',
    'p45': 'puzzle-4x5',
    'p46': 'puzzle-4x6',
    'cs': 'cube-single',
    'cd': 'cube-double',
    'ct': 'cube-triple',
    'cq': 'cube-quadruple',
    'scene': 'scene',
}


def _resolve_env_family(env: str) -> str:
    """Map a short alias or full env name to a canonical family key."""
    key = str(env).strip().lower()
    if key in ENV_ALIAS_TO_FAMILY:
        return ENV_ALIAS_TO_FAMILY[key]
    if key in ENV_MAX_EPISODE_STEPS:
        return key
    # Full env_name such as 'antmaze-medium-navigate-v0' or 'puzzle-3x3-play-v0'.
    # Prefer the longest matching family prefix to disambiguate e.g. 4x4 vs 4x5.
    candidates = [fam for fam in ENV_MAX_EPISODE_STEPS if key.startswith(fam)]
    if candidates:
        return max(candidates, key=len)
    raise KeyError(f'Unknown environment {env!r}; no max_episode_steps mapping found.')


def env_max_episode_steps_for_env(env: str) -> int:
    """Return the OGBench TimeLimit (max episode steps) for ``env``."""
    return ENV_MAX_EPISODE_STEPS[_resolve_env_family(env)]


def eval_max_chunks_for_env(env: str, action_chunk_horizon: int) -> int:
    """Number of action chunks needed to reach the env's max episode length.

    Evaluation always runs to the environment's TimeLimit, so the chunk budget is
    ``ceil(max_episode_steps / action_chunk_horizon)``.
    """
    h = int(action_chunk_horizon)
    if h < 1:
        raise ValueError(f'action_chunk_horizon must be >= 1, got {action_chunk_horizon}.')
    max_steps = env_max_episode_steps_for_env(env)
    return max(1, (max_steps + h - 1) // h)


def trl_critic_agent_config(env_slug: str) -> dict[str, Any]:
    """Return full critic_agent block for env_slug (gap10 TRL baseline)."""
    spec = TRL_ENV_SPECS[env_slug]
    cfg = deepcopy(TRL_CRITIC_COMMON)
    cfg['discount'] = float(spec['discount'])
    cfg['value_distance_weight_power'] = float(spec['value_distance_weight_power'])
    if str(spec['regime']) == 'long_horizon':
        cfg.update(LONG_HORIZON_VALUE_GOAL_SAMPLING)
    else:
        cfg.update(STANDARD_VALUE_GOAL_SAMPLING)
    return cfg


def trl_env_spec(env_slug: str) -> dict[str, Any]:
    return TRL_ENV_SPECS[env_slug]
