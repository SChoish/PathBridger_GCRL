"""Inverse dynamics MLP module + helpers (no training scaffolding)."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class InverseDynamicsMLP(nn.Module):
    """Predict ``a_t`` from concatenated ``(s_t, s_{t+1})``."""

    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, obs: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, next_obs], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=True,
        )(x)


def parse_hidden_dims(hidden_dims: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(hidden_dims, str):
        return tuple(int(x.strip()) for x in hidden_dims.split(',') if x.strip())
    return tuple(hidden_dims)


__all__ = ['InverseDynamicsMLP', 'parse_hidden_dims']
