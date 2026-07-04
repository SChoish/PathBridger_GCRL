"""Minimal network primitives used across the project (only ``MLP``)."""

from __future__ import annotations

from typing import Any, Sequence

import flax.linen as nn


def default_init(scale: float = 1.0):
    """Default kernel initializer (variance scaling, fan_avg, uniform)."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """Multi-layer perceptron with optional final activation and LayerNorm.

    Attributes:
        hidden_dims: Hidden layer dimensions (last entry is the output width).
        activations: Activation function applied between layers.
        activate_final: Whether to apply activation/LayerNorm after the last layer.
        kernel_init: Kernel initializer for ``nn.Dense``.
        layer_norm: Whether to apply ``nn.LayerNorm`` after each activation.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


__all__ = ['default_init', 'MLP']
