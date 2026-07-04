from __future__ import annotations

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import MLP


class DeterministicChunkActor(nn.Module):
    """Deterministic actor that outputs a flattened action chunk."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        x = jnp.concatenate(xs, axis=-1)
        out = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        return jnp.clip(out, -1.0, 1.0)


class ActorAgent(flax.struct.PyTreeNode):
    """SPI actor trained against an external critic scorer."""

    rng: Any
    actor: Any
    config: Any = nonpytree_field()

    def _goals(self, batch: dict) -> jnp.ndarray | None:
        # ``spi_goals`` is the dynamics-predicted subgoal (or broadcast) from
        # ``main._build_actor_batch_from_dynamics``. π and the critic scorer in
        # ``actor_loss`` share this vector so Q stays consistent with π.
        goals = batch.get('spi_goals', None)
        if goals is None:
            goals = batch.get('value_goals', None)
        return None if goals is None else jnp.asarray(goals, dtype=jnp.float32)

    def _chunk_dim(self) -> int:
        return int(self.config['actor_chunk_horizon']) * int(self.config['action_dim'])

    def _dim_mask(self, batch: dict, chunk_dim: int) -> jnp.ndarray:
        valids = batch.get('valids', None)
        if valids is None:
            return jnp.ones((batch['observations'].shape[0], chunk_dim), dtype=jnp.float32)
        valids = jnp.asarray(valids, dtype=jnp.float32)
        if valids.ndim == 1:
            return jnp.repeat(valids[:, None], chunk_dim, axis=1)
        steps = valids.shape[-1]
        if chunk_dim % steps != 0:
            return jnp.ones((batch['observations'].shape[0], chunk_dim), dtype=jnp.float32)
        rep = chunk_dim // steps
        return jnp.repeat(valids, rep, axis=-1)

    def _proposal_chunks(self, batch: dict) -> jnp.ndarray:
        external = batch.get('proposal_partial_chunks', None)
        if external is None:
            raise ValueError('Actor update requires proposal_partial_chunks in the batch.')
        external = jnp.asarray(external, dtype=jnp.float32)
        if external.ndim == 4:
            external = external.reshape(external.shape[0], external.shape[1], -1)
        if external.ndim != 3:
            raise ValueError(f'proposal_partial_chunks must be rank-3/4, got shape={external.shape}')
        chunk_dim = self._chunk_dim()
        if external.shape[-1] != chunk_dim:
            raise ValueError(f'proposal_partial_chunks last dim must be {chunk_dim}, got {external.shape[-1]}.')
        return jax.lax.stop_gradient(external)

    def actor_loss(self, batch: dict, actor_params: dict, critic_agent: Any) -> tuple[jnp.ndarray, dict]:
        proposal_chunks = self._proposal_chunks(batch)
        proposal_scores = batch.get('proposal_scores', None)
        if proposal_scores is None:
            raise ValueError(
                'SPI actor path requires proposal_scores precomputed from the current critic snapshot. '
                'Rescore proposals in the training loop before actor update.'
            )
        proposal_scores = jax.lax.stop_gradient(jnp.asarray(proposal_scores, dtype=jnp.float32))
        if proposal_scores.ndim != 2:
            raise ValueError(f'proposal_scores must be rank-2 [B, K], got shape={proposal_scores.shape}')
        if proposal_scores.shape[:2] != proposal_chunks.shape[:2]:
            raise ValueError(
                'proposal_scores must align with proposal_partial_chunks, '
                f'got scores={proposal_scores.shape} proposals={proposal_chunks.shape}.'
            )

        goals = self._goals(batch)
        rho = jax.nn.softmax(proposal_scores, axis=1)
        rho = jax.lax.stop_gradient(rho)

        actor_chunk = self.actor(batch['observations'], goals, params=actor_params)
        actor_q = critic_agent.score_action_chunks(
            batch['observations'],
            goals,
            actor_chunk,
            network_params=critic_agent.network.params,
            use_partial_critic=True,
        )[:, 0]

        dim_mask = self._dim_mask(batch, proposal_chunks.shape[-1])
        diff = (actor_chunk[:, None, :] - proposal_chunks) * dim_mask[:, None, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        prox = jnp.sum(rho * sqdist, axis=1)
        # Scale critic Q by batch-mean |Q| so the SPI term is ``-Q / (mean|Q| + eps)``.
        q_eps = jnp.asarray(float(self.config.get('spi_q_norm_eps', 1e-6)), dtype=jnp.float32)
        q_scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(actor_q)) + q_eps)
        actor_q_scaled = actor_q / q_scale
        spi_tau = float(self.config['spi_tau'])
        q_term = -actor_q_scaled
        prox_term = prox / (2.0 * spi_tau)
        actor_loss = jnp.mean(q_term + prox_term)

        rho_eps = 1e-8
        rho_entropy = -jnp.sum(rho * jnp.log(rho + rho_eps), axis=1).mean()
        idm_ref_chunk = jnp.sum(rho[:, :, None] * proposal_chunks, axis=1)
        actor_action_norm = jnp.sqrt(jnp.sum(actor_chunk**2, axis=-1) + 1e-12).mean()
        idm_action_norm = jnp.sqrt(jnp.sum(idm_ref_chunk**2, axis=-1) + 1e-12).mean()
        return actor_loss, {
            'spi_actor/actor_loss': actor_loss,
            'spi_actor/q_term': q_term.mean(),
            'spi_actor/prox_term': prox_term.mean(),
            'spi_actor/spi_tau': jnp.asarray(spi_tau, dtype=jnp.float32),
            'spi_actor/actor_action_norm': actor_action_norm,
            'spi_actor/idm_action_norm': idm_action_norm,
            'spi_actor/action_l2_to_idm': prox.mean(),
            'spi_actor/q_mean': actor_q.mean(),
            'spi_actor/q_max': actor_q.max(),
            'spi_actor/q_min': actor_q.min(),
            'spi_actor/prox_mean': prox.mean(),
            'spi_actor/prox_max': prox.max(),
            'spi_actor/prox_min': prox.min(),
            'spi_actor/rho_entropy': rho_entropy,
            'spi_actor/rho_max_mean': jnp.max(rho, axis=1).mean(),
        }

    @jax.jit
    def update(self, batch: dict, critic_agent: Any):
        new_rng, _ = jax.random.split(self.rng)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)

        def loss_fn(actor_params):
            return self.actor_loss(batch, actor_params=actor_params, critic_agent=critic_agent)

        new_actor, info = self.actor.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, actor=new_actor), info

    def sample_actions(self, observations, goals=None):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)[None]
        elif goals is not None:
            goals = jnp.asarray(goals, dtype=jnp.float32)

        chunk = self.actor(observations, goals)
        horizon = int(self.config['actor_chunk_horizon'])
        action_dim = int(self.config['action_dim'])
        chunk = chunk.reshape(chunk.shape[0], horizon, action_dim)
        if squeeze:
            chunk = chunk[0]
        return chunk

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, config: dict, ex_goals: np.ndarray | None = None):
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_goal = None if ex_goals is None else jnp.asarray(ex_goals, dtype=jnp.float32)

        config = dict(config)

        actor_def = DeterministicChunkActor(
            hidden_dims=(512, 512, 512),
            action_dim=int(config['actor_chunk_horizon']) * int(config['action_dim']),
            layer_norm=bool(config['spi_actor_layer_norm']),
        )
        actor_params = actor_def.init(init_rng, ex_obs, ex_goal)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(float(config['lr'])))
        return cls(rng=rng, actor=actor, config=flax.core.FrozenDict(**config))


def get_actor_config():
    return ml_collections.ConfigDict(
        dict(
            lr=3e-4,
            spi_tau=5.0,
            spi_actor_layer_norm=True,
            spi_q_norm_eps=1e-6,
            # π and Q in the SPI loss are always conditioned on the dynamics-predicted
            # subgoal (``spi_goals`` from ``main._build_actor_batch_from_dynamics``).
            actor_chunk_horizon=ml_collections.config_dict.placeholder(int),
            action_dim=2,
        )
    )
