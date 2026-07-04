"""Dataset sampler for full/action-chunk critic training."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np

from utils.datasets import (
    Dataset,
    augment_batch_images,
    gather_stacked_observations,
    goal_final_indices,
)


@dataclasses.dataclass
class CriticSequenceDataset:
    """Samples full/action chunks without crossing episode boundaries.

    ``clip_chunk_to_goal`` (default from critic config: ``True``) mirrors the
    dynamics dataset's ``clip_path_to_goal`` behavior for Q backups. If a sampled
    same-trajectory value goal lies inside the backup window, the target terminates
    at the goal: ``next_obs=s_goal``, ``backup_horizon=steps_to_goal``, and ``mask=0``.
    Random goals or goals outside the window keep the fixed-horizon bootstrap.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        self.full_chunk_horizon = int(self.config['full_chunk_horizon'])
        self.action_chunk_horizon = int(self.config['action_chunk_horizon'])
        if self.full_chunk_horizon < 1:
            raise ValueError('full_chunk_horizon must be >= 1.')
        if self.action_chunk_horizon < 1:
            raise ValueError('action_chunk_horizon must be >= 1.')
        if self.action_chunk_horizon > self.full_chunk_horizon:
            raise ValueError('action_chunk_horizon must be <= full_chunk_horizon.')

        valids = np.zeros(self.size, dtype=np.float32)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            last_start = int(end) - self.full_chunk_horizon
            if last_start >= int(start):
                valids[int(start) : last_start + 1] = 1.0
        (self.valid_starts,) = np.nonzero(valids > 0)
        if len(self.valid_starts) == 0:
            raise ValueError(
                f'No valid starts for full_chunk_horizon={self.full_chunk_horizon} in dataset size={self.size}.'
            )

        # Cached metadata for hot-path sampling.
        self.final_state_for_idx = np.empty(self.size, dtype=np.int64)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            self.final_state_for_idx[int(start) : int(end) + 1] = int(end)
        self.full_offsets = np.arange(self.full_chunk_horizon, dtype=np.int64)
        self.action_offsets = np.arange(self.action_chunk_horizon, dtype=np.int64)
        self.discount_pows = np.power(float(self.config['discount']), self.full_offsets.astype(np.float32))
        # Per-step discount over the *partial* action-chunk window (used by chunk-IQL Q backup).
        self.action_discount_pows = np.power(
            float(self.config['discount']), self.action_offsets.astype(np.float32)
        )
        self.valids_template = np.ones((self.action_chunk_horizon,), dtype=np.float32)
        self.full_chunk_horizon_template = np.full((1,), self.full_chunk_horizon, dtype=np.float32)
        self.action_chunk_horizon_template = np.full((1,), self.action_chunk_horizon, dtype=np.float32)
        self.clip_chunk_to_goal = bool(self.config.get('clip_chunk_to_goal', True))
        # valid_starts guarantee no terminal inside full chunk; default masks are ones.
        self.full_chunk_masks_template = np.ones((1,), dtype=np.float32)
        self.action_chunk_masks_template = np.ones((1,), dtype=np.float32)

        if self.config.get('frame_stack') is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def get_observations(self, idxs):
        if self.config.get('frame_stack') is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: np.asarray(arr[idxs]), self.dataset['observations'])
        return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        return gather_stacked_observations(
            self.dataset['observations'],
            idxs,
            self.initial_locs,
            int(self.config['frame_stack']),
        )

    def _lookup_finals(self, idxs: np.ndarray, *, max_goal_steps: int | None = None) -> np.ndarray:
        if max_goal_steps is None or int(max_goal_steps) <= 0:
            return self.final_state_for_idx[idxs]
        return goal_final_indices(self.terminal_locs, idxs, max_goal_steps)

    def _build_full_chunk_indices(self, idxs: np.ndarray) -> np.ndarray:
        return idxs[:, None] + self.full_offsets[None, :]

    def _build_action_chunk_indices(self, idxs: np.ndarray) -> np.ndarray:
        finals = self._lookup_finals(idxs)
        last_valid = finals - self.action_chunk_horizon
        if np.any(idxs > last_valid):
            raise ValueError('Action chunk start crosses episode boundary.')
        return idxs[:, None] + self.action_offsets[None, :]

    def _chunk_backup(
        self,
        idxs: np.ndarray,
        value_goal_idxs: np.ndarray,
        *,
        horizon: int,
        discount_pows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build clipped or fixed-horizon GC backup fields for one chunk horizon.

        The reward window is ``k = 0 .. horizon-1``. A goal at ``t+horizon`` is
        therefore not considered reached by this chunk; it is handled by the
        bootstrap value at ``s_{t+horizon}``.
        """
        offsets = np.arange(int(horizon), dtype=np.int64)[None, :]
        chunk_idxs = idxs[:, None] + offsets
        success_steps = (chunk_idxs == value_goal_idxs[:, None]).astype(np.float32)
        reward_offset = 1.0 if bool(self.config.get('gc_negative', True)) else 0.0
        step_rewards = success_steps - reward_offset
        rewards = np.sum(step_rewards * discount_pows[None, :], axis=1).astype(np.float32)

        if self.clip_chunk_to_goal:
            goal_offsets = value_goal_idxs - idxs
            reached = (0 <= goal_offsets) & (goal_offsets < int(horizon))
            backup_horizon = np.where(reached, goal_offsets, int(horizon)).astype(np.float32)
            next_idxs = np.where(reached, value_goal_idxs, idxs + int(horizon)).astype(np.int64)
            masks = (1.0 - reached.astype(np.float32)).astype(np.float32)
        else:
            backup_horizon = np.full((idxs.shape[0],), int(horizon), dtype=np.float32)
            next_idxs = (idxs + int(horizon)).astype(np.int64)
            masks = np.ones((idxs.shape[0],), dtype=np.float32)

        next_observations = np.asarray(self.get_observations(next_idxs), dtype=np.float32)
        return rewards, next_observations, backup_horizon, masks

    def sample_goals(self, idxs):
        batch_size = len(idxs)
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        final_state_idxs = self._lookup_finals(
            idxs,
            max_goal_steps=self.config.get('max_goal_steps', None),
        )
        if bool(self.config['value_geom_sample']):
            offsets = np.random.geometric(p=1 - float(self.config['discount']), size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        p_cur = float(self.config['value_p_curgoal'])
        p_traj = float(self.config['value_p_trajgoal'])
        if p_cur == 1.0:
            return idxs
        goal_idxs = np.where(np.random.rand(batch_size) < p_traj / (1.0 - p_cur), traj_goal_idxs, random_goal_idxs)
        goal_idxs = np.where(np.random.rand(batch_size) < p_cur, idxs, goal_idxs)
        return goal_idxs

    def sample_trl_goals(self, idxs):
        """Sample strictly future same-trajectory goals required by TRL midpoint targets."""
        final_state_idxs = self._lookup_finals(
            idxs,
            max_goal_steps=self.config.get('max_goal_steps', None),
        )
        if bool(self.config['value_geom_sample']):
            offsets = np.random.geometric(p=1 - float(self.config['discount']), size=len(idxs))
            return np.minimum(idxs + offsets, final_state_idxs).astype(np.int64)

        distances = np.random.rand(len(idxs))
        return np.round(
            (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
        ).astype(np.int64)

    def _sample_trl_fields(self, idxs: np.ndarray, value_goal_idxs: np.ndarray) -> dict:
        """Sample state-pair transitive value tuples ``i < k < j`` and local Q goals."""
        batch_size = len(idxs)
        finals = self._lookup_finals(idxs)
        horizon = int(self.action_chunk_horizon)
        base_horizon = int(self.config.get('value_base_horizon', horizon))
        base_horizon = max(1, min(base_horizon, int(self.full_chunk_horizon)))

        max_base_offsets = np.maximum(1, np.minimum(base_horizon, finals - idxs))
        base_offsets = 1 + np.floor(np.random.rand(batch_size) * max_base_offsets).astype(np.int64)
        base_offsets = np.minimum(base_offsets, max_base_offsets).astype(np.int64)
        base_goal_idxs = idxs + base_offsets

        split_low = idxs + 1
        split_high = value_goal_idxs - 1
        tri_valid = (value_goal_idxs > idxs + 1) & (split_high >= split_low)
        split_idxs = idxs.copy()
        if np.any(tri_valid):
            lo = split_low[tri_valid]
            hi = split_high[tri_valid]
            split_idxs[tri_valid] = lo + np.random.randint(0, hi - lo + 1, size=int(np.sum(tri_valid)))

        valid = np.asarray(tri_valid, dtype=bool)
        if np.any(valid):
            if np.any(split_idxs[valid] <= idxs[valid]):
                raise ValueError('trl: split_idxs must satisfy i < k for valid samples.')
            if np.any(split_idxs[valid] >= value_goal_idxs[valid]):
                raise ValueError('trl: split_idxs must satisfy k < j for valid samples.')
            if np.any(value_goal_idxs[valid] > finals[valid]):
                raise ValueError('trl: value goals must stay within trajectory.')

        split_obs = np.asarray(self.get_observations(split_idxs), dtype=np.float32)
        value_goals = np.asarray(self.get_observations(value_goal_idxs), dtype=np.float32)
        base_goals = np.asarray(self.get_observations(base_goal_idxs), dtype=np.float32)
        return {
            'value_base_goals': base_goals,
            'value_base_offsets': base_offsets.astype(np.float32),
            'trans_v_split_observations': split_obs,
            'trans_v_left_goals': split_obs,
            'trans_v_right_observations': split_obs,
            'trans_v_right_goals': value_goals,
            'trans_v_valid_mask': tri_valid.astype(np.float32),
            'trans_v_split_offsets': (split_idxs - idxs).astype(np.float32),
            'q_goals': value_goals,
            'q_goal_offsets': (value_goal_idxs - idxs).astype(np.float32),
        }

    def augment(self, batch: dict, keys: list[str]) -> None:
        p_aug = self.config.get('p_aug')
        if not p_aug or float(p_aug) <= 0:
            return
        augment_batch_images(batch, keys, padding=3)

    def _validate_starts(self, idxs: np.ndarray) -> None:
        finals = self._lookup_finals(idxs)
        bad = idxs + self.full_chunk_horizon > finals
        if np.any(bad):
            raise ValueError('CriticSequenceDataset sampled starts crossing episode boundaries.')

    def sample(self, batch_size: int, idxs: np.ndarray | None = None, evaluation: bool = False) -> dict:
        if idxs is None:
            idxs = self.valid_starts[np.random.randint(len(self.valid_starts), size=batch_size)]
        idxs = np.asarray(idxs, dtype=np.int64)
        self._validate_starts(idxs)

        obs = np.asarray(self.get_observations(idxs), dtype=np.float32)
        actions_step = np.asarray(self.dataset['actions'][idxs], dtype=np.float32)
        next_obs = np.asarray(self.get_observations(idxs + 1), dtype=np.float32)

        full_idx = self._build_full_chunk_indices(idxs)
        full_chunk_actions_3d = np.asarray(self.dataset['actions'][full_idx], dtype=np.float32)  # [B, H_full, A]
        action_chunk_actions_3d = full_chunk_actions_3d[:, : self.action_chunk_horizon, :]
        full_chunk_actions = full_chunk_actions_3d.reshape(full_chunk_actions_3d.shape[0], -1)
        action_chunk_actions = action_chunk_actions_3d.reshape(action_chunk_actions_3d.shape[0], -1)

        value_goal_idxs = self.sample_trl_goals(idxs)
        value_goals = np.asarray(self.get_observations(value_goal_idxs), dtype=np.float32)
        full_chunk_rewards, full_chunk_next_observations, full_chunk_horizon, full_chunk_masks = (
            self._chunk_backup(
                idxs,
                value_goal_idxs,
                horizon=self.full_chunk_horizon,
                discount_pows=self.discount_pows,
            )
        )
        # Partial-window (action-chunk) backup quantities. Always emitted so the critic
        # can switch between full-chunk (DQC) and action-chunk (IQL) horizons without
        # re-sampling.
        action_chunk_rewards, action_chunk_next_observations, action_chunk_horizon_per_sample, action_chunk_masks = (
            self._chunk_backup(
                idxs,
                value_goal_idxs,
                horizon=self.action_chunk_horizon,
                discount_pows=self.action_discount_pows,
            )
        )
        valids = np.repeat(self.valids_template[None, :], idxs.shape[0], axis=0).astype(np.float32)

        batch = {
            'observations': obs,
            'actions': actions_step,
            'next_observations': next_obs,
            'value_goals': value_goals,
            'full_chunk_actions': full_chunk_actions,
            'action_chunk_actions': action_chunk_actions,
            'full_chunk_next_observations': full_chunk_next_observations,
            'full_chunk_rewards': full_chunk_rewards,
            'full_chunk_masks': full_chunk_masks,
            'full_chunk_horizon': full_chunk_horizon,
            'action_chunk_next_observations': action_chunk_next_observations,
            'action_chunk_rewards': action_chunk_rewards,
            'action_chunk_masks': action_chunk_masks,
            'action_chunk_horizon_per_sample': action_chunk_horizon_per_sample,
            'valids': valids,
        }

        trl_fields = self._sample_trl_fields(idxs, value_goal_idxs)
        batch.update(
            {
                'value_offsets': (value_goal_idxs - idxs).astype(np.float32),
                **trl_fields,
            }
        )

        if not evaluation:
            aug_keys = [
                'observations',
                'next_observations',
                'value_goals',
                'full_chunk_next_observations',
                'action_chunk_next_observations',
            ]
            aug_keys.extend(
                [
                    'value_base_goals',
                    'trans_v_split_observations',
                    'trans_v_left_goals',
                    'trans_v_right_observations',
                    'trans_v_right_goals',
                    'q_goals',
                ]
            )
            self.augment(batch, aug_keys)
        return batch
