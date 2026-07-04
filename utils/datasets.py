import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def lookup_final_indices(terminal_locs: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Return terminal index for each transition index."""
    return terminal_locs[np.searchsorted(terminal_locs, idxs)]


def goal_final_indices(
    terminal_locs: np.ndarray,
    idxs: np.ndarray,
    max_goal_steps: int | None = None,
) -> np.ndarray:
    """Return the furthest same-trajectory index allowed for goal sampling."""
    finals = lookup_final_indices(terminal_locs, idxs)
    if max_goal_steps is None:
        return finals
    max_steps = int(max_goal_steps)
    if max_steps <= 0:
        return finals
    return np.minimum(finals, idxs + max_steps)


def gather_stacked_observations(
    observations: Any,
    idxs: np.ndarray,
    initial_locs: np.ndarray,
    frame_stack: int,
) -> Any:
    """Shared helper to gather frame-stacked observations."""
    initial_state_idxs = initial_locs[np.searchsorted(initial_locs, idxs, side='right') - 1]
    rets = []
    for i in reversed(range(int(frame_stack))):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], observations))
    return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


def augment_batch_images(batch: dict, keys: list[str], *, padding: int = 3) -> None:
    """Apply random-crop augmentation to image-like leaves in-place."""
    batch_size = len(batch[keys[0]])
    crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
    crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
    for key in keys:
        batch[key] = jax.tree_util.tree_map(
            lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
            batch[key],
        )


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - max_goal_steps: Optional cap on same-trajectory future goal distance.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = goal_final_indices(
            self.terminal_locs,
            idxs,
            self.config.get('max_goal_steps', None),
        )
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        augment_batch_images(batch, keys, padding=3)

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        return gather_stacked_observations(
            self.dataset['observations'],
            idxs,
            self.initial_locs,
            int(self.config['frame_stack']),
        )


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = lookup_final_indices(self.terminal_locs, idxs)
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        goal_final_state_idxs = goal_final_indices(
            self.terminal_locs,
            idxs,
            self.config.get('max_goal_steps', None),
        )
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, goal_final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (
                    np.minimum(idxs + 1, goal_final_state_idxs) * distances
                    + goal_final_state_idxs * (1 - distances)
                )
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)
        # Expose indices so subclasses (e.g. PathHGCDataset) can pad the trajectory
        # segment past the (clipped) goal without re-sampling. Cheap to carry along.
        batch['high_actor_goal_idxs'] = np.asarray(high_goal_idxs, dtype=np.int64)
        batch['high_actor_target_idxs'] = np.asarray(high_target_idxs, dtype=np.int64)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch


@dataclasses.dataclass
class PathHGCDataset(HGCDataset):
    """Extends :class:`HGCDataset` with a contiguous same-episode ``trajectory_segment``.

    Used by dynamics Phase-1 path supervision: for horizon ``K = subgoal_steps``, each batch row
    includes ``s_t, s_{t+1}, ..., s_{t+K}`` as ``trajectory_segment`` of shape ``(B, K+1, D)``.
    When ``idxs`` is omitted, starts ``t`` are resampled until ``t+K`` lies in the same episode
    (does not cross ``terminals``). ``high_actor_targets`` is overwritten to match
    ``trajectory_segment[:, -1]`` so the bridge endpoint and segment tail stay consistent.

    The optional ``clip_path_to_goal`` config flag controls behaviour near the episode goal
    (the dynamics-agent default is ``True``):

    - ``False``: segment endpoint is always ``s_{t+K}`` regardless of where
      the sampled goal ``s_{t_g}`` is. Subgoal_net teacher is ``s_{t+K}`` so it may "overshoot"
      past close goals.
    - ``True`` (default): per-row endpoint becomes ``s_{min(t+K, t_g)}``; segment tail past ``t_g`` is
      padded with ``s_{t_g}``. Both the dynamics bridge and the subgoal_net teacher then learn
      to "arrive and stay" at close goals, keeping inference subgoals in-distribution.

    Image ``p_aug`` is only consistent when augmentations do not reshape state vectors; for
    vector antmaze data the parent augment path is a no-op on 2D observations.
    """

    def __post_init__(self):
        super().__post_init__()
        self.path_horizon = int(self.config['subgoal_steps'])
        self.path_offsets = np.arange(self.path_horizon + 1, dtype=np.int32)[None, :]

        path_valids = np.zeros(self.size, dtype=np.float32)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            last_valid = int(end) - self.path_horizon
            if last_valid >= int(start):
                path_valids[int(start) : last_valid + 1] = 1.0
        (self.path_valid_idxs,) = np.nonzero(path_valids > 0)
        if len(self.path_valid_idxs) == 0:
            raise ValueError(
                'No valid path-supervision indices found. Reduce subgoal_steps or inspect episode lengths.'
            )

    def _sample_path_idxs(self, batch_size: int) -> np.ndarray:
        return self.path_valid_idxs[np.random.randint(len(self.path_valid_idxs), size=batch_size)]

    def _validate_segment_starts(self, idxs: np.ndarray, K: int) -> np.ndarray:
        """Validate that every provided start index admits a same-episode segment of length ``K+1``."""
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1:
            raise ValueError(f'PathHGCDataset expects 1D idxs, got shape {idxs.shape}.')
        if len(idxs) == 0:
            raise ValueError('PathHGCDataset received empty idxs.')
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise ValueError(f'PathHGCDataset idxs out of range for dataset size {self.size}.')

        finals = lookup_final_indices(self.terminal_locs, idxs)
        bad = idxs + K > finals
        if np.any(bad):
            first_bad = int(np.nonzero(bad)[0][0])
            raise ValueError(
                'PathHGCDataset idxs must stay within one episode for trajectory_segment: '
                f'idx={int(idxs[first_bad])}, K={K}, terminal={int(finals[first_bad])}.'
            )
        return idxs

    def validate_sample_batch(self, batch, K: int | None = None, atol: float = 1e-6) -> None:
        """Sanity-check key alignment for one sampled batch."""
        horizon = int(self.config['subgoal_steps']) if K is None else int(K)
        seg = np.asarray(batch['trajectory_segment'])
        obs = np.asarray(batch['observations'])
        tgt = np.asarray(batch['high_actor_targets'])
        nxt = np.asarray(batch['next_observations'])

        if seg.ndim != 3:
            raise ValueError(f'trajectory_segment must be rank-3, got shape {seg.shape}.')
        if seg.shape[1] != horizon + 1:
            raise ValueError(
                f'trajectory_segment second dim must be K+1={horizon + 1}, got {seg.shape[1]}.'
            )
        if not np.allclose(seg[:, 0], obs, atol=atol):
            raise ValueError('trajectory_segment[:, 0] does not match observations.')
        if not np.allclose(seg[:, -1], tgt, atol=atol):
            raise ValueError('trajectory_segment[:, -1] does not match high_actor_targets.')
        if not np.allclose(seg[:, 1], nxt, atol=atol):
            raise ValueError('trajectory_segment[:, 1] does not match next_observations.')
        if 'trajectory_indices' in batch:
            idxs = np.asarray(batch['trajectory_indices'])
            if idxs.shape != seg.shape[:2]:
                raise ValueError(
                    f'trajectory_indices shape must match trajectory_segment[:2], got {idxs.shape} vs {seg.shape[:2]}.'
                )
            diffs = np.diff(idxs, axis=1)
            if bool(self.config.get('clip_path_to_goal', False)):
                # clip_path mode pads segment past t_g; allowed step sizes are 1 (real)
                # or 0 (frozen at t_clip). 0-runs must be a contiguous suffix per row.
                if not np.all((diffs == 1) | (diffs == 0)):
                    raise ValueError(
                        'trajectory_indices step size must be 0 or 1 under clip_path_to_goal.'
                    )
                # First 0 in a row (if any) must mark the start of an all-0 suffix.
                first_zero = np.argmax(diffs == 0, axis=1)
                has_zero = np.any(diffs == 0, axis=1)
                for i in np.flatnonzero(has_zero):
                    if not np.all(diffs[i, first_zero[i]:] == 0):
                        raise ValueError(
                            'trajectory_indices padding must be a contiguous suffix per row.'
                        )
            elif not np.all(diffs == 1):
                raise ValueError('trajectory_indices must be contiguous with step size 1.')
            if 'trajectory_terminal_indices' in batch:
                terms = np.asarray(batch['trajectory_terminal_indices'])
                if np.any(idxs[:, -1] > terms):
                    raise ValueError('trajectory_segment crosses a terminal boundary.')

    def sample(self, batch_size, idxs=None, evaluation=False):
        K = self.path_horizon
        if idxs is None:
            idxs = self._sample_path_idxs(batch_size)
        else:
            idxs = self._validate_segment_starts(idxs, K)
        batch = super().sample(batch_size, idxs, evaluation)
        traj_indices = idxs[:, None] + self.path_offsets
        finals = lookup_final_indices(self.terminal_locs, idxs)

        clip_path = bool(self.config.get('clip_path_to_goal', False))
        if clip_path:
            # Per-row endpoint t_clip = min(t+K, t_g). Past t_clip we freeze the index so
            # trajectory_segment is padded with s_{t_clip} (= clipped high_actor_targets),
            # keeping bridge endpoint, subgoal_net teacher, and segment tail consistent.
            target_idxs = batch.get('high_actor_target_idxs')
            if target_idxs is None:
                raise RuntimeError(
                    'clip_path_to_goal=True requires HGCDataset.sample to expose '
                    'high_actor_target_idxs in the batch.'
                )
            t_clip = np.asarray(target_idxs, dtype=np.int64)
            seg_indices = np.minimum(traj_indices, t_clip[:, None])
        else:
            seg_indices = traj_indices

        traj = self.get_observations(seg_indices)
        batch['trajectory_segment'] = np.asarray(traj, dtype=np.float32)
        batch['trajectory_indices'] = np.asarray(seg_indices, dtype=np.int64)
        batch['trajectory_start_indices'] = np.asarray(idxs, dtype=np.int64)
        batch['trajectory_terminal_indices'] = np.asarray(finals, dtype=np.int64)
        # Override high_actor_targets to match segment endpoint. clip_path=False keeps
        # s_{t+K}; clip_path=True yields s_{min(t+K, t_g)}.
        batch['high_actor_targets'] = np.asarray(traj[:, K], dtype=np.float32)
        return batch


@dataclasses.dataclass
class ChunkHGCDataset(HGCDataset):
    """Hierarchical GC dataset with real-trajectory chunk targets.

    For each sampled transition index ``t``, this dataset additionally builds:

    * ``local_plan_context`` from real future states in the same episode.
    * ``action_chunks`` from real actions ``a_t .. a_{t+H_pi-1}``.
    * ``chunk_future_observations`` containing ``s_{t+1} .. s_{t+H_pi}```.

    The chunk actor is therefore trained only on offline trajectory segments,
    never on planner-generated synthetic transitions.
    """

    def __post_init__(self):
        super().__post_init__()

        self.chunk_context_horizon = int(self.config['chunk_context_horizon'])
        self.chunk_policy_horizon = int(self.config['chunk_policy_horizon'])
        self.chunk_target_horizon = max(self.chunk_context_horizon, self.chunk_policy_horizon)
        self.low_goal_slice = np.asarray(self.config['low_goal_slice'], dtype=np.int32)
        self.chunk_use_relative_context = bool(self.config.get('chunk_use_relative_context', True))

        observations = np.asarray(self.dataset['observations'])
        actions = np.asarray(self.dataset['actions'])
        self._obs_np = observations
        self._actions_np = actions
        if observations.ndim != 2:
            raise ValueError('ChunkHGCDataset currently expects 2D state observations.')
        if actions.ndim != 2:
            raise ValueError('ChunkHGCDataset currently expects 2D continuous actions.')
        if len(self.low_goal_slice) < 1:
            raise ValueError('low_goal_slice must contain at least one observation index.')

        chunk_valids = np.zeros(self.size, dtype=np.float32)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            last_valid = int(end) - self.chunk_target_horizon
            if last_valid >= int(start):
                chunk_valids[int(start) : last_valid + 1] = 1.0
        (self.chunk_valid_idxs,) = np.nonzero(chunk_valids > 0)
        if len(self.chunk_valid_idxs) == 0:
            raise ValueError(
                'No valid chunk indices found. Reduce chunk_context_horizon / chunk_policy_horizon or inspect episodes.'
            )

    def _sample_chunk_idxs(self, batch_size):
        return self.chunk_valid_idxs[np.random.randint(len(self.chunk_valid_idxs), size=batch_size)]

    def _slice_observations(self, start_idxs, horizon):
        offsets = np.arange(1, horizon + 1, dtype=np.int32)[None, :]
        obs_idxs = start_idxs[:, None] + offsets
        return self._obs_np[obs_idxs]

    def _slice_actions(self, start_idxs, horizon):
        offsets = np.arange(horizon, dtype=np.int32)[None, :]
        action_idxs = start_idxs[:, None] + offsets
        return self._actions_np[action_idxs]

    def _build_local_plan_context(self, observations, future_observations):
        goal_obs = observations[:, self.low_goal_slice]
        future_goal_obs = future_observations[:, :, self.low_goal_slice]
        if self.chunk_use_relative_context:
            context = future_goal_obs - goal_obs[:, None, :]
        else:
            context = future_goal_obs
        return context.reshape(context.shape[0], -1)

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch with real chunk targets from the same trajectory."""
        if idxs is None:
            idxs = self._sample_chunk_idxs(batch_size)
        else:
            idxs = np.asarray(idxs, dtype=np.int32)

        batch = super().sample(batch_size, idxs=idxs, evaluation=evaluation)

        observations = np.asarray(batch['observations'])
        plan_future_observations = self._slice_observations(idxs, self.chunk_context_horizon)
        chunk_future_observations = self._slice_observations(idxs, self.chunk_policy_horizon)
        action_chunks = self._slice_actions(idxs, self.chunk_policy_horizon)

        batch['local_plan_context'] = self._build_local_plan_context(observations, plan_future_observations)
        batch['local_plan_endpoint'] = plan_future_observations[:, -1]
        batch['action_chunks'] = action_chunks.reshape(action_chunks.shape[0], -1)
        batch['action_chunk_actions'] = action_chunks
        batch['chunk_future_observations'] = chunk_future_observations

        return batch
