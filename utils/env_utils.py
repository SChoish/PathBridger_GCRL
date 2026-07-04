import collections
import glob
import os
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

import ogbench
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(dataset_name, frame_stack=None, **env_kwargs):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.
        **env_kwargs: Forwarded to ``ogbench.make_env_and_datasets`` / ``gymnasium.make`` (e.g. ``render_mode``).

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """
    dataset_dir = env_kwargs.pop('dataset_dir', None)
    expanded_dataset_dir = os.path.expanduser(dataset_dir) if dataset_dir else None

    if expanded_dataset_dir and os.path.isdir(expanded_dataset_dir):
        train_shards = sorted(glob.glob(os.path.join(expanded_dataset_dir, '*.npz')))
        train_shards = [p for p in train_shards if not p.endswith('-val.npz')]
        val_shards = sorted(glob.glob(os.path.join(expanded_dataset_dir, '*-val.npz')))
    else:
        train_shards = []
        val_shards = []

    if train_shards and val_shards:
        env = ogbench.make_env_and_datasets(dataset_name, env_only=True, **env_kwargs)

        def _merge_shards(paths: list[str]) -> dict:
            merged: dict[str, np.ndarray] = {}
            for shard_path in paths:
                shard = ogbench.load_dataset(shard_path, compact_dataset=True)
                if not merged:
                    merged = {k: np.asarray(v) for k, v in shard.items()}
                else:
                    for k, v in shard.items():
                        merged[k] = np.concatenate([merged[k], np.asarray(v)], axis=0)
            return merged

        train_dataset = Dataset.create(**_merge_shards(train_shards))
        val_dataset = Dataset.create(**_merge_shards(val_shards))
    else:
        # Fall back to OGBench's default dataset resolution when no explicit directory is provided.
        ogbench_kwargs = dict(env_kwargs)
        if expanded_dataset_dir is not None:
            ogbench_kwargs['dataset_dir'] = expanded_dataset_dir

        # Use compact dataset to save memory.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            dataset_name, compact_dataset=True, **ogbench_kwargs
        )
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()

    return env, train_dataset, val_dataset
