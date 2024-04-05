from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple("Batch", "trajectories conditions")
ValueBatch = namedtuple("ValueBatch", "trajectories conditions values")


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()
        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.indices = self.make_indices(fields.path_lengths, horizon)
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def make_indices(self, path_lengths, horizon):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
                if max_start <= 0:
                    print("path_length: ", path_length)
                    print("horizon: ", horizon)
                    print("max_start: ", max_start)
                    raise ValueError("The selected horizon might be larger than the path_length")
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))

        indices = np.array(indices)

        return indices

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch


class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        """
        condition on both the current observation and the last observation in the plan
        """
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    """
    adds a value field to the datapoints for training the value function
    """

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields["rewards"][path_ind, start:]
        discounts = self.discounts[: len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch


class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        env="hopper-medium-replay",
        horizon=64,
        normalizer="LimitsNormalizer",
        preprocess_fns=[],
        max_path_length=None,
        max_n_episodes=10000,
        termination_penalty=0,
        use_padding=True,
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.use_padding = use_padding
        self.max_path_length = max_path_length
        itr = sequence_dataset(env, self.preprocess_fn)
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()
        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields["path_lengths"]
        )
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=["observations", "actions"]):
        """
        normalize fields that will be predicted by the diffusion model
        """
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f"normed_{key}"] = normed.reshape(
                self.n_episodes, self.max_path_length, -1
            )

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """
        return {0: observations[0], len(observations) - 1: observations[-1]}

    def __len__(self):
        return self.fields.observations.shape[0]

    def __getitem__(self, idx, eps=1e-4):
        observations = self.fields.normed_observations[idx, : self.horizon]
        actions = self.fields.normed_actions[idx, : self.horizon]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
