import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader, TensorDataset


class TeacherRolloutStorage(Dataset):

    def __init__(
        self, num_envs, num_transitions_per_env,
        obs_shape, actions_shape, 
        device='cpu', sampler='random', create=True
    ):

        self.device = device
        self.sampler = sampler

        if create:
            # Core
            self.observations = torch.zeros(num_transitions_per_env * num_envs, *obs_shape, device=self.device)
            self.rewards = torch.zeros(num_transitions_per_env * num_envs, 1, device=self.device)
            self.actions = torch.zeros(num_transitions_per_env * num_envs, *actions_shape, device=self.device)
            self.dones = torch.zeros(num_transitions_per_env * num_envs, 1, device=self.device).byte()

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def load_storage(self, obs_list, acts_list):
        self.observations = torch.vstack(obs_list)
        self.actions = torch.vstack(acts_list)

    def add_transitions(
        self, observations, actions, rewards, dones):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step * self.num_envs : (self.step + 1) * self.num_envs].copy_(observations)
        self.actions[self.step * self.num_envs : (self.step + 1) * self.num_envs].copy_(actions)
        self.rewards[self.step * self.num_envs : (self.step + 1) * self.num_envs].copy_(rewards.view(-1, 1))
        self.dones[self.step * self.num_envs : (self.step + 1) * self.num_envs].copy_(dones.view(-1, 1))

        self.step += 1

    def clear(self):
        self.step = 0

    def get_statistics(self):
        done = self.dones.cpu().reshape(self.num_transitions_per_env, self.num_envs, 1)
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def __getitem__(self, index):
        return {
            "obs": self.observations[index],
            "acts": self.actions[index]
        }

    def __len__(self):
        return self.step * self.num_envs


def get_teacher_loader(storage, batch_size, num_workers, shuffle):
  dataset = TensorDataset(
    storage.observations,
    storage.actions
  )
  return DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle,
    num_workers=num_workers, prefetch_factor=2, persistent_workers=True
  )
