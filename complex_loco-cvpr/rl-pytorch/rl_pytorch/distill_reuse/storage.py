import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
from typing import List, Tuple
from torch import Tensor


class LMRolloutStorage:

  def __init__(
      self, num_envs, num_transitions_per_env, obs_shape, teacher_obs_shape,
      actions_shape,
      device='cpu', sampler='sequential'#, with_teacher_mask=False
  ):

    self.device = device
    self.sampler = sampler

    # Core
    # self.observations = torch.zeros(
    #   num_transitions_per_env, num_envs, *obs_shape, device=self.device)
    self.privileged_observations = torch.zeros(
        num_transitions_per_env, num_envs, *teacher_obs_shape, device=self.device)
    self.non_privileged_observations = torch.zeros(
        num_transitions_per_env, num_envs, *obs_shape, device=self.device)
    self.rewards = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.actions = torch.zeros(
        num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    self.dones = torch.zeros(num_transitions_per_env,
                             num_envs, 1, device=self.device)
    self.prev_dones = torch.zeros(num_transitions_per_env,
                                  num_envs, 1, device=self.device)
    # For PPO
    self.actions_log_prob = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.values = torch.zeros(num_transitions_per_env,
                              num_envs, 1, device=self.device)
    self.returns = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.advantages = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.mu = torch.zeros(num_transitions_per_env, num_envs,
                          *actions_shape, device=self.device)
    self.sigma = torch.zeros(num_transitions_per_env,
                             num_envs, *actions_shape, device=self.device)

    # if with_teacher_mask:
    #   self.teacher_masks = torch.zeros(
    #       num_transitions_per_env, num_envs, 1, device=self.device
    #   )

    # self.teacher_actions = torch.zeros(
    #     num_transitions_per_env,
    #     num_envs, *actions_shape, device=self.device)
    # self.teacher_features = torch.zeros(
    #     num_transitions_per_env,
    #     num_envs, teacher_feature_shape, device=self.device)
    self.num_transitions_per_env = num_transitions_per_env
    self.num_envs = num_envs

    self.step = 0

  def add_transitions(
      self,
      # observations,
      non_privileged_observations, privileged_observations,
      actions, rewards,
      dones, prev_dones, values, actions_log_prob,
      mu, sigma,
      # teacher_actions, teacher_features, teacher_masks=None
  ):
    if self.step >= self.num_transitions_per_env:
      raise AssertionError("Rollout buffer overflow")

    # self.observations[self.step].copy_(observations)
    self.non_privileged_observations[self.step].copy_(
        non_privileged_observations)
    self.privileged_observations[self.step].copy_(privileged_observations)
    self.actions[self.step].copy_(actions)
    self.rewards[self.step].copy_(rewards.view(-1, 1))
    self.dones[self.step].copy_(dones.view(-1, 1))
    self.prev_dones[self.step].copy_(prev_dones.view(-1, 1))
    self.values[self.step].copy_(values)
    self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
    self.mu[self.step].copy_(mu)
    self.sigma[self.step].copy_(sigma)

    # self.teacher_actions[self.step].copy_(teacher_actions)
    # self.teacher_features[self.step].copy_(teacher_features)

    # if teacher_masks is not None:
    #   self.teacher_masks[self.step].copy_(teacher_masks.view(-1, 1))

    self.step += 1

  def clear(self):
    self.step = 0

  def compute_returns(self, last_values, gamma, lam):
    advantage = 0
    for step in reversed(range(self.num_transitions_per_env)):
      if step == self.num_transitions_per_env - 1:
        next_values = last_values
      else:
        next_values = self.values[step + 1]
      next_is_not_terminal = 1.0 - self.dones[step].float()
      delta = self.rewards[step] + next_is_not_terminal * \
          gamma * next_values - self.values[step]
      advantage = delta + next_is_not_terminal * gamma * lam * advantage
      self.returns[step] = advantage + self.values[step]

    # Compute and normalize the advantages
    self.advantages = self.returns - self.values
    self.advantages = (self.advantages - self.advantages.mean()
                       ) / (self.advantages.std() + 1e-8)

  def get_statistics(self):
    done = self.dones.cpu()
    done[-1] = 1
    flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor(
        [-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
    trajectory_lengths = (done_indices[1:] - done_indices[:-1])
    return trajectory_lengths.float().mean(), self.rewards.mean()

  def mini_batch_generator(self, num_mini_batches):
    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    if self.sampler == "sequential":
      # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
      # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
      subset = SequentialSampler(range(batch_size))
    elif self.sampler == "random":
      subset = SubsetRandomSampler(range(batch_size))

    batch = BatchSampler(subset, mini_batch_size, drop_last=True)
    return batch
