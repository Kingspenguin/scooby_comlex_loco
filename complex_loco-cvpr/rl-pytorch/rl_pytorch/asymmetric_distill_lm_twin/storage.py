import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler
from typing import List, Tuple
from torch import Tensor


class LMRolloutStorage:

  def __init__(
      self, num_envs, num_transitions_per_env,
      obs_shape, teacher_obs_shape, cpg_phase_info_shape,
      actions_shape, teacher_feature_shape, cpg_action_shape,
      device='cpu', sampler='sequential', with_cpg_mask=False
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

    self.cpg_phase_infos = torch.zeros(
        num_transitions_per_env, num_envs, *cpg_phase_info_shape, device=self.device)

    self.rewards = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)

    self.pd_actions = torch.zeros(
        num_transitions_per_env, num_envs, *actions_shape, device=self.device)
    self.cpg_actions = torch.zeros(
        num_transitions_per_env, num_envs, *cpg_action_shape, device=self.device)

    self.dones = torch.zeros(num_transitions_per_env,
                             num_envs, 1, device=self.device)
    self.prev_dones = torch.zeros(num_transitions_per_env,
                                  num_envs, 1, device=self.device)
    # For PPO
    self.pd_actions_log_prob = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.cpg_actions_log_prob = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)

    self.values = torch.zeros(num_transitions_per_env,
                              num_envs, 1, device=self.device)
    self.returns = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)
    self.advantages = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=self.device)

    self.pd_mu = torch.zeros(num_transitions_per_env, num_envs,
                             *actions_shape, device=self.device)
    self.pd_sigma = torch.zeros(num_transitions_per_env,
                                num_envs, *actions_shape, device=self.device)

    self.cpg_mu = torch.zeros(num_transitions_per_env, num_envs,
                              *cpg_action_shape, device=self.device)
    self.cpg_sigma = torch.zeros(num_transitions_per_env,
                                 num_envs, *cpg_action_shape, device=self.device)

    if with_cpg_mask:
      self.cpg_masks = torch.zeros(
          num_transitions_per_env, num_envs, 1, device=self.device
      )

    self.teacher_pd_actions = torch.zeros(
        num_transitions_per_env,
        num_envs, *actions_shape, device=self.device)

    self.pd_actions_from_cpg = torch.zeros(
        num_transitions_per_env,
        num_envs, *actions_shape, device=self.device)

    self.teacher_cpg_actions = torch.zeros(
        num_transitions_per_env,
        num_envs, *cpg_action_shape, device=self.device)

    self.teacher_features = torch.zeros(
        num_transitions_per_env,
        num_envs, teacher_feature_shape, device=self.device)

    self.num_transitions_per_env = num_transitions_per_env
    self.num_envs = num_envs

    self.step = 0

  def add_transitions(
      self,
      # observations,
      non_privileged_observations, privileged_observations, cpg_phase_infos,
      pd_actions, cpg_actions,
      rewards,
      dones, prev_dones, values,
      pd_actions_log_prob, pd_mu, pd_sigma,
      cpg_actions_log_prob, cpg_mu, cpg_sigma,
      teacher_pd_actions, teacher_cpg_actions, teacher_features,
      pd_actions_from_cpg,
      cpg_masks=None
  ):
    if self.step >= self.num_transitions_per_env:
      raise AssertionError("Rollout buffer overflow")

    # self.observations[self.step].copy_(observations)
    self.non_privileged_observations[self.step].copy_(
        non_privileged_observations)
    self.privileged_observations[self.step].copy_(privileged_observations)
    self.cpg_phase_infos[self.step].copy_(cpg_phase_infos)

    self.pd_actions[self.step].copy_(pd_actions)
    self.cpg_actions[self.step].copy_(cpg_actions)

    self.rewards[self.step].copy_(rewards.view(-1, 1))
    self.dones[self.step].copy_(dones.view(-1, 1))
    self.prev_dones[self.step].copy_(prev_dones.view(-1, 1))
    self.values[self.step].copy_(values)

    self.pd_actions_log_prob[self.step].copy_(pd_actions_log_prob.view(-1, 1))
    self.pd_mu[self.step].copy_(pd_mu)
    self.pd_sigma[self.step].copy_(pd_sigma)

    self.cpg_actions_log_prob[self.step].copy_(
        cpg_actions_log_prob.view(-1, 1))
    self.cpg_mu[self.step].copy_(cpg_mu)
    self.cpg_sigma[self.step].copy_(cpg_sigma)

    self.teacher_pd_actions[self.step].copy_(teacher_pd_actions)
    self.teacher_cpg_actions[self.step].copy_(teacher_cpg_actions)
    self.teacher_features[self.step].copy_(teacher_features)

    self.pd_actions_from_cpg[self.step].copy_(pd_actions_from_cpg)

    if cpg_masks is not None:
      self.cpg_masks[self.step].copy_(cpg_masks.view(-1, 1))

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
    batch_size = self.num_envs
    mini_batch_size = batch_size // num_mini_batches
    subset = SubsetRandomSampler(range(batch_size))
    batch = BatchSampler(subset, mini_batch_size, drop_last=True)
    return batch
