from datetime import datetime
import os
import time

import gym.spaces as spaces
from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_pytorch.ppo import RolloutStorage
from .teacher_storage import TeacherRolloutStorage, get_teacher_loader

import wandb


def cycle(iterable):
  while True:
    for x in iterable:
      yield x


class DAPGTrainer:
  def __init__(
    self,
    vec_env,
    actor_critic_class,
    height_encoder,
    vis_encoder,
    num_transitions_per_env,
    num_learning_epochs,
    num_mini_batches,
    clip_param=0.2,
    gamma=0.998,
    lam=0.95,
    init_noise_std=1.0,
    value_loss_coef=1.0,
    bc_coef=1.0,
    entropy_coef=0.0,
    learning_rate=1e-3,
    max_grad_norm=0.5,
    use_clipped_value_loss=True,
    schedule="fixed",
    desired_kl=None,
    model_cfg=None,
    device='cpu',
    sampler='random',
    student_log_dir='student_run',
    is_testing=False,
    print_log=True,
    apply_reset=False,
    asymmetric=False,
    vidlogdir='video',
    vid_log_step=1000,
    log_video=False,
    storage_load_path=None,
  ):
      # We removed Pretrain stage Here for DAPG since the value is kind of hard to initialize
    if not isinstance(vec_env.observation_space, Space):
      raise TypeError("vec_env.observation_space must be a gym Space")
    if not isinstance(vec_env.state_space, Space):
      raise TypeError("vec_env.state_space must be a gym Space")
    if not isinstance(vec_env.action_space, Space):
      raise TypeError("vec_env.action_space must be a gym Space")
    self.observation_space = vec_env.observation_space
    self.action_space = vec_env.action_space
    self.state_space = vec_env.state_space

    self.device = device
    self.asymmetric = asymmetric

    self.desired_kl = desired_kl
    self.schedule = schedule
    self.step_size = learning_rate

    # PPO components
    self.vec_env = vec_env

    from copy import deepcopy
    vis_model_cfg = deepcopy(model_cfg)

    vis_model_cfg['encoder_params'] = model_cfg['vis_encoder_params']

    self.is_testing = is_testing
    self.vidlogdir = vidlogdir
    self.log_video = log_video
    self.vid_log_step = vid_log_step

    self.student_actor_critic = actor_critic_class(
      vis_encoder, self.observation_space.shape, self.state_space.shape, self.action_space.shape,
      init_noise_std, vis_model_cfg, asymmetric=asymmetric
    )
    self.student_actor_critic.to(self.device)

    self.visual_observation_space = spaces.Box(
      np.ones(self.vec_env.task.state_obs_size +
              self.vec_env.task.image_obs_size) * -np.Inf,
      np.ones(self.vec_env.task.state_obs_size +
              self.vec_env.task.image_obs_size) * np.Inf,
    )
    self.storage = RolloutStorage(
      self.vec_env.num_envs, num_transitions_per_env, self.visual_observation_space.shape,
      self.state_space.shape, self.action_space.shape, self.device, sampler
    )

    self.optimizer = optim.Adam(
      self.student_actor_critic.parameters(), lr=learning_rate
    )

    # PPO parameters
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.num_transitions_per_env = num_transitions_per_env
    self.value_loss_coef = value_loss_coef
    self.bc_coef = bc_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    # student Log
    self.student_log_dir = student_log_dir
    self.print_log = print_log
    self.writer = SummaryWriter(
      log_dir=self.student_log_dir, flush_secs=10)
    self.tot_timesteps = 0
    self.tot_time = 0
    self.is_testing = is_testing
    self.current_learning_iteration = 0

    self.apply_reset = apply_reset

    self.teacher_batch_size = self.vec_env.num_envs * self.num_transitions_per_env
    self.teacher_mini_batch_size = self.teacher_batch_size // num_mini_batches
    # self.teacher_batch_generator = self.teacher_storage.mini_batch_generator(
    #     self.teacher_mini_batch_size
    # )
    # self.teacher_batches = list(self.teacher_batch_generator)
    # self.teacher_batch_nums = len(self.teacher_batches)
    # self.teacher_batch_idx = 0
    self.storage_load_path = storage_load_path
    assert self.storage_load_path is not None

  def test_student(self, path):
    self.student_actor_critic.load_state_dict(torch.load(path))
    self.student_actor_critic.eval()

  def student_load(self, path):
    self.student_actor_critic.load_state_dict(torch.load(path))
    self.current_learning_iteration = int(
      path.split("_")[-1].split(".")[0])
    self.student_actor_critic.train()

  def save(self, path):
    torch.save(self.student_actor_critic.state_dict(), path)

  def get_expert_buffer(self):
    assert os.path.exists(self.storage_load_path)

    observation_list = []
    actions_list = []

    num_teacher_transitions = 0

    for root, dirs, files in os.walk(self.storage_load_path, topdown=False):
      # for name in files:
      #     print(os.path.join(root, name))
      for name in dirs:
        obs = torch.load(
          os.path.join(root, name, "obs.pt")
        )
        acts = torch.load(
          os.path.join(root, name, "acts.pt")
        )
        assert obs.shape[0] == acts.shape[0]
        num_teacher_transitions += obs.shape[0]
        observation_list.append(obs)
        actions_list.append(acts)

    self.num_teacher_transitions = num_teacher_transitions
    self.teacher_storage = TeacherRolloutStorage(
      self.vec_env.num_envs, num_teacher_transitions, self.visual_observation_space.shape,
      self.action_space.shape, 'cpu', 'sequential', create=False
    )
    self.teacher_storage.load_storage(observation_list, actions_list)

  def run(self, num_learning_iterations, log_interval=1):
    current_obs = self.vec_env.reset()
    current_states = self.vec_env.get_state()

    if self.is_testing:
      from pathlib import Path
      Path(self.vidlogdir).mkdir(parents=True, exist_ok=True)
      from vidgear.gears import WriteGear
      output_params = {"-vcodec": "libx264"}
      writer = WriteGear(
        output_filename=os.path.join(
          self.vidlogdir, 'Output_{}.mp4'.format(
            self.current_learning_iteration)
        ),
        logging=True, **output_params
      )
      step = 0
      while (not self.log_video or (step < self.vid_log_step)) and not self.vec_env.task.task_wrapper.test_finished:
        with torch.no_grad():
          if self.apply_reset:
            current_obs = self.vec_env.reset()
          # Compute the action
          actions = self.student_actor_critic.act_inference(
            current_obs)
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          log_img = self.vec_env.log_render()
          log_img = log_img.cpu().detach().numpy().astype(np.uint8)
          writer.write(log_img, rgb_mode=True)
          current_obs.copy_(next_obs)
        step += 1
      writer.close()
      return

    self.get_expert_buffer()
    self.teacher_loader = iter(cycle(get_teacher_loader(
      self.teacher_storage, self.teacher_mini_batch_size,
      num_workers=4, shuffle=True
    )))

    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(
      self.vec_env.num_envs, dtype=torch.float, device=self.device)
    cur_episode_length = torch.zeros(
      self.vec_env.num_envs, dtype=torch.float, device=self.device)

    reward_sum = []
    episode_length = []

    for it in range(self.current_learning_iteration, num_learning_iterations):
      start = time.time()
      ep_infos = []

      if self.apply_reset:
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
      # Rollout
      for _ in range(self.num_transitions_per_env):
        # Compute the action
        with torch.no_grad():
          actions, actions_log_prob, values, mu, sigma = self.student_actor_critic.act(
            current_obs, current_states)
          # actions = self.teacher_actor_critic.act_inference(
          #     current_height_obs)

        # Step the vec_environment
        next_obs, rews, dones, infos = self.vec_env.step(actions)
        next_states = self.vec_env.get_state()
        # Record the transition

        # Store Vision Obs
        self.storage.add_transitions(
          current_obs, current_states,
          actions, rews, dones, values, actions_log_prob, mu, sigma
        )
        current_obs.copy_(next_obs)
        current_states.copy_(next_states)
        # Book keeping
        ep_infos.append(infos)

        if self.print_log:
          cur_reward_sum[:] += rews
          cur_episode_length[:] += 1

          new_ids = (dones > 0).nonzero(as_tuple=False)
          reward_sum.extend(
            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
          episode_length.extend(
            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
          cur_reward_sum[new_ids] = 0
          cur_episode_length[new_ids] = 0

      if self.print_log:
        # reward_sum = [x[0] for x in reward_sum]
        # episode_length = [x[0] for x in episode_length]
        rewbuffer.extend(reward_sum)
        lenbuffer.extend(episode_length)

      _, _, last_values, _, _ = self.student_actor_critic.act(
        current_obs, current_states)
      stop = time.time()
      collection_time = stop - start

      mean_trajectory_length, mean_reward = self.storage.get_statistics()

      # Learning step
      start = stop
      self.storage.compute_returns(last_values, self.gamma, self.lam)
      mean_value_loss, mean_surrogate_loss, mean_bc_loss = self.update()
      mean_loss = self.update()
      self.storage.clear()
      stop = time.time()
      learn_time = stop - start
      if self.print_log:
        self.log(locals())
      if it % log_interval == 0:
        self.save(os.path.join(self.student_log_dir,
                                'model_{}.pt'.format(it)))
      ep_infos.clear()
    self.save(os.path.join(self.student_log_dir,
                            'model_{}.pt'.format(num_learning_iterations)))

  def log(self, locs, width=80, pad=35):
    self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
    self.tot_time += locs['collection_time'] + locs['learn_time']
    iteration_time = locs['collection_time'] + locs['learn_time']

    ep_string = f''
    if locs['ep_infos']:
      for key in locs['ep_infos'][0]:
        infotensor = torch.tensor([], device=self.device)
        for ep_info in locs['ep_infos']:
          infotensor = torch.cat(
            (infotensor, ep_info[key].to(self.device)))
        value = torch.mean(infotensor)
        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    mean_std = self.student_actor_critic.log_std.exp().mean()

    fps = int(self.num_transitions_per_env * self.vec_env.num_envs /
              (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    if len(locs['rewbuffer']) > 0:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                          'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'BC loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      wandb.log({
        'Value function loss:': locs['mean_value_loss'],
        'Surrogate loss:': locs['mean_surrogate_loss'],
        'BC loss:': locs['mean_bc_loss'],
        'Mean action noise std:': mean_std.item(),
        'Mean reward:': statistics.mean(locs['rewbuffer']),
        'Mean episode length:': statistics.mean(locs['lenbuffer']),
        'Mean reward/step:': locs['mean_reward'],
        'Mean episode length/episode:': locs['mean_trajectory_length']
      })
    else:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'BC loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      wandb.log({
        'Value function loss:': locs['mean_value_loss'],
        'Surrogate loss:': locs['mean_surrogate_loss'],
        'BC loss:': locs['mean_bc_loss'],
        'Mean action noise std:': mean_std.item(),
        'Mean reward/step:': locs['mean_reward'],
        'Mean episode length/episode:': locs['mean_trajectory_length']
      })

      log_string += ep_string
      log_string += (f"""{'-' * width}\n"""
                     f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                     f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                     f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                     f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                                  locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
      print(log_string)

  def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_bc_loss = 0

    batch = self.storage.mini_batch_generator(self.num_mini_batches)

    expert_batch_list = []
    for _ in range(self.num_mini_batches):
      teacher_batch = next(self.teacher_loader)
      teacher_batch[0] = teacher_batch[0].to(self.device)
      teacher_batch[1] = teacher_batch[1].to(self.device)
      expert_batch_list.append(teacher_batch)

    for epoch in range(self.num_learning_epochs):
      # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
      #        in self.storage.mini_batch_generator(self.num_mini_batches):

      for batch_idx, indices in enumerate(batch):

        obs_batch = self.storage.observations.view(
          -1, *self.storage.observations.size()[2:])[indices]

        if self.asymmetric:
          states_batch = self.storage.states.view(
            -1, *self.storage.states.size()[2:])[indices]
        else:
          states_batch = None

        actions_batch = self.storage.actions.view(
          -1, self.storage.actions.size(-1))[indices]
        target_values_batch = self.storage.values.view(-1, 1)[indices]
        returns_batch = self.storage.returns.view(-1, 1)[indices]
        old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[
          indices]
        advantages_batch = self.storage.advantages.view(-1, 1)[indices]
        old_mu_batch = self.storage.mu.view(-1,
                                            self.storage.actions.size(-1))[indices]
        old_sigma_batch = self.storage.sigma.view(
          -1, self.storage.actions.size(-1))[indices]

        actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.student_actor_critic.evaluate(
          obs_batch,
          states_batch,
          actions_batch
        )
        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':

          kl = torch.sum(
            sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
          kl_mean = torch.mean(kl)

          if kl_mean > self.desired_kl * 2.0:
            self.step_size = max(1e-5, self.step_size / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.step_size = min(1e-2, self.step_size * 1.5)

          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.step_size

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch -
                          torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                           1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
          value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                          self.clip_param)
          value_losses = (value_batch - returns_batch).pow(2)
          value_losses_clipped = (
            value_clipped - returns_batch).pow(2)
          value_loss = torch.max(
            value_losses, value_losses_clipped).mean()
        else:
          value_loss = (returns_batch - value_batch).pow(2).mean()

        teacher_obs_batch = expert_batch_list[batch_idx][0]
        teacher_actions_batch = expert_batch_list[batch_idx][1]

        student_actions = self.student_actor_critic.act_inference(
          teacher_obs_batch
        )

        bc_loss = torch.abs(student_actions - teacher_actions_batch).mean()

        loss = surrogate_loss + \
          self.value_loss_coef * value_loss - \
          self.entropy_coef * entropy_batch.mean() + \
          self.bc_coef * bc_loss

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
          self.student_actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_bc_loss += bc_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_bc_loss /= num_updates

    return mean_value_loss, mean_surrogate_loss, mean_bc_loss
