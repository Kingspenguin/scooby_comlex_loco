from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl_pytorch.ppo import RolloutStorage

import wandb


class DistillTrainer:
  def __init__(
      self,
      vec_env,
      actor_critic_class,
      height_encoder,
      vis_encoder,
      num_transitions_per_env,
      num_learning_epochs,
      num_mini_batches,
      gamma=0.998,
      lam=0.95,
      init_noise_std=0.05,
      learning_rate=1e-3,
      max_grad_norm=0.5,
      schedule="fixed",
      model_cfg=None,
      device='cpu',
      sampler='sequential',
      teacher_log_dir='run',
      student_log_dir='student_run',
      is_testing=False,
      print_log=True,
      apply_reset=False,
      asymmetric=False,
      teacher_resume="None",
      vidlogdir='video',
      vid_log_step=1000,
      log_video=False
  ):
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

    self.schedule = schedule
    self.step_size = learning_rate

    # PPO components
    self.vec_env = vec_env

    from copy import deepcopy
    vis_model_cfg = deepcopy(model_cfg)
    height_model_cfg = deepcopy(model_cfg)

    vis_model_cfg['encoder_params'] = model_cfg['vis_encoder_params']
    height_model_cfg['encoder_params'] = model_cfg['height_encoder_params']

    self.is_testing = is_testing
    self.vidlogdir = vidlogdir
    self.log_video = log_video
    self.vid_log_step = vid_log_step
    if not self.is_testing:
      self.teacher_actor_critic = actor_critic_class(
        height_encoder, self.observation_space.shape, self.state_space.shape, self.action_space.shape,
        init_noise_std, height_model_cfg, asymmetric=asymmetric
      )
      self.teacher_actor_critic.to(self.device)

    self.student_actor_critic = actor_critic_class(
      vis_encoder, self.observation_space.shape, self.state_space.shape, self.action_space.shape,
      init_noise_std, vis_model_cfg, asymmetric=asymmetric
    )

    self.student_actor_critic.to(self.device)
    self.storage = RolloutStorage(
      self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
      self.state_space.shape, self.action_space.shape, self.device, sampler
    )
    self.optimizer = optim.Adam(
      self.student_actor_critic.parameters(), lr=learning_rate)

    # PPO parameters
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.num_transitions_per_env = num_transitions_per_env
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm

    # Log
    self.teacher_log_dir = teacher_log_dir
    # student Log
    self.student_log_dir = student_log_dir
    self.print_log = print_log
    self.writer = SummaryWriter(log_dir=self.student_log_dir, flush_secs=10)
    self.tot_timesteps = 0
    self.tot_time = 0
    self.is_testing = is_testing
    self.current_learning_iteration = 0

    self.apply_reset = apply_reset
    self.teacher_resume = teacher_resume
    assert teacher_resume is not None

  def test_teacher(self, path):
    self.teacher_actor_critic.load_state_dict(torch.load(path))
    self.teacher_actor_critic.eval()

  def test_student(self, path):
    self.student_actor_critic.load_state_dict(torch.load(path))
    self.student_actor_critic.eval()

  def teacher_load(self, path):
    self.teacher_actor_critic.load_state_dict(torch.load(path))
    # self.current_learning_iteration = int(
    #     path.split("_")[-1].split(".")[0])
    self.teacher_actor_critic.eval()

  def student_load(self, path):
    self.student_actor_critic.load_state_dict(torch.load(path))
    self.current_learning_iteration = int(
      path.split("_")[-1].split(".")[0])
    self.student_actor_critic.train()

  def save(self, path):
    torch.save(self.student_actor_critic.state_dict(), path)

  def split_height_vision_obs(self, obs_batch):
    # print(obs_batch.shape)
    state = obs_batch[:, :self.vec_env.task.state_obs_size]
    vis = obs_batch[
      :, self.vec_env.task.state_obs_size: self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size
    ]
    height = obs_batch[
      :,
      self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size: self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size + self.vec_env.task.height_map_obs_size
    ]
    height_obs = torch.cat([
      state, height
    ], dim=1)
    vision_obs = torch.cat([
      state, vis
    ], dim=1)
    # print(height_obs.shape)
    # print(vision_obs.shape)
    return height_obs, vision_obs

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
          _, current_vis_obs = self.split_height_vision_obs(current_obs)
          actions = self.student_actor_critic.act_inference(current_vis_obs)
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          if self.log_video:
            log_img = self.vec_env.log_render()
            log_img = log_img.cpu().detach().numpy().astype(np.uint8)
          # print(log_img)
          # import matplotlib.pyplot as plt
          # plt.imshow(log_img)
          # plt.pause(0.01)
          writer.write(log_img, rgb_mode=True)
          current_obs.copy_(next_obs)
        step += 1
      writer.close()
    else:
      self.teacher_load(
        "{}/model_{}.pt".format(self.teacher_log_dir, self.teacher_resume))
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

        # Rollout
        for _ in range(self.num_transitions_per_env):
          if self.apply_reset:
            current_obs = self.vec_env.reset()
            current_states = self.vec_env.get_state()

          current_height_obs, current_vis_obs = self.split_height_vision_obs(
            current_obs)
          # Compute the action
          with torch.no_grad():
            actions, actions_log_prob, values, mu, sigma = self.teacher_actor_critic.act(
              current_height_obs, current_states)
            actions = self.teacher_actor_critic.act_inference(
              current_height_obs)

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

        stop = time.time()
        collection_time = stop - start

        mean_trajectory_length, mean_reward = self.storage.get_statistics()

        # Learning step
        start = stop

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
        self.writer.add_scalar('Episode/' + key, value, locs['it'])
        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    mean_std = self.teacher_actor_critic.log_std.exp().mean()

    self.writer.add_scalar('Loss/BC',
                           locs['mean_loss'], locs['it'])
    self.writer.add_scalar('Policy/mean_noise_std',
                           mean_std.item(), locs['it'])
    if len(locs['rewbuffer']) > 0:
      self.writer.add_scalar(
        'Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
      self.writer.add_scalar(
        'Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
      self.writer.add_scalar(
        'Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
      self.writer.add_scalar(
        'Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

    self.writer.add_scalar('Train2/mean_reward/step',
                           locs['mean_reward'], locs['it'])
    self.writer.add_scalar(
      'Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

    fps = int(self.num_transitions_per_env * self.vec_env.num_envs /
              (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    if len(locs['rewbuffer']) > 0:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'BC loss:':>{pad}} {locs['mean_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      wandb.log({
        'BC loss:': locs['mean_loss'],
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
                    f"""{'BC loss:':>{pad}} {locs['mean_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      wandb.log({
        'BC loss:': locs['mean_loss'],
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
    mean_loss = 0

    batch = self.storage.mini_batch_generator(self.num_mini_batches)
    for epoch in range(self.num_learning_epochs):
      # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
      #        in self.storage.mini_batch_generator(self.num_mini_batches):

      for indices in batch:
        # obs for vision
        obs_batch = self.storage.observations.view(
          -1, *self.storage.observations.size()[2:])[indices]

        _, vis_obs_batch = self.split_height_vision_obs(obs_batch)

        actions_batch = self.storage.actions.view(
          -1, self.storage.actions.size(-1))[indices]

        student_actions = self.student_actor_critic.act_inference(
          vis_obs_batch
        )

        loss = torch.abs(student_actions - actions_batch).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
          self.student_actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_loss += loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_loss /= num_updates

    return mean_loss
