from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .storage import RolloutStorage

import wandb


class AsymmetricDistillTrainer:
  def __init__(
      self,
      vec_env,
      actor_critic_class,
      teacher_encoder,
      student_encoder,
      num_transitions_per_env,
      num_learning_epochs,
      num_mini_batches,
      clip_param=0.2,
      gamma=0.998,
      lam=0.95,
      init_noise_std=1.0,
      value_loss_coef=1.0,
      bc_loss_coef=1.0,
      entropy_coef=0.0,
      learning_rate=1e-3,
      max_grad_norm=0.5,
      use_clipped_value_loss=True,
      schedule="fixed",
      desired_kl=None,
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
      log_video=False,
      vid_log_step=500,
      local=False,
      learn_value_by_self=False,
      mask_base_vel=False,
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
    self.local = local
    self.learn_value_by_self = learn_value_by_self
    self.device = device
    del asymmetric
    self.desired_kl = desired_kl

    self.schedule = schedule
    self.step_size = learning_rate

    # PPO parameters
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.num_transitions_per_env = num_transitions_per_env
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.bc_loss_coef = bc_loss_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    # PPO components
    self.vec_env = vec_env

    from copy import deepcopy
    teacher_model_cfg = deepcopy(model_cfg)
    student_model_cfg = deepcopy(model_cfg)

    teacher_model_cfg['encoder_params'] = model_cfg['teacher_encoder_params']
    student_model_cfg['encoder_params'] = model_cfg['student_encoder_params']

    if "student_pi_hid_sizes" in model_cfg:
      student_model_cfg['pi_hid_sizes'] = model_cfg['student_pi_hid_sizes']
    student_model_cfg["encoder_params"]["state_dim"] = self.vec_env.task.state_obs_size - \
      self.vec_env.task.privilege_info_len

    self.is_testing = is_testing
    self.vidlogdir = vidlogdir
    self.log_video = log_video
    self.vid_log_step = vid_log_step
    self.mask_base_vel = mask_base_vel
    if not self.is_testing:
      # Teacher must not using asymmetric training
      # Teacher uses privilege information
      self.teacher_actor_critic = actor_critic_class(
        teacher_encoder, self.state_space.shape, self.state_space.shape, self.action_space.shape,
        init_noise_std, teacher_model_cfg, asymmetric=False
      )
      self.teacher_actor_critic.to(self.device)
    # Student must use asymmetric training
    # Student uses teacher's value model with itself's policy
    # Student uses teacher's policy for behavior cloning
    self.student_actor_critic = actor_critic_class(
      student_encoder, self.observation_space.shape, self.state_space.shape, self.action_space.shape,
      init_noise_std, student_model_cfg, asymmetric=True
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

  def test_teacher(self, path, device='cuda:0'):
    self.teacher_actor_critic.load_state_dict(
      torch.load(path, map_location=device))
    self.teacher_actor_critic.eval()

  def test_student(self, path, device='cuda:0'):
    self.student_actor_critic.load_state_dict(
      torch.load(path, map_location=device))
    self.student_actor_critic.eval()

  def teacher_load(self, path, device='cuda:0'):
    self.teacher_actor_critic.load_state_dict(
      torch.load(path, map_location=device))
    self.teacher_actor_critic.eval()

  def teacher_gives_student_value_function(self):
    self.student_actor_critic.critic.load_state_dict(
      self.teacher_actor_critic.critic.state_dict())

  def student_load(self, path, device='cuda:0'):
    self.student_actor_critic.load_state_dict(
      torch.load(path, map_location=device))
    self.current_learning_iteration = int(
      path.split("_")[-1].split(".")[0])
    self.student_actor_critic.train()

  def save(self, path):
    torch.save(self.student_actor_critic.state_dict(), path)

  def split_obs(self, obs_batch):

    # Obs -> [state, privilege information, vis, height]
    state = obs_batch[:, :self.vec_env.task.state_obs_size -
                      self.vec_env.task.privilege_info_len]
    privilege_info = obs_batch[:, self.vec_env.task.state_obs_size -
                               self.vec_env.task.privilege_info_len: self.vec_env.task.state_obs_size]
    vis = obs_batch[
      :, self.vec_env.task.state_obs_size: self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size
    ]
    height = obs_batch[
      :,
      self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size: self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size + self.vec_env.task.height_map_obs_size
    ]
    privilege_obs = torch.cat([
      state, privilege_info, height
    ], dim=1)
    non_privilege_obs = torch.cat([
      state, vis
    ], dim=1)

    # print("state shape: ", state.shape)
    # print("privilege_info shape: ", privilege_info.shape)
    # print("vis shape: ", vis.shape)
    # print("height shape: ", height.shape)
    # print("privilege_obs shape: ", privilege_obs.shape)
    # print("non_privilege_obs shape: ", non_privilege_obs.shape)
    if self.mask_base_vel:
      non_privilege_obs[..., :3] = 0
    return privilege_obs, non_privilege_obs

  def run(self, num_learning_iterations, log_interval=1):
    current_obs = self.vec_env.reset()
    current_states = self.vec_env.get_state()

    if self.is_testing:
      from pathlib import Path
      Path(self.vidlogdir).mkdir(parents=True, exist_ok=True)
      # from vidgear.gears import WriteGear
      # output_params = {"-vcodec": "libx264"}
      # writer = WriteGear(
      #   output_filename=os.path.join(
      #     self.vidlogdir, 'Output_{}.mp4'.format(
      #       self.current_learning_iteration)
      #   ),
      #   logging=True, **output_params
      # )
      import imageio
      frames = []
      step = 0
      while (not self.log_video or (step < self.vid_log_step)) and not self.vec_env.task.task_wrapper.test_finished:
        with torch.no_grad():
          if self.apply_reset:
            current_obs = self.vec_env.reset()
          # Compute the action
          _, non_privilege_obs = self.split_obs(current_obs)
          actions = self.student_actor_critic.act_inference(non_privilege_obs)
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          if self.log_video:
            log_img = self.vec_env.log_render()
            log_img = log_img.cpu().detach().numpy().astype(np.uint8)
            if self.vec_env.task.get_image:
              camera_img = self.vec_env.task.log_robot_cameras()
              camera_img = camera_img.cpu().detach().numpy().astype(np.uint8)
              if camera_img.shape[0] == camera_img.shape[1]:
                camera_img = cv2.resize(
                  camera_img, (log_img.shape[0], log_img.shape[1]))
              else:
                camera_img = cv2.resize(
                  camera_img, (log_img.shape[0] // 2, log_img.shape[1]))
              camera_img = camera_img.reshape(
                camera_img.shape[0], camera_img.shape[1], 1)
              camera_img = np.repeat(camera_img, 3, axis=2)
              log_img = np.concatenate([log_img, camera_img], axis=1)
          # writer.write(log_img, rgb_mode=True)
            frames.append(log_img)
          current_obs.copy_(next_obs)
        step += 1
      if self.log_video:

        output_filename = os.path.join(
          self.vidlogdir, 'Output_{}.mp4'.format(
            self.current_learning_iteration)
        )
        imageio.mimsave(output_filename, frames, fps=20)
      # writer.close()

    else:
      self.teacher_load(
        "{}/model_{}.pt".format(self.teacher_log_dir, self.teacher_resume))

      if not self.learn_value_by_self:
        # Give the teacher's value function to student
        self.teacher_gives_student_value_function()
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

          privilege_obs, non_privilege_obs = self.split_obs(
            current_obs)
          # Compute the action
          with torch.no_grad():
            # teacher_actions = self.teacher_actor_critic.act_inference(
            #   privilege_obs)
            student_actions, student_actions_log_prob, values, mu, sigma = self.student_actor_critic.act(
              non_privilege_obs, privilege_obs)

          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(student_actions)
          next_states = self.vec_env.get_state()
          # Record the transition

          # Store Vision Obs
          # print("current_obs shape", current_obs.shape)
          self.storage.add_transitions(
            current_obs, current_states,
            student_actions, rews, dones, values, student_actions_log_prob, mu, sigma
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

        privilege_obs, non_privilege_obs = self.split_obs(
          current_obs)

        _, _, last_values, _, _ = self.student_actor_critic.act(
          non_privilege_obs, privilege_obs)

        mean_trajectory_length, mean_reward = self.storage.get_statistics()

        # Learning step
        start = stop
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_bc_loss = self.update()
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
    mean_std = self.student_actor_critic.log_std.exp().mean()

    self.writer.add_scalar('Loss/value_function',
                           locs['mean_value_loss'], locs['it'])
    self.writer.add_scalar(
      'Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])

    self.writer.add_scalar(
      'Loss/surrogate', locs['mean_bc_loss'], locs['it'])

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
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'Behavior cloning loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      if not self.local:
        wandb.log({
          'Value function loss:': locs['mean_value_loss'],
          'Surrogate loss:': locs['mean_surrogate_loss'],
          'Behavior cloning loss:': locs['mean_bc_loss'],
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
                    f"""{'Behavior cloning loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      if not self.local:
        wandb.log({
          'Value function loss:': locs['mean_value_loss'],
          'Surrogate loss:': locs['mean_surrogate_loss'],
          'Behavior cloning loss:': locs['mean_bc_loss'],
          'Mean action noise std:': mean_std.item(),
          'Mean reward/step:': locs['mean_reward'],
          'Total timesteps:': self.tot_timesteps,
          'Iteration time:': iteration_time,
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
    for epoch in range(self.num_learning_epochs):
      # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
      #        in self.storage.mini_batch_generator(self.num_mini_batches):
      for indices in batch:
        obs_batch = self.storage.observations.view(
          -1, *self.storage.observations.size()[2:])[indices]
        # states_batch = self.storage.states.view(
        #   -1, *self.storage.states.size()[2:])[indices]
        actions_batch = self.storage.actions.view(
          -1, self.storage.actions.size(-1))[indices]
        # teacher_actions_batch = self.storage.teacher_actions.view(
        #   -1, self.storage.teacher_actions.size(-1))[indices]
        target_values_batch = self.storage.values.view(-1, 1)[indices]
        returns_batch = self.storage.returns.view(-1, 1)[indices]
        old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[
          indices]
        advantages_batch = self.storage.advantages.view(-1, 1)[indices]
        old_mu_batch = self.storage.mu.view(-1,
                                            self.storage.actions.size(-1))[indices]
        old_sigma_batch = self.storage.sigma.view(
          -1, self.storage.actions.size(-1))[indices]

        privilege_obs_batch, non_privilege_obs_batch = self.split_obs(
          obs_batch)
        actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.student_actor_critic.evaluate(non_privilege_obs_batch,
                                                                                                                       privilege_obs_batch,
                                                                                                                       actions_batch)
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

        # Imitation learning loss
        with torch.no_grad():
          teacher_actions_batch = self.teacher_actor_critic.act_inference(
            privilege_obs_batch)
        # bc_loss = (mu_batch -
        #            teacher_actions_batch).pow(2).mean()
        bc_loss = (
          torch.clamp(mu_batch, -1, 1) -
          torch.clamp(teacher_actions_batch.detach(), -1, 1)
        ).pow(2).mean()
        loss = surrogate_loss + self.value_loss_coef * \
          value_loss - self.entropy_coef * entropy_batch.mean() + self.bc_loss_coef * bc_loss

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
