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
# from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple
from torch import Tensor

from .storage import LMRolloutStorage

from tasks.controller import LegController
import wandb
import copy


class DistillLMFullLocoTransTrainer:
  def __init__(
      self,
      vec_env,
      teacher_actor_critic_class,
      student_actor_critic_class,
      teacher_encoder,
      student_encoder,
      num_transitions_per_env,
      num_learning_epochs,
      num_mini_batches,
      clip_param=0.2,
      gamma=0.998,
      lam=0.95,
      init_noise_std=1.0,
      surrogate_loss_coef=1.0,
      value_loss_coef=1.0,
      bc_loss_coef=1.0,
      lm_loss_coef=1.0,
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
      # asymmetric=False,
      teacher_resume="None",
      vidlogdir='video',
      log_video=False,
      vid_log_step=500,
      local=False,
      learn_value_by_self=False,
      mask_base_vel=False,
      distill_from_cpg=False,
      use_fake_done=False,
      imi_decay_coef=1,
      eval_env_nums=0,
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
    # del asymmetric
    self.desired_kl = desired_kl

    self.schedule = schedule
    self.step_size = learning_rate

    # PPO parameters
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.num_transitions_per_env = num_transitions_per_env
    self.surrogate_loss_coef = surrogate_loss_coef
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.bc_loss_coef = bc_loss_coef
    self.lm_loss_coef = lm_loss_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    self.eval_env_nums = eval_env_nums

    # PPO components
    self.vec_env = vec_env
    from copy import deepcopy
    teacher_model_cfg = deepcopy(model_cfg)
    student_model_cfg = deepcopy(model_cfg)

    teacher_model_cfg['encoder_params'] = model_cfg['teacher_encoder_params']
    student_model_cfg['encoder_params'] = model_cfg['student_encoder_params']
    if "student_pi_hid_sizes" in model_cfg:
      student_model_cfg['pi_hid_sizes'] = model_cfg['student_pi_hid_sizes']
      student_model_cfg['vf_hid_sizes'] = model_cfg['student_vf_hid_sizes']
      student_model_cfg['enc_hid_sizes'] = model_cfg['student_enc_hid_sizes']
      student_model_cfg['state_vf_hid_sizes'] = model_cfg['student_state_vf_hid_sizes']
    student_model_cfg["encoder_params"]["state_dim"] = self.vec_env.task.state_obs_size - \
        self.vec_env.task.privilege_info_len

    self.is_testing = is_testing
    self.vidlogdir = vidlogdir
    self.log_video = log_video
    self.vid_log_step = vid_log_step
    self.mask_base_vel = mask_base_vel

    teacher_actions_shape = self.action_space.shape
    teacher_state_shape = self.state_space.shape
    student_obs_shape = (
        self.vec_env.task.state_obs_size -
        self.vec_env.task.privilege_info_len + self.vec_env.task.image_obs_size,
    )

    self.distill_from_cpg = False
    self.use_l1 = True

    self.teacher_feature_shape = 128
    if not self.is_testing:
      # Teacher must not using asymmetric training
      # Teacher uses privilege information
      self.teacher_actor_critic = teacher_actor_critic_class(
          teacher_encoder, teacher_state_shape, teacher_state_shape, teacher_actions_shape,
          init_noise_std, teacher_model_cfg, asymmetric=False
      )
      self.teacher_actor_critic.to(self.device)
      self.teacher_fake_encoder = nn.Sequential(
          *self.teacher_actor_critic.actor[:6]
      )
      print(self.teacher_fake_encoder)

    # Student must use asymmetric training
    # Student uses teacher's value model with itself's policy
    # Student uses teacher's policy for behavior cloning
    self.student_actor_critic = student_actor_critic_class(
        student_encoder,
        self.observation_space.shape, teacher_state_shape,
        self.action_space.shape,
        init_noise_std, student_model_cfg  # , asymmetric=asymmetric
    )

    self.student_actor_critic.to(self.device)
    self.storage = LMRolloutStorage(
        self.vec_env.num_envs, num_transitions_per_env,
        student_obs_shape, teacher_state_shape,
        # self.observation_space.shape,
        self.action_space.shape,
        self.teacher_feature_shape,
        # teacher_actions_shape,
        self.device, sampler
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
    # self.writer = SummaryWriter(log_dir=self.student_log_dir, flush_secs=10)
    self.tot_timesteps = 0
    self.tot_time = 0
    self.is_testing = is_testing
    self.current_learning_iteration = 0
    self.apply_reset = apply_reset
    self.teacher_resume = teacher_resume
    assert teacher_resume is not None

    self.use_fake_done = use_fake_done

    self.rewbuffer = deque(maxlen=100)
    self.lenbuffer = deque(maxlen=100)
    if self.eval_env_nums > 0:
      self.eval_rewbuffer = deque(maxlen=100)
      self.eval_lenbuffer = deque(maxlen=100)

    if "gail" in self.vec_env.task.reward_scales:
      self.gail_rewbuffer = deque(maxlen=100)

    self.imi_decay_coef = imi_decay_coef

  def test_teacher(self, path, device='cuda:0'):
    self.teacher_actor_critic.load_state_dict(
        torch.load(path, map_location=device))
    self.teacher_actor_critic.eval()

  def test_student(self, path, device='cuda:0'):
    check_point = torch.load(path, map_location=device)
    self.student_actor_critic.load_state_dict(
        check_point["student_ac"]
    )
    # self.student_actor_critic.load_state_dict(
    #     torch.load(path, map_location=device)
    # )
    self.student_actor_critic.eval()

  def teacher_load(self, path, device='cuda:0'):
    check_point = torch.load(path, map_location=device)
    self.teacher_actor_critic.load_state_dict(
        check_point['ac']
    )
    self.teacher_actor_critic.eval()

  def teacher_gives_student_value_function(self):
    self.student_actor_critic.critic.load_state_dict(
        self.teacher_actor_critic.critic.state_dict())

  def student_load(self, path, device='cuda:0'):
    check_point = torch.load(path, map_location=device)
    self.student_actor_critic.load_state_dict(
        check_point['student_ac']
    )
    self.optimizer.load_state_dict(
        check_point['optimizer']
    )
    self.rewbuffer = check_point['rewbuffer']
    self.lenbuffer = check_point['lenbuffer']
    if self.eval_env_nums > 0:
      self.eval_rewbuffer = check_point['eval_rewbuffer']
      self.eval_lenbuffer = check_point['eval_lenbuffer']

    self.current_learning_iteration = int(
        path.split("_")[-1].split(".")[0]
    )
    self.bc_loss_coef = self.bc_loss_coef * \
        (self.imi_decay_coef ** self.current_learning_iteration)
    self.lm_loss_coef = self.lm_loss_coef * \
        (self.imi_decay_coef ** self.current_learning_iteration)

    if "gail" in self.vec_env.task.reward_scales:
      self.vec_env.task.learners["gail"].discriminator.load_state_dict(
          check_point["gail_learner"]
      )
      self.vec_env.task.learners["gail"].optimizer.load_state_dict(
          check_point["gail_learner_optim"]
      )

    self.student_actor_critic.train()

  def save(self, path):
    # torch.save(self.student_actor_critic.state_dict(), path)
    checkpoint = {
        'student_ac': self.student_actor_critic.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'rewbuffer': self.rewbuffer,
        'lenbuffer': self.lenbuffer
    }
    if "gail" in self.vec_env.task.reward_scales:
      checkpoint["gail_learner"] = self.vec_env.task.learners["gail"].discriminator.state_dict()
      checkpoint["gail_learner_optim"] = self.vec_env.task.learners["gail"].optimizer.state_dict()
    if self.eval_env_nums > 0:
      checkpoint["eval_rewbuffer"] = self.eval_rewbuffer
      checkpoint["eval_lenbuffer"] = self.eval_lenbuffer
    torch.save(checkpoint, path)

  def update_teacher_action_buf(self):
    self.teacher_actions_buf = torch.cat(
        [self.teacher_actions_buf[:, 1:], self.last_teacher_actions.unsqueeze(1)], dim=1
    )

  def split_obs(self, obs_batch):
    # Obs -> [state, privilege information, vis, height]
    state = obs_batch[
        ..., :self.vec_env.task.state_obs_size -
        self.vec_env.task.privilege_info_len]
    privilege_info = obs_batch[..., self.vec_env.task.state_obs_size -
                               self.vec_env.task.privilege_info_len: self.vec_env.task.state_obs_size]
    vis = obs_batch[
        ..., self.vec_env.task.state_obs_size: self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size
    ]
    height = obs_batch[
        ...,
        self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size: self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size + self.vec_env.task.height_map_obs_size
    ]
    if self.distill_from_cpg:
      teacher_state = obs_batch[
          ..., :self.vec_env.task.state_obs_size -
          self.vec_env.task.privilege_info_len
      ]
      # print(teacher_state.shape)
      # print(privilege_info.shape)
      # print(height.shape)
      privilege_obs = teacher_state
      if self.teacher_use_aug_phase_info:
        privilege_obs = torch.cat([
            privilege_obs,
            self.teacher_leg_controller.phase_info.view(
                self.num_envs, -1),
        ], dim=-1)
      if self.teacher_use_orig_phase_info:
        privilege_obs = torch.cat([
            privilege_obs,
            self.teacher_leg_controller.time_phase_info.view(
                self.num_envs, -1),
        ], dim=-1)
      privilege_obs = torch.cat([
          privilege_obs, privilege_info, height
      ], dim=-1)

    else:
      privilege_obs = torch.cat([
          state, privilege_info, height
      ], dim=-1)
    non_privilege_obs = torch.cat([
        state, vis
    ], dim=-1)

    if self.mask_base_vel:
      non_privilege_obs[..., :3] = 0
    return privilege_obs, non_privilege_obs

  def get_teacher_action(self, privileged_obs):
    if self.distill_from_cpg:
      teacher_actions_original = self.teacher_actor_critic.act_inference(
          privileged_obs)
      teacher_actions_original = torch.clamp(teacher_actions_original, -1, 1)
      target_act = self.teacher_leg_controller._get_cpg_target_pos(
          self.teacher_cpg_action_scale, teacher_actions_original, self.nominal_residual_phase, self.vec_env.task
      )
      target_act = target_act - \
          self.vec_env.task.default_dof_pos.to(self.device)
      target_act /= self.vec_env.task.action_scale.to(self.device)
    else:
      target_act = self.teacher_actor_critic.act_inference(
          privileged_obs)
      teacher_actions_original = target_act
    return target_act, teacher_actions_original

  def run(self, num_learning_iterations, log_interval=1):
    current_obs = self.vec_env.reset()
    if self.distill_from_cpg:
      self.teacher_leg_controller.reset(
          torch.arange(self.num_envs, device=self.device))
    current_states = self.vec_env.get_state()

    dones = torch.as_tensor([0.0] * self.vec_env.num_envs,
                            device=current_obs.device, dtype=torch.float)
    if self.is_testing:
      from pathlib import Path
      Path(self.vidlogdir).mkdir(parents=True, exist_ok=True)
      import imageio
      frames = []
      step = 0
      while (not self.log_video or (step < self.vid_log_step)) and not self.vec_env.task.task_wrapper.test_finished:
        with torch.no_grad():
          if self.apply_reset:
            current_obs = self.vec_env.reset()
            dones = torch.as_tensor([0.0] * self.vec_env.num_envs,
                                    device=current_obs.device, dtype=torch.float)

          # Compute the action
          _, non_privilege_obs = self.split_obs(current_obs)
          actions = self.student_actor_critic.act_inference(
              non_privilege_obs
          )
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          dones = dones.to(torch.float)
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
        print(self.vidlogdir)
        output_filename = os.path.join(
            self.vidlogdir, 'Output_{}.mp4'.format(
                self.current_learning_iteration)
        )
        imageio.mimsave(output_filename, frames, fps=20)
      # writer.close()

    else:
      self.teacher_load(
          "{}/model_{}.pt".format(self.teacher_log_dir, self.teacher_resume))

      # Give the teacher's value function to student
      if not self.learn_value_by_self:
        # Give the teacher's value function to student
        self.teacher_gives_student_value_function()

      cur_reward_sum = torch.zeros(
          self.vec_env.num_envs, dtype=torch.float, device=self.device)

      if "gail" in self.vec_env.task.reward_scales:
        cur_gail_reward_sum = torch.zeros(
            self.vec_env.num_envs, dtype=torch.float, device=self.device)

      cur_episode_length = torch.zeros(
          self.vec_env.num_envs, dtype=torch.float, device=self.device)

      if self.eval_env_nums > 0:
        cur_eval_reward_sum = torch.zeros(
            self.eval_env_nums, dtype=torch.float, device=self.device)
        cur_eval_episode_length = torch.zeros(
            self.eval_env_nums, dtype=torch.float, device=self.device)

      for it in range(self.current_learning_iteration, num_learning_iterations):
        start = time.time()
        ep_infos = []

        reward_sum = []
        if "gail" in self.vec_env.task.reward_scales:
          gail_reward_sum = []
        episode_length = []

        if self.eval_env_nums > 0:
          eval_reward_sum = []
          eval_episode_length = []

        # Rollout
        for _ in range(self.num_transitions_per_env):
          if self.apply_reset:
            current_obs = self.vec_env.reset()
            # current_states = self.vec_env.get_state()
            if self.distill_from_cpg:
              self.teacher_leg_controller.reset(
                  torch.arange(self.num_envs, device=self.device))
            dones = torch.as_tensor([0.0] * self.vec_env.num_envs,
                                    device=current_obs.device, dtype=torch.float)
            self.last_teacher_actions.zero_()

          # Compute the action
          with torch.no_grad():
            # if self.distill_from_cpg:
            #   self.update_teacher_action_buf()
            privilege_obs, non_privilege_obs = self.split_obs(
                current_obs
            )
            teacher_acts, raw_teacher_action = self.get_teacher_action(
                privilege_obs
            )
            teacher_feature = self.teacher_fake_encoder(privilege_obs)
            # if self.distill_from_cpg:
            #   self.last_teacher_actions.copy_(raw_teacher_action)
            # print(non_privilege_obs.shape)
            student_actions, student_actions_log_prob, values, obs_values, mu, sigma = self.student_actor_critic.act(
                non_privilege_obs, privilege_obs
            )
          # print("teacher acts")

          prev_dones = dones.clone()

          if self.eval_env_nums > 0:
            with torch.no_grad():
              student_actions[-self.eval_env_nums:] = mu[-self.eval_env_nums:]
              student_actions_log_prob[-self.eval_env_nums:] = self.student_actor_critic.evaluate(
                  non_privilege_obs[-self.eval_env_nums:],
                  current_states[-self.eval_env_nums:],
                  mu[-self.eval_env_nums:]
              )[0]
          # next_obs, rews, dones, infos = self.vec_env.step(student_actions)
          # print("pre step")
          # print(student_actions)
          # print(actions_for_env)
          next_obs, rews, dones, infos = self.vec_env.step(student_actions)
          if "gail" in self.vec_env.task.reward_scales:
            gail_rews = infos["gail_rew"]

          # print("aft step")
          dones = dones.to(torch.float)
          fake_dones = infos["fake_done"]
          if self.use_fake_done:
            dones = fake_dones.to(torch.float)
          if self.distill_from_cpg:
            env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
              self.teacher_leg_controller.reset(env_ids)
              # self.teacher_actions_buf[env_ids] = 0
              # self.last_teacher_actions[env_ids] = 0
          next_states = self.vec_env.get_state()

          self.storage.add_transitions(
              # current_obs,
              non_privilege_obs, privilege_obs,
              student_actions, rews, dones, prev_dones, values, obs_values, student_actions_log_prob, mu, sigma,
              teacher_acts, teacher_feature
          )
          current_obs.copy_(next_obs)
          current_states.copy_(next_states)
          ep_infos.append(infos)

          if self.print_log:
            cur_reward_sum[:] += rews
            cur_episode_length[:] += 1

            if "gail" in self.vec_env.task.reward_scales:
              cur_reward_sum[:] -= gail_rews
              cur_gail_reward_sum[:] += gail_rews

            if self.eval_env_nums > 0:
              cur_eval_reward_sum[:] += (
                  rews[-self.eval_env_nums:] - gail_rews[-self.eval_env_nums:]
              )
              cur_eval_episode_length[:] += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)
            reward_sum.extend(
                cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())

            if "gail" in self.vec_env.task.reward_scales:
              gail_reward_sum.extend(
                  cur_gail_reward_sum[new_ids][:, 0].cpu().numpy().tolist())

            episode_length.extend(
                cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            if "gail" in self.vec_env.task.reward_scales:
              cur_gail_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

            if self.eval_env_nums > 0:
              eval_new_ids = (
                  dones[-self.eval_env_nums:] > 0
              ).nonzero(as_tuple=False)
              eval_reward_sum.extend(
                  cur_eval_reward_sum[eval_new_ids][:, 0].cpu().numpy().tolist())
              eval_episode_length.extend(
                  cur_eval_episode_length[eval_new_ids][:, 0].cpu().numpy().tolist())
              cur_eval_reward_sum[eval_new_ids] = 0
              cur_eval_episode_length[eval_new_ids] = 0

        if self.print_log:
          # reward_sum = [x[0] for x in reward_sum]
          # episode_length = [x[0] for x in episode_length]
          self.rewbuffer.extend(reward_sum)
          self.lenbuffer.extend(episode_length)
          if "gail" in self.vec_env.task.reward_scales:
            self.gail_rewbuffer.extend(gail_reward_sum)

          if self.eval_env_nums > 0:
            self.eval_rewbuffer.extend(eval_reward_sum)
            self.eval_lenbuffer.extend(eval_episode_length)

        stop = time.time()
        collection_time = stop - start

        privilege_obs, non_privilege_obs = self.split_obs(
            current_obs)

        with torch.no_grad():
          _, _, last_values, _, _, _ = self.student_actor_critic.act(
              non_privilege_obs, privilege_obs
          )

        mean_trajectory_length, mean_reward = self.storage.get_statistics()

        # Learning step
        start = stop
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        mean_value_loss, mean_obs_value_loss, mean_surrogate_loss, mean_bc_loss, mean_lm_loss = self.update()
        self.storage.clear()
        stop = time.time()
        learn_time = stop - start
        if self.print_log:
          rewbuffer = copy.deepcopy(self.rewbuffer)
          lenbuffer = copy.deepcopy(self.lenbuffer)
          if "gail" in self.vec_env.task.reward_scales:
            gail_rewbuffer = copy.deepcopy(self.gail_rewbuffer)
          if self.eval_env_nums > 0:
            eval_rewbuffer = copy.deepcopy(self.eval_rewbuffer)
            eval_lenbuffer = copy.deepcopy(self.eval_lenbuffer)

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
        # self.writer.add_scalar('Episode/' + key, value, locs['it'])
        ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
    mean_std = self.student_actor_critic.log_std.exp().mean()

    fps = int(self.num_transitions_per_env * self.vec_env.num_envs /
              (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    log_string = (f"""{'#' * width}\n"""
                  f"""{str.center(width, ' ')}\n\n"""
                  f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                  f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                  f"""{'Obs Value function loss:':>{pad}} {locs['mean_obs_value_loss']:.4f}\n"""
                  f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                  f"""{'Behavior cloning loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                  f"""{'Latent Match loss:':>{pad}} {locs['mean_lm_loss']:.4f}\n"""
                  f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                  f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                  f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
    log_dict = {
        'Value function loss:': locs['mean_value_loss'],
        'Obs Value function loss:': locs['mean_obs_value_loss'],
        'Surrogate loss:': locs['mean_surrogate_loss'],
        'Behavior cloning loss:': locs['mean_bc_loss'],
        'Latent match loss:': locs['mean_lm_loss'],
        'Mean action noise std:': mean_std.item(),
        'Mean reward/step:': locs['mean_reward'],
        'Total timesteps:': self.tot_timesteps,
        'Iteration time:': iteration_time,
        'Mean episode length/episode:': locs['mean_trajectory_length']
    }

    if len(locs['rewbuffer']) > 0:
      log_string += (f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                     f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
      log_dict.update({
          'Mean reward:': statistics.mean(locs['rewbuffer']),
          'Mean episode length:': statistics.mean(locs['lenbuffer']),
      })
      if "gail" in self.vec_env.task.reward_scales:
        log_string += f"""{'Mean GAIL reward:':>{pad}} {statistics.mean(locs['gail_rewbuffer']):.2f}\n"""
        log_dict["Mean Gail reward"] = statistics.mean(locs['gail_rewbuffer'])

    if self.eval_env_nums > 0 and len(locs['eval_rewbuffer']) > 0:
      log_string += (
          f"""{'Eval Mean reward:':>{pad}} {statistics.mean(locs['eval_rewbuffer']):.2f}\n"""
          f"""{'Eval Mean episode length:':>{pad}} {statistics.mean(locs['eval_lenbuffer']):.2f}\n"""
      )
      log_dict.update({
          'Eval Mean reward:': statistics.mean(locs['eval_rewbuffer']),
          'Eval Mean episode length:': statistics.mean(locs['eval_lenbuffer']),
      })

    if not self.local:
      wandb.log(log_dict)

    log_string += ep_string
    log_string += (f"""{'-' * width}\n"""
                   f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                   f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                   f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                   f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
    print(log_string)

  def recon_criterion(self, out, target):
    if self.use_l1:
      return torch.abs(
          out - target
      )
    else:
      return (out - target).pow(2)

  def update(self):
    mean_obs_value_loss = 0
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_bc_loss = 0
    mean_lm_loss = 0
    batch = self.storage.mini_batch_generator(self.num_mini_batches)
    for epoch in range(self.num_learning_epochs):
      for indices in batch:
        # obs_batch = self.storage.observations[:, indices]
        privilege_obs_batch = self.storage.privileged_observations[:, indices].detach(
        )
        non_privilege_obs_batch = self.storage.non_privileged_observations[:, indices].detach(
        )
        actions_batch = self.storage.actions[:, indices].detach()

        dones_batch = self.storage.dones[:, indices].detach()
        prev_dones_batch = self.storage.prev_dones[:, indices].detach()

        target_values_batch = self.storage.values[:, indices].detach()
        target_obs_values_batch = self.storage.obs_values[:, indices].detach()
        returns_batch = self.storage.returns[:, indices].detach()
        old_actions_log_prob_batch = self.storage.actions_log_prob[:, indices].detach(
        )
        advantages_batch = self.storage.advantages[:, indices].detach()
        old_mu_batch = self.storage.mu[:, indices].detach()
        old_sigma_batch = self.storage.sigma[:, indices].detach()

        teacher_actions_batch = self.storage.teacher_actions[:, indices].detach(
        )
        teacher_feature_batch = self.storage.teacher_features[:, indices].detach(
        )

        # privilege_obs_batch, non_privilege_obs_batch = self.split_obs(
        #     obs_batch)
        actions_log_prob_batch, entropy_batch, value_batch, obs_value_batch, mu_batch, sigma_batch, rnnout = self.student_actor_critic.evaluate(
            non_privilege_obs_batch,
            privilege_obs_batch,
            actions_batch, return_rnnout=True
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

        # Feature Match Loss
        # with torch.no_grad():
        #   teacher_feature_batch = self.teacher_fake_encoder(
        #       privilege_obs_batch)
        # if self.alternate_sampling:
        #   lm_loss = (
        #       self.recon_criterion(
        #           rnnout, teacher_feature_batch.detach()
        #       ) * teacher_masks_batch
        #   ).mean()
        # else:
        lm_loss = self.recon_criterion(
            rnnout, teacher_feature_batch.detach()
        ).mean()
        # Value function loss
        if self.use_clipped_value_loss:
          value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                          self.clip_param)
          value_losses = (value_batch - returns_batch).pow(2)
          value_losses_clipped = (
              value_clipped - returns_batch
          ).pow(2)
          value_loss = torch.max(
              value_losses, value_losses_clipped
          ).mean()

          obs_value_clipped = target_obs_values_batch + (
              obs_value_batch - target_obs_values_batch).clamp(-self.clip_param, self.clip_param)
          obs_value_losses = (obs_value_batch - returns_batch).pow(2)
          obs_value_losses_clipped = (
              obs_value_clipped - returns_batch
          ).pow(2)
          obs_value_loss = torch.max(
              obs_value_losses, obs_value_losses_clipped
          ).mean()

        else:
          value_loss = (returns_batch - value_batch).pow(2).mean()
          obs_value_loss = (returns_batch - obs_value_batch).pow(2).mean()

        # Imitation learning loss
        bc_loss = self.recon_criterion(
            torch.clamp(mu_batch, -1, 1),
            torch.clamp(teacher_actions_batch.detach(), -1, 1)
        ).mean()

        loss = self.surrogate_loss_coef * surrogate_loss + self.value_loss_coef * \
            value_loss + self.value_loss_coef * \
            obs_value_loss - self.entropy_coef * entropy_batch.mean() + self.bc_loss_coef * \
            bc_loss + self.lm_loss_coef * lm_loss

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.student_actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_value_loss += value_loss.item()
        mean_obs_value_loss += obs_value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_bc_loss += bc_loss.item()
        mean_lm_loss += lm_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_obs_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_bc_loss /= num_updates
    mean_lm_loss /= num_updates

    self.bc_loss_coef *= self.imi_decay_coef
    self.lm_loss_coef *= self.imi_decay_coef

    return mean_value_loss, mean_obs_value_loss, mean_surrogate_loss, mean_bc_loss, mean_lm_loss
