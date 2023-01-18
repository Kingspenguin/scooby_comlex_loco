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


class AsymmetricDistillLMLocoTransTwinTrainer:
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
      pd_from_cpg_bc_loss_coef=1.0,
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
      asymmetric=False,
      teacher_resume="None",
      vidlogdir='video',
      log_video=False,
      vid_log_step=500,
      local=False,
      learn_value_by_self=False,
      distill_from_cpg=False,
      alternate_sampling=False,
      use_fake_done=False,
      imi_decay_coef=1
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
    self.surrogate_loss_coef = surrogate_loss_coef
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.bc_loss_coef = bc_loss_coef
    self.pd_from_cpg_bc_loss_coef = pd_from_cpg_bc_loss_coef
    self.lm_loss_coef = lm_loss_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    self.alterante_sampling = alternate_sampling

    # PPO components
    self.vec_env = vec_env
    from copy import deepcopy
    teacher_model_cfg = deepcopy(model_cfg)
    student_model_cfg = deepcopy(model_cfg)

    teacher_model_cfg['encoder_params'] = model_cfg['teacher_encoder_params']
    student_model_cfg['encoder_params'] = model_cfg['student_encoder_params']
    if "student_pi_enc_hid_sizes" in model_cfg:
      student_model_cfg['pi_enc_hid_sizes'] = model_cfg['student_pi_enc_hid_sizes']
      student_model_cfg['pi_pd_hid_sizes'] = model_cfg['student_pi_pd_hid_sizes']
      student_model_cfg['pi_cpg_hid_sizes'] = model_cfg['student_pi_cpg_hid_sizes']

    student_model_cfg["encoder_params"]["state_dim"] = self.vec_env.task.state_obs_size - \
        self.vec_env.task.privilege_info_len

    self.is_testing = is_testing
    self.vidlogdir = vidlogdir
    self.log_video = log_video
    self.vid_log_step = vid_log_step

    teacher_actions_shape = self.action_space.shape
    teacher_state_shape = self.state_space.shape
    student_obs_shape = (self.vec_env.task.state_obs_size -
                         self.vec_env.task.privilege_info_len + self.vec_env.task.image_obs_size, )

    self.distill_from_cpg = distill_from_cpg
    self.use_l1 = distill_from_cpg
    assert self.distill_from_cpg

    self.cfg = self.vec_env.task.cfg
    self.num_envs = self.vec_env.num_envs

    self.teacher_cpg_action_scale = self.cfg["env"]["control"]["cpg_actionScale"]
    import copy
    teacher_control_cfg = copy.deepcopy(self.cfg["env"]["control"])
    teacher_control_cfg["legcontroller"] = "cpg"
    self.teacher_leg_controller = LegController(
        cfg=teacher_control_cfg,
        num_envs=self.num_envs, device=self.device
    )

    self.teacher_cpg_action_scale = torch.as_tensor(
        self.teacher_cpg_action_scale * 4, device=self.device
    )
    self.phase_scale = torch.as_tensor(
        [teacher_control_cfg["phaseScale"]] * 4, device=self.device)
    nominal_phase = teacher_control_cfg["phaseScale"]
    self.nominal_residual_phase = torch.as_tensor(
        [-nominal_phase, nominal_phase, nominal_phase, -nominal_phase], device=self.device)
    self.teacher_cpg_action_scale = torch.cat(
        [self.teacher_cpg_action_scale, self.phase_scale], dim=-1)
    teacher_actions_shape = (16,)
    self.num_teacher_actions = np.prod(teacher_actions_shape)

    self.teacher_use_aug_phase_info = self.cfg["env"]["teacher_sensor"]["aug_time_phase"]
    self.teacher_use_orig_phase_info = self.cfg["env"]["teacher_sensor"]["original_time_phase"]

    cpg_phase_info_shape = 0
    if self.teacher_use_aug_phase_info:
      teacher_state_shape = (np.prod(teacher_state_shape) + 8, )
      cpg_phase_info_shape += 8
    if self.teacher_use_orig_phase_info:
      teacher_state_shape = (np.prod(teacher_state_shape) + 8, )
      cpg_phase_info_shape += 8

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
        self.observation_space.shape, teacher_state_shape, cpg_phase_info_shape,
        self.action_space.shape, teacher_actions_shape,
        init_noise_std, student_model_cfg, asymmetric=True
    )

    self.student_actor_critic.to(self.device)
    self.storage = LMRolloutStorage(
        self.vec_env.num_envs,
        num_transitions_per_env,
        student_obs_shape, teacher_state_shape, (cpg_phase_info_shape,),
        # self.observation_space.shape,
        self.action_space.shape,
        self.teacher_feature_shape,
        teacher_actions_shape,
        self.device, sampler,
        with_cpg_mask=True
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

    self.pd_rewbuffer = deque(maxlen=100)
    self.pd_lenbuffer = deque(maxlen=100)

    self.cpg_rewbuffer = deque(maxlen=100)
    self.cpg_lenbuffer = deque(maxlen=100)

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
    if self.alterante_sampling:
      self.pd_rewbuffer = check_point['pd_rewbuffer']
      self.pd_lenbuffer = check_point['pd_lenbuffer']

      self.cpg_rewbuffer = check_point['cpg_rewbuffer']
      self.cpg_lenbuffer = check_point['cpg_lenbuffer']
    else:
      self.rewbuffer = check_point['rewbuffer']
      self.lenbuffer = check_point['lenbuffer']
    self.current_learning_iteration = int(
        path.split("_")[-1].split(".")[0]
    )
    self.bc_loss_coef = self.bc_loss_coef * \
        (self.imi_decay_coef ** self.current_learning_iteration)
    self.lm_loss_coef = self.lm_loss_coef * \
        (self.imi_decay_coef ** self.current_learning_iteration)
    self.student_actor_critic.train()

  def save(self, path):
    # torch.save(self.student_actor_critic.state_dict(), path)
    if self.alterante_sampling:
      checkpoint = {
          'student_ac': self.student_actor_critic.state_dict(),
          'optimizer': self.optimizer.state_dict(),
          'pd_rewbuffer': self.pd_rewbuffer,
          'pd_lenbuffer': self.pd_lenbuffer,
          'cpg_rewbuffer': self.cpg_rewbuffer,
          'cpg_lenbuffer': self.cpg_lenbuffer
      }
    else:
      checkpoint = {
          'student_ac': self.student_actor_critic.state_dict(),
          'optimizer': self.optimizer.state_dict(),
          'rewbuffer': self.rewbuffer,
          'lenbuffer': self.lenbuffer
      }
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
    teacher_state = obs_batch[
        ..., :self.vec_env.task.state_obs_size -
        self.vec_env.task.privilege_info_len
    ]

    cpg_phase_infos = []
    privilege_obs = teacher_state
    if self.teacher_use_aug_phase_info:
      privilege_obs = torch.cat([
          privilege_obs,
          self.teacher_leg_controller.phase_info.view(
              self.num_envs, -1),
      ], dim=-1)
      cpg_phase_infos.append(self.teacher_leg_controller.phase_info.view(
          self.num_envs, -1)
      )
    if self.teacher_use_orig_phase_info:
      privilege_obs = torch.cat([
          privilege_obs,
          self.teacher_leg_controller.time_phase_info.view(
              self.num_envs, -1),
      ], dim=-1)
      cpg_phase_infos.append(
          self.teacher_leg_controller.time_phase_info.view(
              self.num_envs, -1
          )
      )

    cpg_phase_infos = torch.cat(
        cpg_phase_infos, dim=-1
    )

    privilege_obs = torch.cat([
        privilege_obs, privilege_info, height
    ], dim=-1)

    non_privilege_obs = torch.cat([
        state, vis
    ], dim=-1)

    return privilege_obs, non_privilege_obs, cpg_phase_infos

  def cpg_actions_to_pd_actions(self, cpg_actions):
    cpg_actions = torch.clamp(cpg_actions, -1, 1)
    pd_actions = self.teacher_leg_controller._get_cpg_target_pos(
        self.teacher_cpg_action_scale, cpg_actions, self.nominal_residual_phase, self.vec_env.task
    )
    pd_actions = pd_actions - \
        self.vec_env.task.default_dof_pos.to(self.device)
    pd_actions /= self.vec_env.task.action_scale.to(self.device)
    return pd_actions

  def get_teacher_action(self, privileged_obs):

    teacher_cpg_actions = self.teacher_actor_critic.act_inference(
        privileged_obs)
    teacher_pd_actions = self.cpg_actions_to_pd_actions(
        teacher_cpg_actions
    )
    return teacher_pd_actions, teacher_cpg_actions

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
            dones = torch.as_tensor(
                [0.0] * self.vec_env.num_envs,
                device=current_obs.device, dtype=torch.float
            )

          # Compute the action
          _, non_privilege_obs, _ = self.split_obs(current_obs)
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

      pd_cur_reward_sum = torch.zeros(
          self.vec_env.num_envs // 2, dtype=torch.float, device=self.device)
      pd_cur_episode_length = torch.zeros(
          self.vec_env.num_envs // 2, dtype=torch.float, device=self.device)

      cpg_cur_reward_sum = torch.zeros(
          self.vec_env.num_envs // 2, dtype=torch.float, device=self.device)
      cpg_cur_episode_length = torch.zeros(
          self.vec_env.num_envs // 2, dtype=torch.float, device=self.device)

      for it in range(self.current_learning_iteration, num_learning_iterations):
        start = time.time()
        ep_infos = []

        pd_reward_sum = []
        pd_episode_length = []

        cpg_reward_sum = []
        cpg_episode_length = []

        # Rollout
        for _ in range(self.num_transitions_per_env):
          if self.apply_reset:
            current_obs = self.vec_env.reset()
            # current_states = self.vec_env.get_state()
            self.teacher_leg_controller.reset(
                torch.arange(self.num_envs, device=self.device))
            dones = torch.as_tensor([0.0] * self.vec_env.num_envs,
                                    device=current_obs.device, dtype=torch.float)
            self.last_teacher_actions.zero_()

          # Compute the action
          with torch.no_grad():
            # if self.distill_from_cpg:
            #   self.update_teacher_action_buf()
            privilege_obs, non_privilege_obs, cpg_phase_infos = self.split_obs(
                current_obs
            )
            teacher_pd_actions, teacher_cpg_actions = self.get_teacher_action(
                privilege_obs
            )
            teacher_feature = self.teacher_fake_encoder(privilege_obs)

            pd_actions, pd_actions_log_prob,\
                cpg_actions, cpg_actions_log_prob,\
                values,\
                pd_mu, pd_sigma,\
                cpg_mu, cpg_sigma = self.student_actor_critic.act(
                    non_privilege_obs, privilege_obs, cpg_phase_infos
                )

          # Step the vec_environment
          pd_actions_from_cpg = self.cpg_actions_to_pd_actions(
              cpg_mu
          )

          prev_dones = dones.clone()

          actions_for_env = torch.cat([
              pd_actions[:self.vec_env.num_envs // 2],
              self.cpg_actions_to_pd_actions(
                  cpg_actions
              )[self.vec_env.num_envs // 2:]
          ], dim=0)

          # next_obs, rews, dones, infos = self.vec_env.step(student_actions)
          next_obs, rews, dones, infos = self.vec_env.step(actions_for_env)

          dones = dones.to(torch.float)
          fake_dones = infos["fake_done"]
          if self.use_fake_done:
            dones = fake_dones.to(torch.float)

          env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
          if len(env_ids) > 0:
            self.teacher_leg_controller.reset(env_ids)
            # self.teacher_actions_buf[env_ids] = 0
            # self.last_teacher_actions[env_ids] = 0
          next_states = self.vec_env.get_state()

          cpg_masks = torch.zeros_like(rews, device=rews.device)
          cpg_masks[self.vec_env.num_envs // 2:] = 1
          self.storage.add_transitions(
              # current_obs,
              non_privilege_obs, privilege_obs, cpg_phase_infos,
              pd_actions, cpg_actions,
              rews,
              dones, prev_dones, values,
              pd_actions_log_prob, pd_mu, pd_sigma,
              cpg_actions_log_prob, cpg_mu, cpg_sigma,
              teacher_pd_actions, teacher_cpg_actions, teacher_feature,
              pd_actions_from_cpg,
              cpg_masks=cpg_masks
          )

          current_obs.copy_(next_obs)
          current_states.copy_(next_states)
          ep_infos.append(infos)

          if self.print_log:
            pd_cur_reward_sum[:] += rews[:self.vec_env.num_envs // 2]
            pd_cur_episode_length[:] += 1

            cpg_cur_reward_sum[:] += rews[self.vec_env.num_envs // 2:]
            cpg_cur_episode_length[:] += 1

            pd_new_ids = (
                fake_dones[:self.vec_env.num_envs // 2] > 0
            ).nonzero(as_tuple=False)
            pd_reward_sum.extend(
                pd_cur_reward_sum[pd_new_ids][:, 0].cpu().numpy().tolist())
            pd_episode_length.extend(
                pd_cur_episode_length[pd_new_ids][:, 0].cpu().numpy().tolist())
            pd_cur_reward_sum[pd_new_ids] = 0
            pd_cur_episode_length[pd_new_ids] = 0

            cpg_new_ids = (
                fake_dones[self.vec_env.num_envs // 2:] > 0
            ).nonzero(as_tuple=False)
            cpg_reward_sum.extend(
                cpg_cur_reward_sum[cpg_new_ids][:, 0].cpu().numpy().tolist())
            cpg_episode_length.extend(
                cpg_cur_episode_length[cpg_new_ids][:, 0].cpu().numpy().tolist())
            cpg_cur_reward_sum[cpg_new_ids] = 0
            cpg_cur_episode_length[cpg_new_ids] = 0

        if self.print_log:
          # reward_sum = [x[0] for x in reward_sum]
          # episode_length = [x[0] for x in episode_length]
          self.pd_rewbuffer.extend(pd_reward_sum)
          self.pd_lenbuffer.extend(pd_episode_length)

          self.cpg_rewbuffer.extend(cpg_reward_sum)
          self.cpg_lenbuffer.extend(cpg_episode_length)

        stop = time.time()
        collection_time = stop - start

        privilege_obs, non_privilege_obs, cpg_phase_infos = self.split_obs(
            current_obs)

        with torch.no_grad():
          _, _, _, _, last_values, _, _, _, _ = self.student_actor_critic.act(
              non_privilege_obs, privilege_obs, cpg_phase_infos
          )

        mean_trajectory_length, mean_reward = self.storage.get_statistics()

        # Learning step
        start = stop
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_lm_loss, mean_pd_from_cpg_bc_loss = self.update()
        self.storage.clear()
        stop = time.time()
        learn_time = stop - start
        if self.print_log:
          pd_rewbuffer = copy.deepcopy(self.pd_rewbuffer)
          pd_lenbuffer = copy.deepcopy(self.pd_lenbuffer)

          cpg_rewbuffer = copy.deepcopy(self.cpg_rewbuffer)
          cpg_lenbuffer = copy.deepcopy(self.cpg_lenbuffer)
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
    pd_mean_std = self.student_actor_critic.pd_log_std.exp().mean()
    cpg_mean_std = self.student_actor_critic.cpg_log_std.exp().mean()

    fps = int(self.num_transitions_per_env * self.vec_env.num_envs /
              (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    if len(locs['pd_rewbuffer']) > 0:
      log_string = (
          f"""{'#' * width}\n"""
          f"""{str.center(width, ' ')}\n\n"""
          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                  'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
          f"""{'Behavior cloning loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
          f"""{'PD from CPG Behavior cloning loss:':>{pad}} {locs['mean_pd_from_cpg_bc_loss']:.4f}\n"""
          f"""{'Latent Match loss:':>{pad}} {locs['mean_lm_loss']:.4f}\n"""
          f"""{'PD Mean action noise std:':>{pad}} {pd_mean_std.item():.2f}\n"""
          f"""{'CPG Mean action noise std:':>{pad}} {cpg_mean_std.item():.2f}\n"""
          f"""{'PD Mean reward:':>{pad}} {statistics.mean(locs['pd_rewbuffer']):.2f}\n"""
          f"""{'PD Mean episode length:':>{pad}} {statistics.mean(locs['pd_lenbuffer']):.2f}\n"""
          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
      )
      if len(locs['cpg_rewbuffer']) > 0:
        log_string += (
            f"""{'CPG Mean reward:':>{pad}} {statistics.mean(locs['cpg_rewbuffer']):.2f}\n"""
            f"""{'CPG Mean episode length:':>{pad}} {statistics.mean(locs['cpg_lenbuffer']):.2f}\n"""
        )
      if not self.local:
        log_dict = {
            'Value function loss:': locs['mean_value_loss'],
            'Surrogate loss:': locs['mean_surrogate_loss'],
            'Behavior cloning loss:': locs['mean_bc_loss'],
            'PD from CPG Behavior cloning loss:': locs['mean_pd_from_cpg_bc_loss'],
            'Latent match loss:': locs['mean_lm_loss'],
            'PD Mean action noise std:': pd_mean_std.item(),
            'CPG Mean action noise std:': cpg_mean_std.item(),
            'PD Mean reward:': statistics.mean(locs['pd_rewbuffer']),
            'PD Mean episode length:': statistics.mean(locs['pd_lenbuffer']),
            'Mean reward/step:': locs['mean_reward'],
            'Mean episode length/episode:': locs['mean_trajectory_length']
        }
        if len(locs['cpg_rewbuffer']) > 0:
          log_dict["CPG Mean reward:"] = statistics.mean(
              locs['cpg_rewbuffer'])
          log_dict["CPG Mean episode length:"] = statistics.mean(
              locs['cpg_lenbuffer'])
        wandb.log(log_dict)
    else:
      log_string = (f"""{'#' * width}\n"""
                    f"""{str.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'Behavior cloning loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                    f"""{'PD from CPG Behavior cloning loss:':>{pad}} {locs['mean_pd_from_cpg_bc_loss']:.4f}\n"""
                    f"""{'Latent Match loss:':>{pad}} {locs['mean_lm_loss']:.4f}\n"""
                    f"""{'PD Mean action noise std:':>{pad}} {pd_mean_std.item():.2f}\n"""
                    f"""{'CPG Mean action noise std:':>{pad}} {cpg_mean_std.item():.2f}\n"""
                    f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                    f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
      if not self.local:
        wandb.log({
            'Value function loss:': locs['mean_value_loss'],
            'Surrogate loss:': locs['mean_surrogate_loss'],
            'Behavior cloning loss:': locs['mean_bc_loss'],
            'PD from CPG Behavior cloning loss:': locs['mean_pd_from_cpg_bc_loss'],
            'Latent match loss:': locs['mean_lm_loss'],
            'PD Mean action noise std:': pd_mean_std.item(),
            'CPG Mean action noise std:': cpg_mean_std.item(),
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

  def recon_criterion(self, out, target, l1=True):
    if l1:
      return torch.abs(
          out - target
      )
    else:
      return (out - target).pow(2)

  def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_bc_loss = 0
    mean_pd_from_cpg_bc_loss = 0
    mean_lm_loss = 0
    batch = self.storage.mini_batch_generator(self.num_mini_batches)
    for epoch in range(self.num_learning_epochs):
      for indices in batch:
        # obs_batch = self.storage.observations[:, indices]
        privilege_obs_batch = self.storage.privileged_observations[:, indices].detach(
        )
        non_privilege_obs_batch = self.storage.non_privileged_observations[:, indices].detach(
        )
        cpg_phase_infos_batch = self.storage.cpg_phase_infos[:, indices].detach(
        )

        pd_actions_batch = self.storage.pd_actions[:, indices].detach()
        cpg_actions_batch = self.storage.cpg_actions[:, indices].detach()

        dones_batch = self.storage.dones[:, indices].detach()
        prev_dones_batch = self.storage.prev_dones[:, indices].detach()

        target_values_batch = self.storage.values[:, indices].detach()
        returns_batch = self.storage.returns[:, indices].detach()

        pd_old_actions_log_prob_batch = self.storage.pd_actions_log_prob[:, indices].detach(
        )
        cpg_old_actions_log_prob_batch = self.storage.cpg_actions_log_prob[:, indices].detach(
        )

        advantages_batch = self.storage.advantages[:, indices].detach()
        pd_old_mu_batch = self.storage.pd_mu[:, indices].detach()
        pd_old_sigma_batch = self.storage.pd_sigma[:, indices].detach()

        cpg_old_mu_batch = self.storage.cpg_mu[:, indices].detach()
        cpg_old_sigma_batch = self.storage.cpg_sigma[:, indices].detach()

        teacher_pd_actions_batch = self.storage.teacher_pd_actions[:, indices].detach(
        )
        teacher_cpg_actions_batch = self.storage.teacher_cpg_actions[:, indices].detach(
        )
        teacher_feature_batch = self.storage.teacher_features[:, indices].detach(
        )

        pd_actions_from_cpg_batch = self.storage.pd_actions_from_cpg[:, indices].detach(
        )
        cpg_masks_batch = self.storage.cpg_masks[:, indices].detach()

        # privilege_obs_batch, non_privilege_obs_batch = self.split_obs(
        #     obs_batch)
        pd_actions_log_prob_batch, pd_entropy_batch, pd_mu_batch, pd_sigma_batch, \
            cpg_actions_log_prob_batch, cpg_entropy_batch, cpg_mu_batch, cpg_sigma_batch, \
            value_batch, enc_out = self.student_actor_critic.evaluate(
                non_privilege_obs_batch,
                privilege_obs_batch,
                cpg_phase_infos_batch,
                pd_actions_batch,
                cpg_actions_batch,
                return_rnnout=True
            )

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':

          kl = torch.sum(
              pd_sigma_batch - pd_old_sigma_batch + (
                  torch.square(pd_old_sigma_batch.exp()) +
                  torch.square(pd_old_mu_batch - pd_mu_batch)
              ) / (2.0 * torch.square(pd_sigma_batch.exp())) - 0.5, axis=-1
          )
          kl_mean = torch.mean(kl)

          if kl_mean > self.desired_kl * 2.0:
            self.step_size = max(1e-5, self.step_size / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.step_size = min(1e-2, self.step_size * 1.5)

          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.step_size

        # Surrogate loss PD
        pd_ratio = torch.exp(
            pd_actions_log_prob_batch - torch.squeeze(
                pd_old_actions_log_prob_batch
            )
        )
        pd_surrogate = -torch.squeeze(advantages_batch) * pd_ratio
        pd_surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            pd_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        pd_surrogate_loss = (
            torch.max(pd_surrogate, pd_surrogate_clipped) *
            torch.squeeze(1 - cpg_masks_batch)
        ).mean()

        cpg_ratio = torch.exp(
            cpg_actions_log_prob_batch - torch.squeeze(
                cpg_old_actions_log_prob_batch
            )
        )
        cpg_surrogate = -torch.squeeze(advantages_batch) * cpg_ratio
        cpg_surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            cpg_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        cpg_surrogate_loss = (
            torch.max(cpg_surrogate, cpg_surrogate_clipped) *
            torch.squeeze(cpg_masks_batch)
        ).mean()

        surrogate_loss = (
            pd_surrogate_loss + cpg_surrogate_loss
        ) / 2

        lm_loss = (
          self.recon_criterion(
            enc_out, teacher_feature_batch.detach(), l1=False
          ) * cpg_masks_batch
        ).mean()
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

        pd_bc_loss = (
            self.recon_criterion(
                torch.clamp(pd_mu_batch, -1, 1),
                torch.clamp(teacher_pd_actions_batch.detach(), -1, 1), l1=False
            ) * (1 - cpg_masks_batch)
        ).mean()

        cpg_bc_loss = (
            self.recon_criterion(
                torch.clamp(cpg_mu_batch, -1, 1),
                torch.clamp(teacher_cpg_actions_batch.detach(), -1, 1), l1=False
            ) * cpg_masks_batch
        ).mean()
        bc_loss = (pd_bc_loss + cpg_bc_loss) / 2

        pd_from_cpg_bc_loss = self.recon_criterion(
            torch.clamp(pd_mu_batch, -1, 1),
            torch.clamp(pd_actions_from_cpg_batch.detach(), -1, 1), l1=False
        ).mean()

        loss = self.surrogate_loss_coef * surrogate_loss + \
            self.value_loss_coef * value_loss - \
            self.entropy_coef * (pd_entropy_batch.mean() + cpg_entropy_batch.mean()) + \
            self.bc_loss_coef * bc_loss + \
            self.lm_loss_coef * lm_loss + \
            self.pd_from_cpg_bc_loss_coef * pd_from_cpg_bc_loss

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.student_actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_bc_loss += bc_loss.item()
        mean_lm_loss += lm_loss.item()
        mean_pd_from_cpg_bc_loss += pd_from_cpg_bc_loss

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_bc_loss /= num_updates
    mean_lm_loss /= num_updates
    mean_pd_from_cpg_bc_loss /= num_updates

    self.bc_loss_coef *= self.imi_decay_coef
    self.lm_loss_coef *= self.imi_decay_coef

    return mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_lm_loss, mean_pd_from_cpg_bc_loss
