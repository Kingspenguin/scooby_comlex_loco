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

import wandb
import copy


class DistillRNNLMReuseTrainer:
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
      teacher_head_lr_scale=0.1,
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
      learn_by_self=False,
      teacher_use_rnn=False,
      mask_base_vel=False,
      use_fake_done=False,
      imi_decay_coef=1,
      eval_env_nums=0,
      use_l1=False,
      distillation_weight=None,
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

    self.learn_by_self = learn_by_self
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

    self.distillation_weight = distillation_weight
    if self.distillation_weight is not None:
      self.distillation_weight = torch.Tensor(
          self.distillation_weight
      ).unsqueeze(0).to(self.device)

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

    if self.vec_env.task.use_stacked_state:
      student_model_cfg["encoder_params"]["state_dim"] = self.vec_env.task.state_obs_size - \
          self.vec_env.task.privilege_info_len * self.vec_env.task.stacked_state_input_num
    else:
      student_model_cfg["encoder_params"]["state_dim"] = self.vec_env.task.state_obs_size - \
          self.vec_env.task.privilege_info_len

    # Recurrent state size
    self.hidden_state_size = student_model_cfg['recurrent']["hidden_size"]
    self.hidden_state_num = student_model_cfg['recurrent']["num_layers"]

    self.is_testing = is_testing
    self.vidlogdir = vidlogdir
    self.log_video = log_video
    self.vid_log_step = vid_log_step
    self.mask_base_vel = mask_base_vel

    teacher_actions_shape = self.action_space.shape
    teacher_state_shape = self.state_space.shape
    if self.vec_env.task.use_stacked_state:
      # self.vec_env.task.stacked_state_input_num
      student_obs_shape = (
          self.vec_env.task.state_obs_size -
          self.vec_env.task.privilege_info_len *
          self.vec_env.task.stacked_state_input_num + self.vec_env.task.image_obs_size,
      )
    else:
      student_obs_shape = (
          self.vec_env.task.state_obs_size -
          self.vec_env.task.privilege_info_len + self.vec_env.task.image_obs_size,
      )

    self.distill_from_cpg = False
    self.use_l1 = use_l1

    self.teacher_use_rnn = teacher_use_rnn
    if not self.is_testing:
      # Teacher must not using asymmetric training
      # Teacher uses privilege information
      self.teacher_actor_critic = teacher_actor_critic_class(
          teacher_encoder, teacher_state_shape, teacher_state_shape, teacher_actions_shape,
          init_noise_std, teacher_model_cfg, asymmetric=False
      )
      self.teacher_actor_critic.to(self.device)

    self.student_actor_critic = student_actor_critic_class(
        student_encoder,
        self.observation_space.shape, teacher_state_shape,
        self.action_space.shape,
        init_noise_std, student_model_cfg, asymmetric=asymmetric
    )

    self.student_actor_critic.to(self.device)
    self.storage = LMRolloutStorage(
        self.vec_env.num_envs, num_transitions_per_env,
        student_obs_shape, teacher_state_shape,
        self.action_space.shape,
        self.hidden_state_size,
        self.hidden_state_num,
        # self.teacher_feature_shape,
        self.device, sampler
    )
    if self.teacher_use_rnn:
      self.optimizer = optim.Adam([
          {"params": self.student_actor_critic.encoder.parameters(),
           "lr": learning_rate},
          {"params": self.student_actor_critic.log_std,
           "lr": learning_rate},
          {"params": self.student_actor_critic.actor.parameters(),
           "lr": learning_rate * teacher_head_lr_scale},
          {"params": self.student_actor_critic.critic.parameters(),
           "lr": learning_rate * teacher_head_lr_scale},
          {"params": self.student_actor_critic.rnn.parameters(),
           "lr": learning_rate * teacher_head_lr_scale}
      ])
    else:
      self.optimizer = optim.Adam([
          {"params": self.student_actor_critic.encoder.parameters(),
           "lr": learning_rate},
          {"params": self.student_actor_critic.log_std,
           "lr": learning_rate},
          {"params": self.student_actor_critic.actor.parameters(),
           "lr": learning_rate * teacher_head_lr_scale},
          {"params": self.student_actor_critic.critic.parameters(),
           "lr": learning_rate * teacher_head_lr_scale},
          {"params": self.student_actor_critic.rnn.parameters(),
           "lr": learning_rate}
      ])
    # else:
    #   self.optimizer = optim.Adam(
    #       self.student_actor_critic.parameters(), lr=learning_rate)

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
    if "gail" in self.vec_env.task.reward_scales:
      self.vec_env.task.learners["gail"].discriminator.load_state_dict(
          check_point["gail_learner"]
      )
    self.student_actor_critic.eval()

  def teacher_load(self, path, device='cuda:0'):
    check_point = torch.load(path, map_location=device)
    self.teacher_actor_critic.load_state_dict(
        check_point['ac']
    )
    self.teacher_actor_critic.eval()

  def teacher_gives_student_actor_critic_head(self):
    self.student_actor_critic.critic.load_state_dict(
        self.teacher_actor_critic.critic.state_dict()
    )
    self.student_actor_critic.actor.load_state_dict(
        self.teacher_actor_critic.actor.state_dict()
    )
    if self.teacher_use_rnn:
      self.student_actor_critic.rnn.load_state_dict(
          self.teacher_actor_critic.rnn.state_dict()
      )

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
    self.tot_timesteps = self.current_learning_iteration * \
        self.num_transitions_per_env * self.vec_env.num_envs
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
    if self.vec_env.task.use_stacked_state:
      propri_obs = obs_batch[..., :self.vec_env.task.state_obs_size].reshape(
          obs_batch.shape[:-1] + (
              self.vec_env.task.stacked_state_input_num,
              self.vec_env.task.state_obs_size // self.vec_env.task.stacked_state_input_num,
          )
      )
      state = propri_obs[
          ..., :-self.vec_env.task.privilege_info_len
      ].flatten(-2)
      # privilege_info = obs_batch[..., self.vec_env.task.state_obs_size -
      #                            self.vec_env.task.privilege_info_len: self.vec_env.task.state_obs_size]
      privilege_info = propri_obs[..., 0, :]
    else:
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
    if self.vec_env.task.use_stacked_state:
      privilege_obs = torch.cat([
          privilege_info, height
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
    return self.teacher_actor_critic.act_inference_with_encoder(privileged_obs)

  def run(self, num_learning_iterations, log_interval=1):
    current_obs = self.vec_env.reset()
    current_states = self.vec_env.get_state()

    hidden_states = torch.zeros(
        self.hidden_state_num,
        current_obs.shape[0],
        self.hidden_state_size,
        dtype=torch.float, device=current_obs.device
    )

    dones = torch.as_tensor(
        [0.0] * self.vec_env.num_envs,
        device=current_obs.device, dtype=torch.float
    )

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
            hidden_states = torch.zeros(
                self.hidden_state_num,
                current_obs.shape[0],
                self.hidden_state_size,
                dtype=torch.float, device=self.rl_device
            )
            dones = torch.as_tensor([0.0] * self.vec_env.num_envs,
                                    device=current_obs.device, dtype=torch.float)

          # Compute the action
          _, non_privilege_obs = self.split_obs(current_obs)
          actions, next_hidden_states = self.student_actor_critic.act_inference(
              non_privilege_obs, hidden_states
          )
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          dones = dones.to(torch.float)
          next_hidden_states[:, dones.nonzero(
              as_tuple=False).squeeze(-1), :] = 0
          hidden_states.copy_(next_hidden_states)
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
      if not self.learn_by_self:
        # Give the teacher's value function to student
        self.teacher_gives_student_actor_critic_head()

      cur_reward_sum = torch.zeros(
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
        episode_length = []

        if self.eval_env_nums > 0:
          eval_reward_sum = []
          eval_episode_length = []

        # Rollout
        for _ in range(self.num_transitions_per_env):
          if self.apply_reset:
            current_obs = self.vec_env.reset()
            # current_states = self.vec_env.get_state()
            hidden_states = torch.zeros(
                self.hidden_state_num,
                current_obs.shape[0],
                self.hidden_state_size,
                dtype=torch.float, device=self.rl_device
            )
            dones = torch.as_tensor([0.0] * self.vec_env.num_envs,
                                    device=current_obs.device, dtype=torch.float)

          # Compute the action
          with torch.no_grad():
            privilege_obs, non_privilege_obs = self.split_obs(
                current_obs
            )
            # teacher_acts, teacher_feature = self.get_teacher_action(privilege_obs)
            # teacher_feature = self.teacher_fake_encoder(privilege_obs)

            student_actions, student_actions_log_prob, values, mu, sigma, next_hidden_states = self.student_actor_critic.act(
                non_privilege_obs, privilege_obs, hidden_states,
            )
          # print("teacher acts")

          prev_dones = dones.clone()

          # student_actions = teacher_acts
          if self.eval_env_nums > 0:
            with torch.no_grad():
              student_actions[-self.eval_env_nums:] = mu[-self.eval_env_nums:]
              student_actions_log_prob[-self.eval_env_nums:] = self.student_actor_critic.evaluate(
                  non_privilege_obs[-self.eval_env_nums:],
                  current_states[-self.eval_env_nums:],
                  mu[-self.eval_env_nums:],
                  hidden_states[:, -self.eval_env_nums:, :]
              )[0]

          next_obs, rews, dones, infos = self.vec_env.step(student_actions)

          if "gail" in self.vec_env.task.reward_scales:
            self.vec_env.task.learners["gail"].save_transition(
                infos["last_dof_pos"], infos["dof_pos"]
            )

          # print("aft step")
          dones = dones.to(torch.float)
          fake_dones = infos["fake_done"]
          if self.use_fake_done:
            dones = fake_dones.to(torch.float)
          next_states = self.vec_env.get_state()

          self.storage.add_transitions(
              # current_obs,
              non_privilege_obs, privilege_obs,
              student_actions, rews, dones, prev_dones, values, student_actions_log_prob, mu, sigma, hidden_states,
              # teacher_acts, teacher_feature
          )
          current_obs.copy_(next_obs)
          current_states.copy_(next_states)
          ep_infos.append(infos)
          next_hidden_states[:, dones.nonzero(
              as_tuple=False).squeeze(-1), :] = 0
          hidden_states.copy_(next_hidden_states)

          if self.print_log:
            cur_reward_sum[:] += rews
            cur_episode_length[:] += 1

            if self.eval_env_nums > 0:
              cur_eval_reward_sum[:] += rews[-self.eval_env_nums:]
              cur_eval_episode_length[:] += 1

            new_ids = dones.nonzero(as_tuple=False)
            if np.prod(new_ids.shape) > 0:
              reward_sum.append(
                  cur_reward_sum[new_ids].cpu().numpy()
              )

              episode_length.append(
                  cur_episode_length[new_ids].cpu().numpy()
              )
              cur_reward_sum[new_ids] = 0
              cur_episode_length[new_ids] = 0

            if self.eval_env_nums > 0:
              eval_new_ids = dones[-self.eval_env_nums:
                                   ].nonzero(as_tuple=False)
              if np.prod(eval_new_ids.shape) > 0:
                eval_reward_sum.append(
                    cur_eval_reward_sum[eval_new_ids].cpu().numpy())
                eval_episode_length.extend(
                    cur_eval_episode_length[eval_new_ids].cpu().numpy())
                cur_eval_reward_sum[eval_new_ids] = 0
                cur_eval_episode_length[eval_new_ids] = 0

        if self.print_log:
          if len(reward_sum) > 0:
            self.rewbuffer.extend(
                np.vstack(reward_sum)[:, 0].tolist()
            )
            self.lenbuffer.extend(
                np.vstack(episode_length)[:, 0].tolist()
            )

          if self.eval_env_nums > 0 and len(eval_reward_sum) > 0:
            self.eval_rewbuffer.extend(
                np.vstack(eval_reward_sum)[:, 0].tolist()
            )
            self.eval_lenbuffer.extend(
                np.vstack(eval_episode_length)[:, 0].tolist()
            )

        stop = time.time()
        collection_time = stop - start

        privilege_obs, non_privilege_obs = self.split_obs(
            current_obs)

        with torch.no_grad():
          _, _, last_values, _, _, _ = self.student_actor_critic.act(
              non_privilege_obs, privilege_obs, hidden_states
          )

        mean_trajectory_length, mean_reward = self.storage.get_statistics()

        # Learning step
        start = stop
        if "gail" in self.vec_env.task.reward_scales:
          gail_rewards = self.vec_env.task.learners["gail"].reward()
          mean_gail_reward = gail_rewards.mean()
          self.vec_env.task.learners["gail"].update()
          self.storage.rewards += gail_rewards * \
              self.vec_env.task.reward_scales["gail"]
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_lm_loss = self.update()
        self.storage.clear()
        stop = time.time()
        learn_time = stop - start
        if self.print_log:
          rewbuffer = copy.deepcopy(self.rewbuffer)
          lenbuffer = copy.deepcopy(self.lenbuffer)
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
                  f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                  f"""{'Behavior cloning loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                  f"""{'Latent Match loss:':>{pad}} {locs['mean_lm_loss']:.4f}\n"""
                  f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                  f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                  f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
    log_dict = {
        'Value function loss:': locs['mean_value_loss'],
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
        log_string += f"""{'Mean GAIL reward/step:':>{pad}} {locs['mean_gail_reward']:.2f}\n"""
        log_dict["Mean Gail reward/step"] = locs['mean_gail_reward']

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
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_bc_loss = 0
    mean_lm_loss = 0

    batch = self.storage.mini_batch_generator(self.num_mini_batches)
    # teacher_actions = []
    # teacher_features = []
    # for indices in batch:
    #   with torch.no_grad():
    #     privilege_obs_batch = self.storage.privileged_observations[:, indices].detach()
    #     teacher_action, teacher_feature = self.teacher_actor_critic.act_inference_with_encoder(
    #       privilege_obs_batch
    #     )
    #     teacher_action = teacher_action.detach()
    #     teacher_feature = teacher_feature.detach()
    #   teacher_actions.append(teacher_action)
    #   teacher_features.append(teacher_feature)

    for epoch in range(self.num_learning_epochs):
      for idx, indices in enumerate(batch):
        # obs_batch = self.storage.observations[:, indices]
        privilege_obs_batch = self.storage.privileged_observations[:, indices].detach(
        )
        non_privilege_obs_batch = self.storage.non_privileged_observations[:, indices].detach(
        )
        actions_batch = self.storage.actions[:, indices].detach()
        hidden_states_batch = self.storage.hidden_states[:, :, indices].detach(
        )
        # print(indices, actions_batch.shape, hidden_states_batch.shape, self.storage.hidden_states.shape)

        dones_batch = self.storage.dones[:, indices].detach()
        prev_dones_batch = self.storage.prev_dones[:, indices].detach()

        target_values_batch = self.storage.values[:, indices].detach()
        returns_batch = self.storage.returns[:, indices].detach()
        old_actions_log_prob_batch = self.storage.actions_log_prob[:, indices].detach(
        )
        advantages_batch = self.storage.advantages[:, indices].detach()
        old_mu_batch = self.storage.mu[:, indices].detach()
        old_sigma_batch = self.storage.sigma[:, indices].detach()

        # teacher_actions_batch = self.storage.teacher_actions[:, indices].detach(
        # )
        # teacher_feature_batch = self.storage.teacher_features[:, indices].detach(
        # )
        if self.teacher_use_rnn:
          teacher_actions_batch, teacher_feature_batch = self.teacher_actor_critic.act_inference_with_encoder(
              privilege_obs_batch, hidden_states_batch
          )
        else:
          teacher_actions_batch, teacher_feature_batch = self.teacher_actor_critic.act_inference_with_encoder(
              privilege_obs_batch
          )
        # teacher_actions_batch = teacher_actions[idx]
        # teacher_feature_batch = teacher_features[idx]

        actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch, hidden_states_batch, rnnout = self.student_actor_critic.evaluate(
            non_privilege_obs_batch,
            privilege_obs_batch,
            actions_batch, hidden_states_batch, return_rnnout=True
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

        if self.surrogate_loss_coef > 0:
          # Surrogate loss
          ratio = torch.exp(actions_log_prob_batch -
                            torch.squeeze(old_actions_log_prob_batch))
          surrogate = -torch.squeeze(advantages_batch) * ratio
          surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                             1.0 + self.clip_param)

          surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        else:
          surrogate_loss = 0

        if self.lm_loss_coef > 0:
          lm_loss = self.recon_criterion(
              rnnout, teacher_feature_batch.detach()
          ).mean()
        else:
          lm_loss = 0

        if self.value_loss_coef > 0:
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
        else:
          value_loss = 0

        bc_loss = self.recon_criterion(
            torch.clamp(mu_batch, -1, 1),
            torch.clamp(teacher_actions_batch.detach(), -1, 1)
        )
        if self.distillation_weight is not None:
          # print(bc_loss)
          bc_loss = bc_loss * self.distillation_weight.detach()
          # print(bc_loss)
        bc_loss = bc_loss.mean()

        loss = self.surrogate_loss_coef * surrogate_loss + self.value_loss_coef * \
            value_loss - self.entropy_coef * entropy_batch.mean() + self.bc_loss_coef * \
            bc_loss + self.lm_loss_coef * lm_loss

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.student_actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.value_loss_coef > 0:
          mean_value_loss += value_loss.item()
        if self.surrogate_loss_coef > 0:
          mean_surrogate_loss += surrogate_loss.item()

        mean_bc_loss += bc_loss.item()
        if self.lm_loss_coef > 0:
          mean_lm_loss += lm_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_bc_loss /= num_updates
    mean_lm_loss /= num_updates

    self.bc_loss_coef *= self.imi_decay_coef
    self.lm_loss_coef *= self.imi_decay_coef

    return mean_value_loss, mean_surrogate_loss, mean_bc_loss, mean_lm_loss
