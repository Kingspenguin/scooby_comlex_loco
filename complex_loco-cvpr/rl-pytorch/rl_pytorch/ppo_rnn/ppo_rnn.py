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
# from torch.utils.tensorboard import SummaryWriter

from rl_pytorch.ppo_rnn import RolloutStorage
import copy
import wandb


class PPORNN:

  def __init__(
      self,
      vec_env,
      actor_critic_class,
      encoder,
      num_transitions_per_env,
      num_learning_epochs,
      num_mini_batches,
      clip_param=0.2,
      gamma=0.998,
      lam=0.95,
      init_noise_std=1.0,
      value_loss_coef=1.0,
      entropy_coef=0.0,
      learning_rate=1e-3,
      max_grad_norm=0.5,
      use_clipped_value_loss=True,
      schedule="fixed",
      desired_kl=None,
      model_cfg=None,
      device='cpu',
      sampler='sequential',
      log_dir='run',
      is_testing=False,
      print_log=True,
      apply_reset=False,
      asymmetric=False,
      vidlogdir='video',
      log_video=False,
      vid_log_step=500,
      local=False,
      eval_env_nums=0
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

    self.desired_kl = desired_kl
    self.schedule = schedule
    self.step_size = learning_rate

    self.log_video = log_video
    self.vid_log_step = vid_log_step
    self.vidlogdir = vidlogdir
    self.local = local
    # PPO components
    self.vec_env = vec_env
    model_cfg["encoder_params"]["state_dim"] = self.vec_env.task.state_obs_size
    self.actor_critic = actor_critic_class(
        encoder, self.observation_space.shape,
        self.state_space.shape, self.action_space.shape,
        init_noise_std, model_cfg, asymmetric=asymmetric)
    self.actor_critic.to(self.device)
    # Recurrent state size
    self.hidden_state_size = model_cfg['recurrent']["hidden_size"]
    self.hidden_state_num = model_cfg['recurrent']["num_layers"]

    self.storage = RolloutStorage(
        self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,
        self.state_space.shape, self.action_space.shape,
        self.hidden_state_size,
        self.hidden_state_num,
        self.device, sampler
    )

    self.optimizer = optim.Adam(
        self.actor_critic.parameters(), lr=learning_rate)

    self.eval_env_nums = eval_env_nums

    # PPO parameters
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.num_transitions_per_env = num_transitions_per_env
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    # Log
    self.log_dir = log_dir
    self.print_log = print_log
    # self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
    self.tot_timesteps = 0
    self.tot_time = 0
    self.is_testing = is_testing
    self.current_learning_iteration = 0

    self.rewbuffer = deque(maxlen=100)
    self.lenbuffer = deque(maxlen=100)
    if self.eval_env_nums > 0:
      self.eval_rewbuffer = deque(maxlen=100)
      self.eval_lenbuffer = deque(maxlen=100)

    self.apply_reset = apply_reset

  def test(self, path, device='cuda:0'):
    # self.actor_critic.load_state_dict(torch.load(path, map_location=device))
    check_point = torch.load(path, map_location=device)
    self.actor_critic.load_state_dict(
        check_point['ac']
    )
    if "gail" in self.vec_env.task.reward_scales:
      self.vec_env.task.learners["gail"].discriminator.load_state_dict(
          check_point["gail_learner"]
      )
    self.actor_critic.eval()

  def load(self, path, device='cuda:0'):
    # self.actor_critic.load_state_dict(torch.load(path, map_location=device))
    check_point = torch.load(path, map_location=device)
    self.actor_critic.load_state_dict(
        check_point['ac']
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
        path.split("_")[-1].split(".")[0])
    self.tot_timesteps = self.current_learning_iteration * \
        self.num_transitions_per_env * self.vec_env.num_envs
    self.actor_critic.train()

    if "gail" in self.vec_env.task.reward_scales:
      self.vec_env.task.learners["gail"].discriminator.load_state_dict(
          check_point["gail_learner"]
      )
      self.vec_env.task.learners["gail"].optimizer.load_state_dict(
          check_point["gail_learner_optim"]
      )

  def save(self, path):
    # torch.save(self.actor_critic.state_dict(), path)
    checkpoint = {
        'ac': self.actor_critic.state_dict(),
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

  def run(self, num_learning_iterations, log_interval=1):
    current_obs = self.vec_env.reset()
    current_states = self.vec_env.get_state()

    hidden_states = torch.zeros(
        self.hidden_state_num,
        current_obs.shape[0],
        self.hidden_state_size,
        dtype=torch.float, device=current_obs.device
    )

    if self.is_testing:
      if self.log_video:
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
      step = 0
      frames = []
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
          actions, next_hidden_states = self.actor_critic.act_inference(
              current_obs, hidden_states)
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          next_hidden_states[:, dones.nonzero(
              as_tuple=False).squeeze(-1), :] = 0
          hidden_states.copy_(next_hidden_states)
          if self.log_video:
            log_img = self.vec_env.log_render()
            log_img = log_img.cpu().detach().numpy().astype(np.uint8)
            frames.append(log_img)
            step += 1
          current_obs.copy_(next_obs)
      if self.log_video:
        output_filename = os.path.join(
            self.vidlogdir, 'Output_{}.mp4'.format(
                self.current_learning_iteration)
        )
        print(output_filename)
        imageio.mimsave(output_filename, frames, fps=20)
      # writer.close()

    else:
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
        reward_sum = []
        # if "gail" in self.vec_env.task.reward_scales:
        #   gail_reward_sum = []
        episode_length = []

        if self.eval_env_nums > 0:
          eval_reward_sum = []
          eval_episode_length = []

        start = time.time()
        ep_infos = []

        # Rollout
        for c in range(self.num_transitions_per_env):
          if self.apply_reset:
            current_obs = self.vec_env.reset()
            hidden_states = torch.zeros(
                self.hidden_state_num,
                current_obs.shape[0],
                self.hidden_state_size,
                dtype=torch.float, device=self.rl_device
            )
            current_states = self.vec_env.get_state()
          # Compute the action
          with torch.no_grad():
            actions, actions_log_prob, values, mu, sigma, next_hidden_states = self.actor_critic.act(
                current_obs, current_states, hidden_states)

          if self.eval_env_nums > 0:
            with torch.no_grad():
              actions[-self.eval_env_nums:] = mu[-self.eval_env_nums:]
              actions_log_prob[-self.eval_env_nums:] = self.actor_critic.evaluate(
                  current_obs[-self.eval_env_nums:],
                  current_states[-self.eval_env_nums:],
                  mu[-self.eval_env_nums:],
                  hidden_states[:, -self.eval_env_nums:, :].contiguous()
              )[0]
          # Step the vec_environment
          next_obs, rews, dones, infos = self.vec_env.step(actions)
          # print(c)

          if "gail" in self.vec_env.task.reward_scales:
            self.vec_env.task.learners["gail"].save_transition(
                infos["last_dof_pos"], infos["dof_pos"]
            )
            # gail_rews = infos["gail_rew"]

          fake_dones = infos["fake_done"]
          dones = fake_dones
          # print(dones)
          next_states = self.vec_env.get_state()
          # Record the transition
          self.storage.add_transitions(
              current_obs, current_states, actions, rews, dones,
              values, actions_log_prob, mu, sigma, hidden_states)
          current_obs.copy_(next_obs)
          current_states.copy_(next_states)
          # Book keeping
          ep_infos.append(infos)
          next_hidden_states[:, dones.nonzero(
              as_tuple=False).squeeze(-1), :] = 0
          hidden_states.copy_(next_hidden_states)

          if self.print_log:
            cur_reward_sum[:] += rews
            cur_episode_length[:] += 1

            # if "gail" in self.vec_env.task.reward_scales:
            #   cur_reward_sum[:] -= gail_rews
            #   cur_gail_reward_sum[:] += gail_rews

            if self.eval_env_nums > 0:
              cur_eval_reward_sum[:] += rews[-self.eval_env_nums:]
              # if "gail" in self.vec_env.task.reward_scales:
              #   cur_eval_reward_sum[:] -= gail_rews[-self.eval_env_nums:]
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
          # reward_sum = [x[0] for x in reward_sum]
          # episode_length = [x[0] for x in episode_length]
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

        with torch.no_grad():
          _, _, last_values, _, _, _ = self.actor_critic.act(
              current_obs, current_states, hidden_states)
        stop = time.time()
        collection_time = stop - start

        mean_trajectory_length, mean_reward = self.storage.get_statistics()
        # Learning step
        start = stop
        if "gail" in self.vec_env.task.reward_scales:
          gail_rewards = self.vec_env.task.learners["gail"].reward()
          # print(gail_rewards.shape)
          mean_gail_reward = gail_rewards.mean()
          self.vec_env.task.learners["gail"].update()
          self.storage.rewards += gail_rewards * \
              self.vec_env.task.reward_scales["gail"]
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss = self.update()
        self.storage.clear()
        stop = time.time()
        learn_time = stop - start
        if self.print_log:
          rewbuffer = copy.deepcopy(self.rewbuffer)
          lenbuffer = copy.deepcopy(self.lenbuffer)
          # if "gail" in self.vec_env.task.reward_scales:
          #   gail_rewbuffer = copy.deepcopy(self.gail_rewbuffer)
          if self.eval_env_nums > 0:
            eval_rewbuffer = copy.deepcopy(self.eval_rewbuffer)
            eval_lenbuffer = copy.deepcopy(self.eval_lenbuffer)
          self.log(locals())
        if it % log_interval == 0:
          self.save(os.path.join(self.log_dir,
                    'model_{}.pt'.format(it)))
        ep_infos.clear()
      self.save(os.path.join(self.log_dir,
                'model_{}.pt'.format(num_learning_iterations)))

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
    mean_std = self.actor_critic.log_std.exp().mean()

    fps = int(self.num_transitions_per_env * self.vec_env.num_envs /
              (locs['collection_time'] + locs['learn_time']))

    str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

    log_dict = {
        'Value function loss:': locs['mean_value_loss'],
        'Surrogate loss:': locs['mean_surrogate_loss'],
        'Mean action noise std:': mean_std.item(),
        'Mean reward/step:': locs['mean_reward'],
        'Total timesteps:': self.tot_timesteps,
        'Iteration time:': iteration_time,
        'Mean episode length/episode:': locs['mean_trajectory_length']
    }
    log_string = (f"""{'#' * width}\n"""
                  f"""{str.center(width, ' ')}\n\n"""
                  f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                  f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                  f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                  f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                  f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                  f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

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

  def update(self):
    mean_value_loss = 0
    mean_surrogate_loss = 0

    batch = self.storage.mini_batch_generator(self.num_mini_batches)
    for epoch in range(self.num_learning_epochs):
      # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
      #        in self.storage.mini_batch_generator(self.num_mini_batches):

      for indices in batch:
        obs_batch = self.storage.observations.view(
            -1, *self.storage.observations.size()[2:])[indices]
        if self.asymmetric:
          states_batch = self.storage.states.view(
              -1, *self.storage.states.size()[2:])[indices]
        else:
          states_batch = None
        actions_batch = self.storage.actions.view(
            -1, self.storage.actions.size(-1))[indices]
        hidden_states_batch = self.storage.hidden_states.view(
            self.hidden_state_num, -1, self.hidden_state_size
        )[:, indices].detach()
        target_values_batch = self.storage.values.view(-1, 1)[indices]
        returns_batch = self.storage.returns.view(-1, 1)[indices]
        old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[
            indices]
        advantages_batch = self.storage.advantages.view(-1, 1)[indices]
        old_mu_batch = self.storage.mu.view(-1,
                                            self.storage.actions.size(-1))[indices]
        old_sigma_batch = self.storage.sigma.view(
            -1, self.storage.actions.size(-1))[indices]

        actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch, _ = self.actor_critic.evaluate(
            obs_batch, states_batch, actions_batch, hidden_states_batch
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

        loss = surrogate_loss + self.value_loss_coef * \
            value_loss - self.entropy_coef * entropy_batch.mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates

    return mean_value_loss, mean_surrogate_loss
