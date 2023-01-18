from datetime import datetime
import os
import time

import gym.spaces as spaces
from gym.spaces import Space

import numpy as np
import statistics

import torch
import torch.nn as nn

from rl_pytorch.dapg.teacher_storage import TeacherRolloutStorage

import wandb

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class StorageGenerator:
    def __init__(
        self,
        vec_env,
        actor_critic_class,
        height_encoder,
        vis_encoder,
        num_teacher_transitions,
        init_noise_std=1.0,
        model_cfg=None,
        device='cpu',
        teacher_log_dir='run',
        asymmetric=False,
        teacher_resume="None",
        vidlogdir='video',
        vid_log_step=1000,
        log_video=False,
        storage_save_path=None,
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

        # PPO components
        self.vec_env = vec_env

        from copy import deepcopy
        vis_model_cfg = deepcopy(model_cfg)
        height_model_cfg = deepcopy(model_cfg)

        vis_model_cfg['encoder_params'] = model_cfg['vis_encoder_params']
        height_model_cfg['encoder_params'] = model_cfg['height_encoder_params']

        self.vidlogdir = vidlogdir
        self.log_video = log_video
        self.vid_log_step = vid_log_step

        self.teacher_actor_critic = actor_critic_class(
            height_encoder, self.observation_space.shape, self.state_space.shape, self.action_space.shape,
            init_noise_std, height_model_cfg, asymmetric=asymmetric
        )
        self.teacher_actor_critic.to(self.device)

        self.visual_observation_space = spaces.Box(
            np.ones(self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size) * -np.Inf,
            np.ones(self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size) * np.Inf,
        )
        # exit()
        self.num_teacher_transitions = num_teacher_transitions
        self.teacher_storage = TeacherRolloutStorage(
            self.vec_env.num_envs, num_teacher_transitions, self.visual_observation_space.shape,
            self.action_space.shape, 'cpu', 'sequential'
        )


        # Log
        self.teacher_log_dir = teacher_log_dir
        # student Log
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.teacher_resume =teacher_resume
        assert teacher_resume is not None

        self.storage_save_path = storage_save_path

    def test_teacher(self, path):
        self.teacher_actor_critic.load_state_dict(torch.load(path))
        self.teacher_actor_critic.eval()

    def teacher_load(self, path):
        self.teacher_actor_critic.load_state_dict(torch.load(path))
        self.teacher_actor_critic.eval()

    def save(self, path):
        torch.save(self.student_actor_critic.state_dict(), path)

    def split_height_vision_obs(self, obs_batch):
        # print(obs_batch.shape)
        state = obs_batch[:, :self.vec_env.task.state_obs_size]
        vis = obs_batch[
            :, self.vec_env.task.state_obs_size : self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size
        ]
        height = obs_batch[
            :, 
            self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size : self.vec_env.task.state_obs_size + self.vec_env.task.image_obs_size + self.vec_env.task.height_map_obs_size
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

    def get_expert_buffer(self):
        self.vec_env.task.enable_test()
        current_obs = self.vec_env.reset()

        reward_sum = []
        episode_length = []

        start = time.time()
        ep_infos = []

        cur_reward_sum = torch.zeros(
            self.vec_env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(
            self.vec_env.num_envs, dtype=torch.float, device=self.device)

        reward_sum = []
        episode_length = []

        # Rollout
        for _ in range(self.num_teacher_transitions):
            
            current_height_obs, current_vis_obs = self.split_height_vision_obs(current_obs)
            # Compute the action
            with torch.no_grad():
                actions = self.teacher_actor_critic.act_inference(
                    current_height_obs
                )
            
            # Step the vec_environment
            next_obs, rews, dones, infos = self.vec_env.step(actions)
            # Record the transition

            # Store Vision Obs
            self.teacher_storage.add_transitions(
                current_vis_obs, actions, rews, dones
            )
            current_obs.copy_(next_obs)
            # Book keeping
            ep_infos.append(infos)

            # if self.print_log:
            cur_reward_sum[:] += rews
            cur_episode_length[:] += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)
            reward_sum.extend(
                cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
            episode_length.extend(
                cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum[new_ids] = 0
            cur_episode_length[new_ids] = 0

        stop = time.time()
        collection_time = stop - start
        mean_trajectory_length, mean_reward = self.teacher_storage.get_statistics()

        # Learning step
        # if self.print_log:
        self.teacher_log(locals())
        ep_infos.clear()

        self.vec_env.task.disable_test()

    def teacher_log(self, locs, width=80, pad=35):
        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(self.num_teacher_transitions * self.vec_env.num_envs /
                  (locs['collection_time']))

        str = f" \033[1m Teacher Buffer Loading \033[0m "

        log_string = (
            f"""{'#' * width}\n"""
            f"""{str.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                'collection_time']:.3f}s)\n"""
            f"""{'Mean reward:':>{pad}} {statistics.mean(locs['reward_sum']):.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['episode_length']):.2f}\n"""
            f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
        )
        log_string += ep_string
        print(log_string)

    def run(self):
        self.teacher_load("{}/model_{}.pt".format(self.teacher_log_dir, self.teacher_resume))
        self.get_expert_buffer()
        import pathlib
        pathlib.Path(self.storage_save_path).mkdir(parents=True, exist_ok=True)
        print(self.teacher_storage.observations.shape)
        print(self.teacher_storage.actions.shape)
        torch.save(
            self.teacher_storage.observations, 
            os.path.join(self.storage_save_path, "obs.pt")
        )
        torch.save(
            self.teacher_storage.actions, 
            os.path.join(self.storage_save_path, "acts.pt")
        )
