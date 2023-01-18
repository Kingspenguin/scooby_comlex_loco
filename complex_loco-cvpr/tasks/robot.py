
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pathlib
from typing import List
import numpy as np
import os
import sys
from utils.torch_jit_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
import torch
import time
from PIL import Image
from torch import Tensor
from tasks.task_wrappers import build_task_wrapper
from tasks.env_wrappers import build_env_wrapper
from tasks.utilizers import build_utilizer
from tasks.learner import build_learner
from tasks.controller import LegController
from tasks.terrain import Terrain
import torch.nn.functional as F
import matplotlib.pyplot
import cv2

HIP_LINK_LENGTH = 0.08505
THIGH_LINK_LENGTH = 0.2
CALF_LINK_LENGTH = 0.2


class Robot:
  def __init__(
      self, cfg, sim_params, physics_engine,
      device_type, device_id, headless, test_mode, log_video,
      terrain_params=None, tracking_cam_viz=False, get_statistics=False,
  ):

    self.gym = gymapi.acquire_gym()
    self.test_mode = test_mode
    self.cfg = cfg
    self.task_name = cfg["task"]["name"]
    self.env_name = cfg["env"]["name"]
    self.num_envs = cfg["env"]["numEnvs"]
    self.device_type = device_type
    self.device_id = device_id
    self.debug_viz = False
    self.feet_viz = False
    self.tracking_cam_viz = tracking_cam_viz
    self.get_statistics = get_statistics
    self.device = "cpu"
    if self.device_type == "cuda" or self.device_type == "GPU":
      self.device = "cuda" + ":" + str(self.device_id)

    self._prepare_wrappers()

    self.input_inverse_depth = self.cfg["env"].get(
        "input_inverse_depth", False
    )

    self.input_original_depth = self.cfg["env"].get(
        "input_original_depth", False
    )

    self.height_clip_low = self.cfg["env"].get(
        "height_clip_low", -2
    )

    self.height_clip_high = self.cfg["env"].get(
        "height_clip_low", 2
    )

    self.use_stacked_state = self.env_vision_cfg.get(
        "use_stacked_state", False)

    self.get_image = self.env_vision_cfg["get_image"]
    self.camera_number = self.env_vision_cfg["camera_number"]
    self.enable_neck_camera = self.env_vision_cfg["neck_camera"] \
        if "neck_camera" in self.env_vision_cfg else False

    self.log_video = log_video

    self.use_height_map = self.env_vision_cfg["use_height_map"]
    if self.use_height_map:
      self.height_update_freq = self.env_vision_cfg["height_update_freq"]
      self.use_multi_height_map = self.env_vision_cfg["multi_height_map"]
    self.use_wide_height_map = self.env_vision_cfg.get(
        "use_wide_height_map", False)
    self.valid_height_range = 10
    # double check!
    self.graphics_device_id = self.device_id
    self.headless = headless
    if self.headless and not self.get_image:
      self.graphics_device_id = -1

    self.headless = cfg["headless"]

    # self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

    # optimization flags for pytorch JIT
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    self.terrain_params = terrain_params

    self.sim_params = sim_params
    self.physics_engine = physics_engine
    self._register()
    self.create_sim()
    self.gym.prepare_sim(self.sim)
    self.cfg["device_type"] = device_type
    self.cfg["device_id"] = device_id
    self.cfg["headless"] = headless
    # self._build_legcontroller()
    self.control_mode = self.cfg["env"]["control"]["legcontroller"]
    self._build_viewer()
    self._build_buf()
    self._build_utilizers()
    self._prepare_reward_function()
    self._build_learners()
    self.updated = False
    self.reset(torch.arange(self.num_envs, device=self.device))
    self.img_count = 0

  def create_sim(self):

    self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
    self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id,
                                   self.physics_engine, self.sim_params)
    if self.sim is None:
      print("*** Failed to create sim")
      quit()

    self._sample_init_state()
    self._create_terrain()
    self._create_envs(self.cfg["env"]['envSpacing'],
                      int(np.sqrt(self.num_envs)))
    if self.log_video:

      self.log_width = 600
      self.log_height = 600
      camera_properties = gymapi.CameraProperties()
      camera_properties.width = self.log_width
      camera_properties.height = self.log_height
      camera_properties.near_plane = 1.0
      camera_properties.enable_tensors = True
      self.log_camera = self.gym.create_camera_sensor(
          self.envs[0], camera_properties)

      if "viewer" in self.cfg:
        camera_offset = gymapi.Vec3(
            self.cfg["viewer"]["offset"][0],
            self.cfg["viewer"]["offset"][1],
            self.cfg["viewer"]["offset"][2]
        )

        body_handle = self.gym.get_actor_rigid_body_handle(
            self.envs[0], self.a1_handles[0], 0
        )
        camera_rotation = gymapi.Quat.from_euler_zyx(
            np.deg2rad(self.cfg["viewer"]["rotation"][0]),
            np.deg2rad(self.cfg["viewer"]["rotation"][1]),
            np.deg2rad(self.cfg["viewer"]["rotation"][2]),
        )
        self.gym.attach_camera_to_body(
            self.log_camera, self.envs[0], body_handle, gymapi.Transform(
                camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)
      else:
        camera_offset = gymapi.Vec3(1.0, -1.0, 1.2)

        body_handle = self.gym.get_actor_rigid_body_handle(
            self.envs[0], self.a1_handles[0], 0
        )
        camera_rotation = gymapi.Quat.from_euler_zyx(
            np.deg2rad(0), np.deg2rad(45), np.deg2rad(120)
        )
        self.gym.attach_camera_to_body(
            self.log_camera, self.envs[0], body_handle, gymapi.Transform(
                camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)
      self.log_image_buf = torch.zeros(
          (self.log_width, self.log_height, 3), device=self.device
      )
      print("created log camera")

  def set_sim_params_up_axis(self, sim_params, axis):
    if axis == 'z':
      sim_params.up_axis = gymapi.UP_AXIS_Z
      sim_params.gravity.x = 0
      sim_params.gravity.y = 0
      sim_params.gravity.z = -9.81
      return 2
    return 1

  def _create_terrain(self):
    self.terrain = Terrain(
        self.terrain_params,
    )
    self.terrain.convert_to_trimesh()
    # self.robot_origin = torch.zeros((self.num_envs, 3), device=self.device)
    # self.robot_origin = self.env_wrapper.create_ground(self)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = self.terrain.vertices.shape[0]
    tm_params.nb_triangles = self.terrain.triangles.shape[0]
    tm_params.transform.p.x = -self.terrain.border_size
    tm_params.transform.p.y = -self.terrain.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = self.terrain_params["static_friction"]
    tm_params.dynamic_friction = self.terrain_params["dynamic_friction"]
    tm_params.restitution = self.terrain_params["restitution"]

    self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(
        order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
    # self.height_samples = torch.tensor(self.terrain.heightsamples).view(
    #     self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    self.height_samples = torch.tensor(self.terrain.percept_height_samples).view(
        self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    random_idx = [i % (self.terrain.env_cols * self.terrain.env_rows)
                  for i in range(self.num_envs)]
    # print("selected env idx:", random_idx)
    self.robot_origin = torch.Tensor(self.terrain.env_origins.reshape((-1, 3))[
        random_idx
    ])
    self.robot_env_high = torch.Tensor(self.terrain.env_height_high).flatten()[
        random_idx
    ]
    self.robot_env_low = torch.Tensor(self.terrain.env_height_low).flatten()[
        random_idx
    ]
    self.robot_reachable_distance = torch.Tensor(self.terrain.env_reachable_distance.reshape((-1))[
        random_idx
    ])

    self.robot_random_origin_x_low = torch.Tensor(self.terrain.env_random_origin_x_low.reshape((-1))[
        random_idx
    ])
    self.robot_random_origin_x_range = torch.Tensor(self.terrain.env_random_origin_x_range.reshape((-1))[
        random_idx
    ])
    self.robot_random_origin_y_low = torch.Tensor(self.terrain.env_random_origin_y_low.reshape((-1))[
        random_idx
    ])
    self.robot_random_origin_y_range = torch.Tensor(self.terrain.env_random_origin_y_range.reshape((-1))[
        random_idx
    ])

    self.robot_origin = self.robot_origin.to(self.device)
    self.robot_env_high = self.robot_env_high.to(self.device)
    self.robot_env_low = self.robot_env_low.to(self.device)
    self.robot_reachable_distance = self.robot_reachable_distance.to(
        self.device)
    self.robot_random_origin_x_low = self.robot_random_origin_x_low.to(
        self.device)
    self.robot_random_origin_x_range = self.robot_random_origin_x_range.to(
        self.device)
    self.robot_random_origin_y_low = self.robot_random_origin_y_low.to(
        self.device)
    self.robot_random_origin_y_range = self.robot_random_origin_y_range.to(
        self.device)

  def _register(self):

    # normalization
    self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
    self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
    self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
    self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
    self.height_scale = self.cfg["env"]["learn"]["heightScale"]
    self.action_scale = self.cfg["env"]["control"]["actionScale"]
    # self.hip_action_scale = self.cfg["env"]["control"]["hipActionScale"]
    if self.get_image:
      # vision-guided mode
      # Total Number
      self.frame_stack = self.env_vision_cfg["frame_stack"]
      self.frame_skip = self.env_vision_cfg.get("frame_skip", 1)
      self.vision_update_freq = self.env_vision_cfg["update_freq"]
      self.image_type = self.env_vision_cfg["image_type"]
      self.render_width = self.env_vision_cfg["width"]
      self.render_height = self.env_vision_cfg["height"]

      self.obs_width = self.env_vision_cfg["width"]
      self.obs_height = self.env_vision_cfg["height"]

      self.reshape_vision = "target_shape" in self.env_vision_cfg
      if self.reshape_vision:
        self.target_shape = self.env_vision_cfg["target_shape"]
        self.horizontal_clip = int(self.render_height / 10)

        self.obs_width = self.env_vision_cfg["target_shape"][1]
        self.obs_height = self.env_vision_cfg["target_shape"][0]

      # the second camera's position is fixed
      self.camera_angle = self.env_vision_cfg["camera_angle"]
      self.frame_skip_indexes = []
      for i in range(self.camera_number):
        self.frame_skip_indexes.append((torch.arange(
            0, self.frame_stack * self.camera_number, self.frame_skip * self.camera_number
        ) + i).reshape(-1, 1))
      self.frame_skip_indexes = torch.cat(
          self.frame_skip_indexes, dim=1).reshape(-1)
      self.frame_input_num = self.frame_skip_indexes.shape[0]

    if self.use_stacked_state:
      self.state_skip_indexes = torch.arange(
          0, self.frame_stack, self.frame_skip
      )
      self.stacked_state_input_num = self.state_skip_indexes.shape[0]

    self.frame_count = 0

    # use diagonal action
    self.diagonal_act = self.cfg["env"]["learn"]["diagonal_act"]

    # time
    self.control_freq_inv = self.cfg["env"]["control"]["controlFrequencyInv"]
    self.dt = self.sim_params.dt * self.control_freq_inv
    self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
    self.max_episode_length = int(
        self.max_episode_length_s / (self.dt) + 0.5)

    # push robot
    if "push_robots" in self.cfg["env"]["learn"]:
      self.push_robots = self.cfg["env"]["learn"]["push_robots"]
      self.push_interval_s = self.cfg["env"]["learn"]["push_interval_s"]
      self.max_push_vel_xy = self.cfg["env"]["learn"]["max_push_vel_xy"]
      self.push_interval = self.push_interval_s / self.dt
    else:
      self.push_robots = False
    # commands
    self.command_type = self.cfg["env"]["command"]
    self.command_change_step = self.cfg["env"]["commandChangeStep"]

    # sensor settings
    self.historical_step = self.cfg["env"]["sensor"]["historical_step"]
    self.use_sys_information = self.cfg["env"]["sensor"]["sys_id"]

    self.risk_reward = self.cfg["env"]["risk_reward"]
    self.rush_reward = self.cfg["env"]["rush_reward"]
    self.vel_reward_exp_coeff = self.cfg["env"]["vel_reward_exp_coeff"]

    # default joint positions
    self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

    self.privilege_info_len = 3 if self.use_sys_information else 0

    self.use_feet_observation = False
    self.use_contact_force = False
    self.use_contact_state = False
    self.use_feet_air_time = False

    if "feet_observation" in self.cfg["env"]["sensor"] and self.cfg["env"]["sensor"]["feet_observation"]:
      self.use_feet_observation = True
      self.privilege_info_len += 12
    if "contact_force" in self.cfg["env"]["sensor"] and self.cfg["env"]["sensor"]["contact_force"]:
      self.use_contact_force = True
      self.contact_forces_scale = 0.05
      self.privilege_info_len += 12
    if "contact_state" in self.cfg["env"]["sensor"] and self.cfg["env"]["sensor"]["contact_state"]:
      self.use_contact_state = True
      self.contact_state_scale = 1.0
      self.privilege_info_len += 4
    if "feet_air_time" in self.cfg["env"]["sensor"] and self.cfg["env"]["sensor"]["feet_air_time"]:
      self.use_feet_air_time = True
      self.feet_air_time_scale = 5.0
      self.privilege_info_len += 4

    if "lin_vel" in self.cfg["env"]["sensor"] and self.cfg["env"]["sensor"]["lin_vel"]:
      self.use_lin_vel = True
      self.privilege_info_len += 3
    else:
      self.use_lin_vel = False

    if "ang_vel" in self.cfg["env"]["sensor"] and self.cfg["env"]["sensor"]["ang_vel"]:
      self.use_ang_vel = True
      self.privilege_info_len += 3
    else:
      self.use_ang_vel = False

    self.use_prev_action = self.cfg["env"]["sensor"].get("prev_actions", True)
    if self.use_prev_action:
      # if self.diagonal_act:
      #   self.cfg["env"]["numObservations"] = 18 * \
      #       self.historical_step + 24 + self.privilege_info_len
      #   self.cfg["env"]["numActions"] = 6
      # else:
      self.cfg["env"]["numObservations"] = 24 * \
          self.historical_step + 15 + self.privilege_info_len
      # self.cfg["env"]["numObservations"] = 24 * \
      #     self.historical_step + 24 + self.privilege_info_len
      self.cfg["env"]["numActions"] = 12
    else:
      # if self.diagonal_act:
      #   self.cfg["env"]["numObservations"] = 12 * \
      #       self.historical_step + 24 + self.privilege_info_len
      #   self.cfg["env"]["numActions"] = 6
      # else:
      self.cfg["env"]["numObservations"] = 12 * \
          self.historical_step + 15 + self.privilege_info_len
      # self.cfg["env"]["numObservations"] = 24 * \
      #     self.historical_step + 24 + self.privilege_info_len
      self.cfg["env"]["numActions"] = 12

    if self.cfg["env"]["control"]["legcontroller"] in ["cpg", "pd_foot"]:
      if self.use_prev_action:
        self.cfg["env"]["numObservations"] += 4 * self.historical_step

      self.cfg["env"]["numActions"] += 4

      self.use_aug_time_phase = self.cfg["env"]["sensor"].get(
          "aug_time_phase", True
      )
      if self.use_aug_time_phase:
        self.cfg["env"]["numObservations"] += 4 * \
            2  # two more residual phase information

      self.use_original_time_phase = self.cfg["env"]["sensor"].get(
          "original_time_phase", False
      )
      if self.use_original_time_phase:
        self.cfg["env"]["numObservations"] += 4 * \
            2  # two more residual phase information

    if self.use_stacked_state:
      self.cfg["env"]["numObservations"] = self.stacked_state_input_num * \
          self.cfg["env"]["numObservations"]
    self.state_obs_size = self.cfg["env"]["numObservations"]

    if self.get_image:
      if self.image_type == "depth":
        image_obs_size = self.obs_width * \
            self.obs_height * self.frame_input_num  # * self.camera_number
      elif self.image_type == "rgb":
        image_obs_size = self.obs_width * \
            self.obs_height * self.frame_input_num * 3  # * self.camera_number
      elif self.image_type == "rgbd":
        image_obs_size = self.obs_width * \
            self.obs_height * self.frame_input_num * 4  # * self.camera_number
      else:
        raise NotImplementedError

      self.cfg["env"]["numObservations"] += image_obs_size
      self.image_obs_size = image_obs_size
    else:
      self.image_obs_size = 0

    if self.use_height_map:
      # height_map_obs_size = self.width * \
      #   self.height
      height_map_obs_size = 546
      if self.use_multi_height_map:
        if self.use_wide_height_map:
          height_map_obs_size += 399
        else:
          height_map_obs_size += 209
      self.cfg["env"]["numObservations"] += height_map_obs_size
      self.height_map_obs_size = height_map_obs_size
    else:
      self.height_map_obs_size = 0

    self.num_obs = self.cfg["env"]["numObservations"]
    if self.use_stacked_state:
      self.num_states = self.cfg["env"].get(
          "numStates", self.height_map_obs_size + self.state_obs_size // self.stacked_state_input_num)
    else:
      self.num_states = self.cfg["env"].get(
          "numStates", self.height_map_obs_size + self.state_obs_size)
    self.num_actions = self.cfg["env"]["numActions"]

  def _build_utilizers(self):
    self.randomizer = {}
    self.curriculum_scheduler = {}
    if self.cfg["randomize_state"]["randomize"]:
      self.randomizer["randomize_state"] = build_utilizer(
          "randomize_state", self.cfg)
      self.randomize_input = True
    else:
      self.randomize_input = False
    if self.cfg["randomize_reward"]["randomize"]:
      self.randomizer["randomize_reward"] = build_utilizer(
          "randomize_reward", self.cfg)

  def _build_learners(self):
    self.learners = {}
    if "learners" not in self.cfg:
      return
    for key, _ in self.cfg["learners"].items():
      self.learners[key] = build_learner(
          key, self.num_envs, self.device, self.cfg, self.dt)

  def _build_buf(self):
    # get gym state tensors
    # allocate buffers
    if "historical_buffer" in self.cfg["env"]:
      self.save_historical_buffer = True
      self.historical_buffer_size = int(
          self.cfg["env"]["historical_buffer"]["length_s"] / self.dt)
      self.historical_states_buffer = torch.zeros(
          (self.num_envs, self.historical_buffer_size, self.state_obs_size - self.privilege_info_len), device=self.device, dtype=torch.float)
      self.cfg["env"]["numObservations"] += self.historical_buffer_size * \
          (self.state_obs_size - self.privilege_info_len)
      self.num_obs = self.cfg["env"]["numObservations"]
    else:
      self.save_historical_buffer = False

    self.obs_buf = torch.zeros(
        (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
    if self.use_stacked_state:
      self.stacked_obs_buf = torch.zeros(
          (self.num_envs, self.frame_stack, self.state_obs_size // self.stacked_state_input_num), device=self.device
      )

    self.states_buf = torch.zeros(
        (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
    self.rew_buf = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.float)
    self.reset_buf = torch.ones(
        self.num_envs, device=self.device, dtype=torch.long)
    self.progress_buf = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.long)
    self.randomize_buf = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.long)
    self.feet_pos_buf = torch.zeros(
        (self.num_envs, 12), device=self.device, dtype=torch.float)
    self.leg_jacobian = torch.zeros(
        self.num_envs, 4, 3, 3, device=self.device, dtype=torch.float
    )

    self.dr_randomizations = {}

    self.last_step = -1
    self.last_rand_step = -1
    self.last_rew_rand_step = -1

    actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
    dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
    net_contact_forces = self.gym.acquire_net_contact_force_tensor(
        self.sim)
    torques = self.gym.acquire_dof_force_tensor(self.sim)

    self.gym.refresh_dof_state_tensor(self.sim)
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)
    self.gym.refresh_dof_force_tensor(self.sim)
    # create some wrapper tensors for different slices
    self.root_states = gymtorch.wrap_tensor(
        actor_root_state).view(-1, 13)
    self.last_root_states = torch.zeros_like(self.root_states)
    self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    self.last_dof_pos = self.dof_state.view(
        self.num_envs, self.num_dof, 2)[..., 0]
    self.last_dof_vel = self.dof_state.view(
        self.num_envs, self.num_dof, 2)[..., 1]

    self.dof_pos = self.dof_state.view(
        self.num_envs, self.num_dof, 2)[..., 0]
    self.dof_vel = self.dof_state.view(
        self.num_envs, self.num_dof, 2)[..., 1]
    self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
        self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
    self.torques = gymtorch.wrap_tensor(
        torques).view(self.num_envs, self.num_dof)

    if self.historical_step > 1:
      self.dof_pos_buf = torch.zeros(
          (self.num_envs, self.historical_step, self.num_dof), device=self.device)
      # if self.diagonal_act:
      #   self.actions_buf = torch.zeros(
      #       (self.num_envs, self.historical_step, self.num_actions), device=self.device)
      #   self.last_actions = torch.zeros(
      #       self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
      # else:
      self.actions_buf = torch.zeros(
          (self.num_envs, self.historical_step, self.num_actions), device=self.device)
      self.last_actions = torch.zeros(
          self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

      self.torques_buf = torch.zeros(
          (self.num_envs, self.historical_step, self.num_dof), device=self.device)

    if self.get_image:
      if self.frame_stack >= 1:
        if self.image_type == "depth":
          self.image_buf = torch.zeros(
              (self.num_envs, self.frame_stack * self.camera_number, self.obs_width * self.obs_height), device=self.device
          )
        elif self.image_type == "rgb":
          self.image_buf = torch.zeros(
              (self.num_envs, self.frame_stack * self.camera_number, self.obs_width * self.obs_height * 3), device=self.device
          )
        elif self.image_type == "rgbd":
          self.image_buf = torch.zeros(
              (self.num_envs, self.frame_stack * self.camera_number, self.obs_width * self.obs_height * 4), device=self.device
          )
        else:
          raise NotImplementedError

    if self.use_height_map:
      self.height_buf = torch.zeros(
          (self.num_envs, self.height_map_obs_size), device=self.device
      )
      self.height_points = self.init_height_points()
      self.measured_heights = torch.zeros(
          self.num_envs, self.num_height_points, device=self.device)
    self.commands = torch.zeros(
        self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
    self.commands_y = self.commands.view(
        self.num_envs, 3)[..., 1]
    self.commands_x = self.commands.view(
        self.num_envs, 3)[..., 0]
    self.commands_yaw = self.commands.view(
        self.num_envs, 3)[..., 2]

    self.default_dof_pos = torch.zeros_like(
        self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

    # relative to base frame
    self.default_feet_pos = torch.as_tensor([
        [0.1478, 0.11459, -0.45576,
         0.1478, -0.11459, -0.45576,
         -0.2895, 0.11459, -0.45576,
         -0.2895, -0.11459, -0.45576]], device=self.device, dtype=torch.float)
    for i in range(self.num_dof):
      name = self.dof_names[i]
      angle = self.named_default_joint_angles[name]
      self.default_dof_pos[:, i] = angle

    self.num_legs = 4
    self._com_offset = - \
        torch.as_tensor([0.012731, 0.002186, 0.000515],
                        device=self.device)
    self._hip_offset = torch.as_tensor([[0.183, -0., 0.], [0.183, 0., 0.],
                                        [-0.183, -0., 0.], [-0.183, 0., 0.]
                                        ], device=self.device) + self._com_offset
    self._default_hip_positions = torch.as_tensor([
        [0.17, -0.14, 0],
        [0.17, 0.14, 0],
        [-0.17, -0.14, 0],
        [-0.17, 0.14, 0],
    ], device=self.device)

    jt = self.gym.acquire_jacobian_tensor(self.sim, "a1")
    self.jacobian_tensor = gymtorch.wrap_tensor(jt)

    # initialize some data used later on
    self.extras = {}
    self.initial_root_states = self.root_states.clone()
    self.initial_root_states[self.a1_indices] = to_torch(
        self.init_states_for_each_env, device=self.device, requires_grad=False)
    self.gravity_vec = to_torch(
        get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
    self.actions = torch.zeros(self.num_envs, self.num_actions,
                               dtype=torch.float, device=self.device, requires_grad=False)
    self.time_out_buf = torch.zeros_like(self.reset_buf)

    # self.feet_dof_pos = self.dof_pos[..., self.feet_indices]
    # if self.diagonal_act:
    #   self.action_scale = torch.as_tensor(
    #       self.action_scale * 2, device=self.device)
    # else:
    self.action_scale = torch.as_tensor(
        self.action_scale * 4, device=self.device)
    # if self.leg_controller.mode in ["cpg", "pd_foot"]:
    #   self.phase_scale = torch.as_tensor(
    #       [self.cfg["env"]["control"]["phaseScale"]] * 4, device=self.device)
    #   nominal_phase = self.cfg["env"]["control"]["phaseScale"]
    #   self.nominal_residual_phase = torch.as_tensor(
    #       [-nominal_phase, nominal_phase, nominal_phase, -nominal_phase], device=self.device)
    #   self.action_scale = torch.cat(
    #       [self.action_scale, self.phase_scale], dim=-1)

  def _sample_init_state(self):
    # base init state
    pos = self.cfg["env"]["baseInitState"]["pos"]
    rot = self.cfg["env"]["baseInitState"]["rot"]
    v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
    v_ang = self.cfg["env"]["baseInitState"]["vAngular"]

    state = pos + rot + v_lin + v_ang
    self.base_init_state = state
    self.init_states_for_each_env = torch.as_tensor(
        [self.base_init_state for _ in range(self.num_envs)], device=self.device)

  def _prepare_motor_params(self):

    self.Kp = self.cfg["env"]["control"]["stiffness"]
    self.Kd = self.cfg["env"]["control"]["damping"]

  def _prepare_wrappers(self):
    self.task_wrapper = build_task_wrapper(
        self.task_name, self.device, self.cfg)
    self.env_wrapper = build_env_wrapper(
        self.env_name, self.device, self.cfg)
    self.env_vision_cfg = self.env_wrapper.env_cfg["vision"]
    self.env_viewer_cfg = self.env_wrapper.env_cfg["viewer"]

  def _create_envs(self, spacing, num_per_row):
    asset_root = "assets"
    asset_file = "urdf/a1/a1.urdf"
    asset_path = os.path.join(asset_root, asset_file)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.collapse_fixed_joints = False
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False

    a1_asset = self.gym.load_asset(
        self.sim, asset_root, asset_file, asset_options)
    dof_props_asset = self.gym.get_asset_dof_properties(a1_asset)
    self.num_dof = self.gym.get_asset_dof_count(a1_asset)
    self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset)
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(*self.base_init_state[: 3])

    body_names = self.gym.get_asset_rigid_body_names(a1_asset)
    self.dof_names = self.gym.get_asset_dof_names(a1_asset)
    # feet_names = [(i, s) for i, s in enumerate(body_names) if "lower" in s]
    feet_names = [(i, s) for i, s in enumerate(body_names) if "toe" in s]
    # print([(i, s) for i, s in enumerate(body_names) if "lower" in s])
    # print([(i, s) for i, s in enumerate(body_names) if "toe" in s])
    # exit()
    self.feet_indices_in_bodies = torch.as_tensor(
        [fn[0] for fn in feet_names], device=self.device)
    self.feet_indices = torch.zeros(
        len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
    knee_names = [s for s in body_names if "upper" in s]
    self.knee_indices = torch.zeros(
        len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
    self.base_index = 0

    env_lower = gymapi.Vec3(-0.0, -0.0, 0.0)
    env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
    self.a1_indices = []
    self.a1_handles = []

    self.envs = []
    if self.get_image:
      self.camera_handles = []
      self.depth_image = []
      if self.camera_number == 3:
        self.head_depth_image = []
        self.front_depth_image = []
        self.bottom_depth_image = []
      if self.camera_number == 2:
        self.head_depth_image = []
        self.front_depth_image = []

    self._prepare_motor_params()
    self.init_states_for_each_env[..., : 3] += self.robot_origin

    for i in range(self.num_envs):
      # create env instances
      env_ptr = self.gym.create_env(
          self.sim, env_lower, env_upper, num_per_row)
      if self.robot_origin is not None:
        start_pose.p = gymapi.Vec3(*self.robot_origin[i])
        # start_pose.p = gymapi.Vec3(0,0,0.33)
      a1_handle = self.gym.create_actor(
          env_ptr, a1_asset, start_pose, "a1", i, 0, 0)
      dof_props = self._process_dof_props(dof_props_asset, i)
      # print(dof_props)/
      self.gym.set_actor_dof_properties(
          env_ptr, a1_handle, dof_props)
      self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
      self.envs.append(env_ptr)
      self.a1_handles.append(a1_handle)
      a1_idx = self.gym.get_actor_index(
          env_ptr, a1_handle, gymapi.DOMAIN_SIM)
      self.a1_indices.append(a1_idx)

      if self.get_image:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.render_width
        camera_properties.height = self.render_height
        camera_properties.horizontal_fov = 86
        camera_properties.enable_tensors = True
        camera_properties.far_plane = 5

        if self.enable_neck_camera and self.camera_number == 1:
          head_camera = self.gym.create_camera_sensor(
              env_ptr, camera_properties)
          camera_offset = gymapi.Vec3(0.03, -0.04, -0.1)
          camera_rotation = gymapi.Quat.from_axis_angle(
              gymapi.Vec3(0, 1, 0), np.deg2rad(90))
          body_handle = self.gym.get_actor_rigid_body_handle(
              env_ptr, a1_handle, 1)
          self.gym.attach_camera_to_body(head_camera, env_ptr, body_handle, gymapi.Transform(
              camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
        else:
          head_camera = self.gym.create_camera_sensor(
              env_ptr, camera_properties)
          camera_offset = gymapi.Vec3(0.24, 0, 0)
          camera_rotation = gymapi.Quat.from_axis_angle(
              gymapi.Vec3(0, 1, 0), np.deg2rad(self.camera_angle))
          body_handle = self.gym.get_actor_rigid_body_handle(
              env_ptr, a1_handle, 0)
          self.gym.attach_camera_to_body(head_camera, env_ptr, body_handle, gymapi.Transform(
              camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)

        if self.camera_number == 1:
          self.camera_handles.append(head_camera)
        elif self.camera_number == 2:
          front_camera = self.gym.create_camera_sensor(
              env_ptr, camera_properties)
          camera_offset = gymapi.Vec3(0.03, -0.04, -0.1)
          camera_rotation = gymapi.Quat.from_axis_angle(
              gymapi.Vec3(0, 1, 0), np.deg2rad(90))
          body_handle = self.gym.get_actor_rigid_body_handle(
              env_ptr, a1_handle, 1)
          self.gym.attach_camera_to_body(front_camera, env_ptr, body_handle, gymapi.Transform(
              camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
          self.camera_handles.append(
              [head_camera, front_camera])
        elif self.camera_number == 3:
          front_camera = self.gym.create_camera_sensor(
              env_ptr, camera_properties)
          camera_offset = gymapi.Vec3(0.03, -0.04, -0.1)
          camera_rotation = gymapi.Quat.from_axis_angle(
              gymapi.Vec3(0, 1, 0), np.deg2rad(90))
          body_handle = self.gym.get_actor_rigid_body_handle(
              env_ptr, a1_handle, 1)
          self.gym.attach_camera_to_body(front_camera, env_ptr, body_handle, gymapi.Transform(
              camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)

          bottom_camera = self.gym.create_camera_sensor(
              env_ptr, camera_properties)
          camera_offset = gymapi.Vec3(-0.3, -0.04, -0.04)
          camera_rotation = gymapi.Quat.from_axis_angle(
              gymapi.Vec3(0, 1, 0), np.deg2rad(90))
          body_handle = self.gym.get_actor_rigid_body_handle(
              env_ptr, a1_handle, 1)
          self.gym.attach_camera_to_body(bottom_camera, env_ptr, body_handle, gymapi.Transform(
              camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
          self.camera_handles.append(
              [head_camera, front_camera, bottom_camera])

    if self.use_height_map:
      self.map_scale = 0.04
      self.env_spacing = self.cfg["env"]['envSpacing']
      # self.height_maps = self.terrain.height_field_raw * self.terrain.vertical_scale
      self.height_maps = self.terrain.percept_height * self.terrain.vertical_scale
      self.height_maps = torch.Tensor(self.height_maps).to(self.device)

    if self.get_image:
      self.camera_handles = np.asarray(self.camera_handles)

    for i in range(len(feet_names)):
      self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
          self.envs[0], self.a1_handles[0], feet_names[i][1])
    for i in range(len(knee_names)):
      self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
          self.envs[0], self.a1_handles[0], knee_names[i])
    self.base_index = self.gym.find_actor_rigid_body_handle(
        self.envs[0], self.a1_handles[0], "trunk")
    self.a1_indices = to_torch(
        self.a1_indices, dtype=torch.long, device=self.device)
    penalized_contact_names = []
    for name in self.cfg["env"]["asset"]["penalize_contacts_on"]:
      penalized_contact_names.extend(
          [s for s in body_names if name in s])
    self.penalised_contact_indices = torch.zeros(len(
        penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
    for i in range(len(penalized_contact_names)):
      self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
          self.envs[0], self.a1_handles[0], penalized_contact_names[i])
    termination_contact_names = []
    for name in self.cfg["env"]["asset"]["terminate_after_contacts_on"]:
      termination_contact_names.extend(
          [s for s in body_names if name in s])
    self.termination_contact_indices = torch.zeros(len(
        termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
    for i in range(len(termination_contact_names)):
      self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
          self.envs[0], self.a1_handles[0], termination_contact_names[i])
    self.feet_air_time = torch.zeros(
        self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

  def init_height_points(self):
    # 1mx1.6m rectangle (without center line)
    # y = 0.05 * torch.tensor([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    #                         device=self.device, requires_grad=False)  # 10-50cm on each side
    y = 0.05 * torch.arange(-10, 11, device=self.device, requires_grad=False)
    # x = 0.05 * torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    #                         device=self.device, requires_grad=False)  # 20-80cm on each side
    x = 0.05 * torch.arange(-10, 16, device=self.device, requires_grad=False)

    if self.use_multi_height_map:
      sparse_x = 0.10 * \
          torch.arange(1, 20, device=self.device, requires_grad=False)
      if self.use_wide_height_map:
        # 10-50cm on each side
        sparse_y = 0.10 * \
            torch.arange(-10, 11, device=self.device, requires_grad=False)
        # 10-50cm on each side,
      else:
        # 10-50cm on each side
        sparse_y = 0.10 * \
            torch.arange(-5, 6, device=self.device, requires_grad=False)

    grid_x, grid_y = torch.meshgrid(x, y)
    if self.use_multi_height_map:
      grid_sparse_x, grid_sparse_y = torch.meshgrid(sparse_x, sparse_y)

    self.num_height_points = grid_x.numel()
    if self.use_multi_height_map:
      self.num_height_points += grid_sparse_x.numel()
    # print(self.num_height_points)
    points = torch.zeros(self.num_envs, self.num_height_points,
                         3, device=self.device, requires_grad=False)
    points[:, :546, 0] = grid_x.flatten()
    points[:, :546, 1] = grid_y.flatten()
    if self.use_multi_height_map:
      points[:, 546:, 0] = grid_sparse_x.flatten()
      points[:, 546:, 1] = grid_sparse_y.flatten()
    return points

  def get_heights(self, env_ids=None):

    if env_ids:
      points = quat_apply_yaw(self.base_quat[env_ids].repeat(
          1, self.num_height_points), self.height_points[env_ids]) + self.root_states[self.a1_indices[env_ids], :3].unsqueeze(1)
    else:
      points = quat_apply_yaw(self.base_quat.repeat(
          1, self.num_height_points), self.height_points) + self.root_states[self.a1_indices, :3].unsqueeze(1)
    points += self.terrain.border_size
    points /= (self.terrain.horizontal_scale)

    points = points.long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)

    px = torch.clip(px, 0, self.height_maps.shape[0] - 2)
    py = torch.clip(py, 0, self.height_maps.shape[1] - 2)
    heights0 = self.height_maps[px, py]
    heights1 = self.height_maps[px, py + 1]
    heights2 = self.height_maps[px + 1, py]
    heights = torch.min(torch.min(heights1, heights2), heights0)
    heights = heights.view(self.num_envs, -1)

    return heights.view(self.num_envs, -1)

  def pre_physics_step(self, actions):
    self.actions = actions.clone().to(self.device)
    self.last_root_states = self.root_states.clone()
    self.last_dof_pos = self.dof_pos.clone()
    self.last_dof_vel = self.dof_vel.clone()
    self.last_actions[:] = self.actions[:]
    if self.historical_step > 1:
      self.actions_buf = torch.cat(
          [self.actions_buf[:, 1:], self.actions.unsqueeze(1)], dim=1)

  def get_states(self):
    return self.states_buf

  def render(self):
    if self.viewer:
      # check for window closed
      if self.gym.query_viewer_has_closed(self.viewer):
        sys.exit()

      # check for keyboard events
      for evt in self.gym.query_viewer_action_events(self.viewer):
        if evt.action == "QUIT" and evt.value > 0:
          sys.exit()
        elif evt.action == "toggle_viewer_sync" and evt.value > 0:
          self.enable_viewer_sync = not self.enable_viewer_sync
        elif evt.action == "debug_visualize" and evt.value > 0:
          self.debug_viz = not self.debug_viz
          if not self.debug_viz and not self.feet_viz:
            self.gym.clear_lines(self.viewer)
        elif evt.action == "feet_visualize" and evt.value > 0:
          self.feet_viz = not self.feet_viz
          if not self.debug_viz and not self.feet_viz:
            self.gym.clear_lines(self.viewer)
        elif evt.action == "push_robots" and evt.value > 0:
          self._push_robots()

      # fetch results
      if self.device != 'cpu':
        self.gym.fetch_results(self.sim, True)

      # step graphics
      if self.enable_viewer_sync:
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)
      else:
        self.gym.poll_viewer_events(self.viewer)

  # # def _compute_torques(self, actions):
  # def _compute_torques(self):
  #   """ Compute torques from actions.
  #       Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
  #       [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
  #   Args:
  #       actions (torch.Tensor): Actions
  #   Returns:
  #       [torch.Tensor]: Torques sent to the simulation
  #   """
  #   # pd controller
  #   # actions_scaled = actions * self.action_scale
  #   # control_type = self.control_mode
  #   # assert control_type == "pd_joint"
  #   # self.target_dof_pos = actions_scaled + self.default_dof_pos
  #   torques = self.Kp * \
  #       (self.target_dof_pos - self.dof_pos) - self.Kd * self.dof_vel
  #   return torch.clip(torques, -self.torque_limits, self.torque_limits)

  # def _compute_actions(self, actions):
  #   """ Compute torques from actions.
  #       Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
  #       [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
  #   Args:
  #       actions (torch.Tensor): Actions
  #   Returns:
  #       [torch.Tensor]: Torques sent to the simulation
  #   """
  #   # pd controller
  #   actions_scaled = actions * self.action_scale
  #   # control_type = self.control_mode
  #   # assert control_type == "pd_joint"
  #   self.target_dof_pos = actions_scaled + self.default_dof_pos

  def _compute_torques(self, actions):
    """ Compute torques from actions.
        Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
        [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
    Args:
        actions (torch.Tensor): Actions
    Returns:
        [torch.Tensor]: Torques sent to the simulation
    """
    # pd controller
    actions_scaled = actions * self.action_scale
    control_type = self.control_mode
    if control_type == "pd_joint":
      self.target_dof_pos = actions_scaled + self.default_dof_pos
      torques = self.Kp * \
          (self.target_dof_pos - self.dof_pos) - self.Kd * self.dof_vel
    elif control_type == "V":
      torques = self.Kp * (actions_scaled - self.dof_vel) - self.Kd * \
          (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
    elif control_type == "T":
      torques = actions_scaled
    else:
      raise NameError(f"Unknown controller type: {control_type}")
    return torch.clip(torques, -self.torque_limits, self.torque_limits)

  def step(self, actions):
    # start_time = time.time()
    if self.randomize_input:
      actions = self.randomizer["randomize_state"].state_randomizations['actions']['noise_lambda'](
          actions)

    # apply actions
    self.pre_physics_step(actions)
    # step physics and render each frame
    for i in range(self.control_freq_inv):
      self.computed_torques = self._compute_torques(
          self.actions).view(self.torques.shape)
      self.gym.set_dof_position_target_tensor(
          self.sim, gymtorch.unwrap_tensor(self.target_dof_pos))
      self.gym.simulate(self.sim)
      if self.device == 'cpu':
        self.gym.fetch_results(self.sim, True)
      if i % 2 == 0:
        # From experience
        self.render()
      self.gym.refresh_dof_state_tensor(self.sim)

    # self.computed_torques = self._compute_torques().view(self.torques.shape)
    self.post_physics_step()
    if self.randomize_input:
      self.obs_buf = self.randomizer["randomize_state"].state_randomizations['observations']['noise_lambda'](
          self.obs_buf
      )

  def enable_test(self):
    self.test_mode = True

  def disable_test(self):
    self.test_mode = False

  def post_physics_step(self):

    self.gym.refresh_dof_state_tensor(self.sim)  # done in step
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_jacobian_tensors(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)
    self.gym.refresh_dof_force_tensor(self.sim)

    self.frame_count += 1
    self.progress_buf += 1
    self.base_quat = self.root_states[self.a1_indices, 3:7]
    self.base_lin_vel = quat_rotate_inverse(
        self.base_quat, self.root_states[self.a1_indices, 7:10])
    self.base_ang_vel = quat_rotate_inverse(
        self.base_quat, self.root_states[self.a1_indices, 10:13])
    self.projected_gravity = quat_rotate_inverse(
        self.base_quat, self.gravity_vec)

    # change_commmand_env_ids = (torch.fmod(
    #     self.progress_buf, self.command_change_step) == 0).float().nonzero(as_tuple=False).squeeze(-1)

    # if self.task_wrapper.task_name == "following_command" and len(change_commmand_env_ids) > 0:
    #   self.reset_command(change_commmand_env_ids)

    if self.push_robots and (self.frame_count % self.push_interval == 0):
      self._push_robots()
    self.task_wrapper.check_termination(self)
    if self.get_statistics:
      self.task_wrapper.get_statistics(self)

    # if not self.test_mode and not self.task_name in ["following_command", "moving_forward"]:
    #   for k in self.learners.keys():
    #     if k == "gail":
    #       self.learners[k].save_transition(
    #           self.last_dof_pos, self.dof_pos
    #       )
      # if self.learners[k].check_update():
      #   self.learners[k].update()

    if self.use_height_map and self.frame_count % self.height_update_freq == 0:
      self.measured_heights = self.get_heights()
      # self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
    self.extras["last_dof_pos"] = self.last_dof_pos
    self.extras["dof_pos"] = self.dof_pos
    self.compute_reward()
    env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    self.fake_done = self.reset_buf.clone()
    self.extras["fake_done"] = self.fake_done
    self.extras["pose"] = self.base_quat.clone()
    self.extras["pos"] = self.root_states[self.a1_indices, 0:3].clone()

    if len(env_ids) > 0:
      self.reset(env_ids)
    if self.tracking_cam_viz:
      self._update_viewer()
    if self.historical_step > 1:
      self.dof_pos_buf = torch.cat(
          [self.dof_pos_buf[:, 1:], self.dof_pos.unsqueeze(1)], dim=1)
      self.torques_buf = torch.cat(
          [self.torques_buf[:, 1:], self.torques.unsqueeze(1)], dim=1)

    if self.use_feet_observation:
      feet_observation = self._footPositionsInBaseFrame()
      self.feet_pos_buf[:] = feet_observation.view(self.num_envs, -1)

    if self.get_image and self.frame_count % self.vision_update_freq == 0:
      if self.headless:
        self.gym.step_graphics(self.sim)
        self.gym.fetch_results(self.sim, True)
        # self.gym.sync_frame_time(self.sim)
        # render the camera sensors
      self.gym.render_all_camera_sensors(self.sim)
      self.gym.start_access_image_tensors(self.sim)
      if not self.updated:
        image_vectors = torch.stack([
            self.update_image(i) for i in range(self.num_envs)
        ], dim=0)
      else:
        if self.camera_number == 1:
          image_vectors = torch.stack(
              self.depth_image, dim=0).unsqueeze(1)
        elif self.camera_number == 2:
          image_vectors = concat_2cam_images(
              self.head_depth_image, self.front_depth_image)
        elif self.camera_number == 3:
          image_vectors = concat_3cam_images(
              self.head_depth_image, self.front_depth_image, self.bottom_depth_image)

      if self.reshape_vision:
        image_vectors = F.interpolate(
            image_vectors[:, :, :, self.horizontal_clip:-self.horizontal_clip],
            size=self.target_shape
        )
      image_vectors = image_vectors.flatten(-2)

      randomization_mask = torch.rand_like(
          image_vectors, device=image_vectors.device)
      image_vectors[randomization_mask >= 0.99] = -100

      if self.device == "cpu":
        image_vectors = image_vectors.to(self.device)

      self.image_buf = depth_image_processes(
          image_vectors, self.image_buf, self.camera_number,
          input_inverse_depth=self.input_inverse_depth,
          input_original_depth=self.input_original_depth
      )
      if len(env_ids) > 0:
        self.image_buf[env_ids, :] = self.image_buf[
            env_ids, :self.camera_number
        ].repeat(1, self.frame_stack, 1)
      self.gym.end_access_image_tensors(self.sim)

    self.compute_observations()

    if self.viewer and self.enable_viewer_sync:
      if self.debug_viz or self.feet_viz:
        self.gym.clear_lines(self.viewer)
      if self.debug_viz:
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.02, 4, 4, None, color=(1, 1, 0))
        height_points = quat_apply_yaw(self.base_quat.repeat(
            1, self.num_height_points), self.height_points)
        for i in range(self.num_envs):
          base_pos = (self.root_states[i, :3]).cpu().numpy()
          heights = self.measured_heights[i].cpu().numpy()

          for j in range(heights.shape[0]):
            x = height_points[i, j, 0] + base_pos[0]
            y = height_points[i, j, 1] + base_pos[1]
            z = heights[j]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym,
                               self.viewer, self.envs[i], sphere_pose)

      if self.feet_viz:
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.04, 8, 8, None, color=(1, 0, 0))
        feet_points = self.feet_pos_buf.clone()
        feet_points = feet_points.reshape(self.num_envs, -1, 3)
        # feet_points += self._hip_offset.unsqueeze(
        #   0).repeat(self.num_envs, 1, 1)
        for j in range(feet_points.shape[1]):
          # For foot height visualization
          feet_points[:, j] = quat_rotate(self.base_quat, feet_points[:, j])
        feet_points = feet_points.cpu().numpy()
        for i in range(self.num_envs):
          base_pos = (self.root_states[i, :3]).cpu().numpy()
          for j in range(feet_points.shape[1]):
            x = feet_points[i, j, 0] + base_pos[0]
            y = feet_points[i, j, 1] + base_pos[1]
            z = feet_points[i, j, 2] + base_pos[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym,
                               self.viewer, self.envs[i], sphere_pose)

    if self.log_video:
      # if self.headless:
      #     self.gym.step_graphics(self.sim)
      #     self.gym.fetch_results(self.sim, True)
      #     self.gym.sync_frame_time(self.sim)
        # render the camera sensors
      # self.gym.render_all_camera_sensors(self.sim)
      if not self.get_image:
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
      self.update_log_image()

  def update_log_image(self):
    # output image and then write it to disk using Pillow
    # communicate physics to graphics system
    rgba_image = self.gym.get_camera_image_gpu_tensor(
        self.sim, self.envs[0], self.log_camera, gymapi.IMAGE_COLOR
    )
    rgb_image = gymtorch.wrap_tensor(rgba_image)[..., :3].float()

    self.log_image_buf.copy_(rgb_image)

  def log_render(self):
    return self.log_image_buf.clone()

  def compute_observations(self):
    self.obs_buf[:, :self.state_obs_size] = self.compute_proprioceptive_observations()
    if self.get_image:
      self.obs_buf[
          :, self.state_obs_size:self.state_obs_size + self.image_obs_size
      ] = self.image_buf[:, self.frame_skip_indexes, :].flatten(1)

    if self.use_height_map:
      self.obs_buf[
          :, self.state_obs_size + self.image_obs_size: self.state_obs_size + self.image_obs_size + self.height_map_obs_size
      ] = torch.clip(self.root_states[self.a1_indices, 2].unsqueeze(1) - self.measured_heights, -2., 2.) * self.height_scale
      # self.obs_buf[
      #     :, self.state_obs_size + self.image_obs_size: self.state_obs_size + self.image_obs_size + self.height_map_obs_size
      # ] = torch.clip(
      #     self.root_states[self.a1_indices, 2].unsqueeze(
      #         1) - self.measured_heights,
      #     self.height_clip_low,
      #     self.height_clip_high
      # ) * self.height_scale
    if self.save_historical_buffer:
      self.obs_buf[:, -self.historical_buffer_size *
                   (self.state_obs_size - self.privilege_info_len):] = self.historical_states_buffer.view(
          self.num_envs, -1).clone()

  def _process_rigid_shape_props(self, props, env_id):
    """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
        Called During environment creation.
        Base behavior: randomizes the friction of each environment
    Args:
        props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
        env_id (int): Environment id
    Returns:
        [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
    """
    if self.randomize_friction:
      if env_id == 0:
        # prepare friction randomization
        friction_range = self.cfg["env"]["learn"]["friction_range"]
        num_buckets = 64
        bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
        friction_buckets = torch_rand_float(
            friction_range[0], friction_range[1], (num_buckets, 1), device='cpu')
        self.friction_coeffs = friction_buckets[bucket_ids]

      for s in range(len(props)):
        props[s].friction = self.friction_coeffs[env_id]
    return props

  def _process_dof_props(self, props, env_id):
    """ Callback allowing to store/change/randomize the DOF properties of each environment.
        Called During environment creation.
        Base behavior: stores position, velocity and torques limits defined in the URDF
    Args:
        props (numpy.array): Properties of each DOF of the asset
        env_id (int): Environment id
    Returns:
        [numpy.array]: Modified DOF properties
    """
    if env_id == 0:
      self.dof_pos_limits = torch.zeros(
          self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
      self.dof_vel_limits = torch.zeros(
          self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
      self.torque_limits = torch.zeros(
          self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
      for i in range(len(props)):
        self.dof_pos_limits[i, 0] = props["lower"][i].item()
        self.dof_pos_limits[i, 1] = props["upper"][i].item()
        self.dof_vel_limits[i] = props["velocity"][i].item()
        self.torque_limits[i] = props["effort"][i].item()
        # soft limits
        m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
        r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
        self.dof_pos_limits[i, 0] = m - 0.5 * r * \
            self.cfg["env"]["learn"]["soft_dof_pos_limit"]
        self.dof_pos_limits[i, 1] = m + 0.5 * r * \
            self.cfg["env"]["learn"]["soft_dof_pos_limit"]
    for i in range(self.num_dof):
      props['driveMode'][i] = gymapi.DOF_MODE_POS
      props['stiffness'][i] = self.Kp
      props['damping'][i] = self.Kd
      # props['stiffness'][i] = 0.0
      # props['damping'][i] = 0.0
    return props

  def reset(self, env_ids):
    # Randomization can happen only at reset time, since it can reset actor positions on GPU
    self.randomize_buf[env_ids] += 1
    for _, randomizer in self.randomizer.items():
      randomizer.apply_randomizations(self)

    # positions_offset = torch_rand_float(
    #     0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
    velocities = torch_rand_float(
        -0.05, 0.05, (len(env_ids), self.num_dof), device=self.device
    )

    self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + 0.1 * \
        (torch.rand_like(
            self.default_dof_pos[env_ids], device=self.device) - 0.5)
    self.dof_vel[env_ids] = velocities
    self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
    self.last_dof_vel[env_ids] = self.dof_vel[env_ids]
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    a1_indices = self.a1_indices[env_ids].to(torch.int32)

    root_states = self.init_states_for_each_env.clone()

    if self.save_historical_buffer:
      self.historical_states_buffer[env_ids] = 0

    if self.terrain_params is not None:
      random_x = 2 * torch.rand(env_ids.size(), device=env_ids.device) - 1
      random_y = 2 * torch.rand(env_ids.size(), device=env_ids.device) - 1

      root_states[env_ids, 0] += random_x * \
          self.robot_random_origin_x_range[env_ids]
      root_states[env_ids, 1] += random_y * \
          self.robot_random_origin_y_range[env_ids]

    self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                 gymtorch.unwrap_tensor(
                                                     root_states),
                                                 gymtorch.unwrap_tensor(a1_indices), len(env_ids_int32))
    self.gym.set_dof_state_tensor_indexed(self.sim,
                                          gymtorch.unwrap_tensor(
                                              self.dof_state),
                                          gymtorch.unwrap_tensor(a1_indices), len(env_ids_int32))

    if self.historical_step > 1:
      self.dof_pos_buf[env_ids] = 0
      self.actions_buf[env_ids] = 0
      self.torques_buf[env_ids] = 0
    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 0
    self.feet_air_time[env_ids] = 0
    if self.get_image:
      self.image_buf[env_ids] = 0
    if self.use_stacked_state:
      self.stacked_obs_buf[env_ids] = 0
    if self.tracking_cam_viz:
      self._update_viewer()

  def update_image(self, env_ids):
    # output image and then write it to disk using Pillow
    # communicate physics to graphics system
    image_vec = []
    if self.image_type != "rgb":
      if self.camera_number == 1:
        if not self.updated:
          depth_image = self.gym.get_camera_image_gpu_tensor(
              self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_DEPTH)
          self.depth_image.append(gymtorch.wrap_tensor(depth_image))
        image_vec = self.depth_image[env_ids].unsqueeze(0)

      elif self.camera_number == 2:
        if not self.updated:
          head_depth_image = self.gym.get_camera_image_gpu_tensor(
              self.sim, self.envs[env_ids], self.camera_handles[env_ids, 0], gymapi.IMAGE_DEPTH)
          front_depth_image = self.gym.get_camera_image_gpu_tensor(
              self.sim, self.envs[env_ids], self.camera_handles[env_ids, 1], gymapi.IMAGE_DEPTH)
          self.head_depth_image.append(
              gymtorch.wrap_tensor(head_depth_image))
          self.front_depth_image.append(
              gymtorch.wrap_tensor(front_depth_image))

        image_vec = torch.stack(
            [self.head_depth_image[env_ids], self.front_depth_image[env_ids]], dim=0)

      elif self.camera_number == 3:
        if not self.updated:
          head_depth_image = self.gym.get_camera_image_gpu_tensor(
              self.sim, self.envs[env_ids], self.camera_handles[env_ids, 0], gymapi.IMAGE_DEPTH)
          front_depth_image = self.gym.get_camera_image_gpu_tensor(
              self.sim, self.envs[env_ids], self.camera_handles[env_ids, 1], gymapi.IMAGE_DEPTH)
          bottom_depth_image = self.gym.get_camera_image_gpu_tensor(
              self.sim, self.envs[env_ids], self.camera_handles[env_ids, 2], gymapi.IMAGE_DEPTH)
          self.head_depth_image.append(
              gymtorch.wrap_tensor(head_depth_image))
          self.front_depth_image.append(
              gymtorch.wrap_tensor(front_depth_image))
          self.bottom_depth_image.append(
              gymtorch.wrap_tensor(bottom_depth_image))

        image_vec = torch.stack(
            [self.head_depth_image[env_ids], self.front_depth_image[env_ids], self.bottom_depth_image[env_ids]], dim=0)

    if self.image_type != "depth":
      raise NotImplementedError
      if self.camera_number == 1:
        rgba_image = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_COLOR)
        rgb_image = gymtorch.wrap_tensor(rgba_image)[:3].float()
      elif self.camera_number == 3:
        head_rgba_image = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_ids], self.camera_handles[env_ids, 0], gymapi.IMAGE_COLOR)
        front_rgba_image = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_ids], self.camera_handles[env_ids, 1], gymapi.IMAGE_COLOR)
        bottom_rgba_image = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_ids], self.camera_handles[env_ids, 2], gymapi.IMAGE_COLOR)
        head_rgba_image = gymtorch.wrap_tensor(head_rgba_image)[
            :3].float()
        front_rgba_image = gymtorch.wrap_tensor(front_rgba_image)[
            :3].float()
        bottom_rgba_image = gymtorch.wrap_tensor(
            bottom_rgba_image)[:3].float()
        rgba_image = torch.stack(
            [head_rgba_image, front_rgba_image, bottom_rgba_image], dim=1)
      rgb_image = rgba_image / 255.
      image_vec.append(rgb_image)

    # image_vec = image_vec.flatten(1)
    if self.device == "cpu":
      image_vec = image_vec.to(self.device)

    if not self.updated and env_ids == self.num_envs - 1:
      self.updated = True
    return image_vec

  def reset_command(self, env_ids):
    # Randomization can happen only at reset time, since it can reset actor positions on GPU
    if self.command_type == "vel":
      self.commands_x[env_ids] = torch_rand_float(
          self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
      self.commands_y[env_ids] = torch_rand_float(
          self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
      self.commands_yaw[env_ids] = torch_rand_float(
          self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
    else:
      raise NotImplementedError

  def _build_legcontroller(self):
    self.leg_controller = LegController(
        cfg=self.cfg["env"]["control"], num_envs=self.num_envs, device=self.device)

  def _build_viewer(self):
    self.enable_viewer_sync = True
    self.viewer = None
    self.refEnv = self.env_viewer_cfg["refEnv"]
    # if running with a viewer, set up keyboard shortcuts and camera
    if self.headless == False:
      # subscribe to keyboard shortcuts
      self.viewer = self.gym.create_viewer(
          self.sim, gymapi.CameraProperties())
      self.gym.subscribe_viewer_keyboard_event(
          self.viewer, gymapi.KEY_ESCAPE, "QUIT")
      self.gym.subscribe_viewer_keyboard_event(
          self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
      self.gym.subscribe_viewer_keyboard_event(
          self.viewer, gymapi.KEY_D, "debug_visualize")
      self.gym.subscribe_viewer_keyboard_event(
          self.viewer, gymapi.KEY_P, "push_robots")
      self.gym.subscribe_viewer_keyboard_event(
          self.viewer, gymapi.KEY_F, "feet_visualize")
      # set the camera position based on up axis
      sim_params = self.gym.get_sim_params(self.sim)
      if sim_params.up_axis == gymapi.UP_AXIS_Z:
        cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
        cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
      else:
        cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
        cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

      p = self.env_viewer_cfg["pos"]
      lookat = self.env_viewer_cfg["lookat"]
      self.camera_distance = [
          _lookat - _p for _lookat, _p in zip(lookat, p)]
      cam_pos = gymapi.Vec3(p[0], p[1], p[2])
      cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
      self.gym.viewer_camera_look_at(
          self.viewer, None, cam_pos, cam_target)

  def _update_viewer(self):
    if self.viewer is not None:
      lookat = self.root_states[self.a1_indices[self.refEnv], 0:3]
      p = [_lookat - _camera_distance for _camera_distance,
           _lookat in zip(self.camera_distance, lookat)]
      cam_pos = gymapi.Vec3(p[0], p[1], p[2])
      cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
      self.gym.viewer_camera_look_at(
          self.viewer, None, cam_pos, cam_target)

  def _push_robots(self):
    """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
    """
    max_vel = self.max_push_vel_xy
    self.root_states[:, 7:9] = torch_rand_float(
        -max_vel, max_vel, (self.num_envs, 2), device=self.device)
    self.gym.set_actor_root_state_tensor(
        self.sim, gymtorch.unwrap_tensor(self.root_states))

  def _foot_position_in_hip_frame(self, angles, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[...,
                                             0], angles[..., 1], angles[..., 2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = torch.sqrt(
        l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * torch.sin(eff_swing)
    off_z_hip = -leg_distance * torch.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = torch.cos(theta_ab) * off_y_hip - \
        torch.sin(theta_ab) * off_z_hip
    off_z = torch.sin(theta_ab) * off_y_hip + \
        torch.cos(theta_ab) * off_z_hip
    return torch.stack([off_x, off_y, off_z], dim=-1)

  def _foot_velocity_in_hip_frame(self, angles, angles_vel, l_hip_sign=1):
    theta_ab, theta_hip, theta_knee = angles[...,
                                             0], angles[..., 1], angles[..., 2]
    d_theta_ab, d_theta_hip, d_theta_knee = angles_vel[...,
                                                       0], angles_vel[..., 1], angles_vel[..., 2]
    l_up = 0.2
    l_low = 0.2
    l_hip = 0.08505 * l_hip_sign
    leg_distance = torch.sqrt(
        l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee))
    d_leg_distance = -l_up * l_low * \
        torch.sin(theta_knee) * d_theta_knee / leg_distance

    eff_swing = theta_hip + theta_knee / 2
    d_eff_swing = d_theta_hip + d_theta_knee / 2

    off_x_hip = -leg_distance * torch.sin(eff_swing)
    off_z_hip = -leg_distance * torch.cos(eff_swing)
    off_y_hip = l_hip

    d_off_x_hip = -(d_leg_distance * torch.sin(eff_swing) +
                    leg_distance * torch.cos(eff_swing) * d_eff_swing)
    d_off_z_hip = -(d_leg_distance * torch.cos(eff_swing) -
                    leg_distance * torch.sin(eff_swing) * d_eff_swing)
    d_off_y_hip = 0

    # off_x = off_x_hip
    # off_y = torch.cos(theta_ab) * off_y_hip - \
    #   torch.sin(theta_ab) * off_z_hip
    # off_z = torch.sin(theta_ab) * off_y_hip + \
    #   torch.cos(theta_ab) * off_z_hip

    d_off_x = d_off_x_hip
    d_off_y = torch.cos(theta_ab) * d_off_y_hip - torch.sin(theta_ab) * d_theta_ab * off_z_hip - \
        (torch.sin(theta_ab) * d_off_z_hip +
         d_theta_ab * torch.cos(theta_ab) * off_y_hip)
    d_off_z = torch.sin(theta_ab) * d_off_y_hip + d_theta_ab * torch.cos(theta_ab) * off_z_hip + \
        (torch.cos(theta_ab) * d_off_z_hip -
         d_theta_ab * torch.sin(theta_ab) * off_y_hip)
    return torch.stack([d_off_x, d_off_y, d_off_z], dim=-1)

  def _footPositionsInBaseFrame(self):
    return self._footPositionsInHipFrame() + self._hip_offset.unsqueeze(
        0).repeat(self.num_envs, 1, 1)

  def _footPositionsInHipFrame(self):
    """Get the robot's foot position in the hip frame."""
    angles = self.dof_pos
    angles = angles.reshape(self.num_envs, self.num_legs, 3)
    foot_positions = torch.zeros_like(angles, device=self.device)
    for i in range(self.num_legs):
      foot_positions[:, i] = self._foot_position_in_hip_frame(
          angles[:, i], l_hip_sign=(-1)**(i))
    return foot_positions
    # a = self.root_states[self.a1_indices, 3:7].repeat(1, 4)
    # b = self._hip_offset.unsqueeze(0).repeat(self.num_envs, 1, 1)
    # print("a", a.shape)
    # print("b", b.shape)
    # return foot_positions + quat_apply_yaw(self.root_states[self.a1_indices, 3:7].repeat(1, 4),
    #                                        self._hip_offset.unsqueeze(0).repeat(self.num_envs, 1, 1))

  def _footVelocitiesInHipFrame(self):
    """Get the robot's foot velocities in the base frame (same as in hip frame)."""
    angles = self.dof_pos
    angles_vel = self.dof_vel
    angles = angles.reshape(self.num_envs, self.num_legs, 3)
    angles_vel = angles_vel.reshape(self.num_envs, self.num_legs, 3)
    foot_velocities = torch.zeros_like(angles_vel, device=self.device)
    for i in range(self.num_legs):
      foot_velocities[:, i] = self._foot_velocity_in_hip_frame(
          angles[:, i], angles_vel[:, i], l_hip_sign=(-1)**(i))
    return foot_velocities

  def _compute_leg_jacobian(self):
    l1 = HIP_LINK_LENGTH
    l2 = THIGH_LINK_LENGTH
    l3 = CALF_LINK_LENGTH

    # Only for A1
    joint_pos = self.dof_pos.reshape(
        self.num_envs, 4, 3
    )

    side_sign = torch.Tensor(
        [1, -1, 1, -1]
    ).to(self.device).reshape((1, 4))

    s1 = torch.sin(joint_pos[..., 0])
    s2 = torch.sin(joint_pos[..., 1])
    s3 = torch.sin(joint_pos[..., 2])

    c1 = torch.cos(joint_pos[..., 0])
    c2 = torch.cos(joint_pos[..., 1])
    c3 = torch.cos(joint_pos[..., 2])

    c23 = c2 * c3 - s2 * s3
    s23 = s2 * c3 + c2 * s3

    self.leg_jacobian[..., 0, 0] = 0
    self.leg_jacobian[..., 1, 0] = -side_sign * \
        l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
    self.leg_jacobian[..., 2, 0] = side_sign * \
        l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
    self.leg_jacobian[..., 0, 1] = -l3 * c23 - l2 * c2
    self.leg_jacobian[..., 1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
    self.leg_jacobian[..., 2, 1] = l2 * s2 * c1 + l3 * s23 * c1
    self.leg_jacobian[..., 0, 2] = -l3 * c23
    self.leg_jacobian[..., 1, 2] = -l3 * s23 * s1
    self.leg_jacobian[..., 2, 2] = l3 * s23 * c1
    return self.leg_jacobian

  def _footPositionsToJointAngles(self, foot_positions):
    feet_err = foot_positions - \
        self.default_feet_pos.reshape(4, 3).unsqueeze(0)
    # print("Feet Err:", feet_err)
    self._compute_leg_jacobian()
    # print("Leg Jacobian:", self.leg_jacobian)
    u = (torch.inverse(self.leg_jacobian) @
         feet_err.unsqueeze_(-1)).squeeze(-1)
    # print("U:", u)
    pos_target = u.reshape(self.num_envs, 4, 3) + \
        self.default_dof_pos.reshape(-1, 4, 3)
    # print(pos_target)
    return pos_target

  def _getContactFootState(self):
    contact_forces = torch.norm(
        self.contact_forces[:, self.feet_indices], dim=-1)
    contact_states = (contact_forces > 0.5)
    return contact_states

  def _massMatrix(self):
    # TODO: implement
    mass_matrix = compute_mass_matrix(
        self.num_envs,
        108 / 9.81,
        torch.as_tensor([0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00],
                        device=self.device).view((3, 3)),
        self._footPositionsInBaseFrame(), self.device)

    return mass_matrix

  # ============ Reward Functions ==============

  def _prepare_reward_function(self):
    """ Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
    """
    self.reward_functions = []
    self.reward_names = []

    # reward scales
    self.reward_scales = self.task_wrapper.task_cfg["learn"]["reward_scales"]
    self.reward_params = self.task_wrapper.task_cfg["learn"]["reward_params"]
    self.num_rew_terms = len(self.reward_scales)

    print(self.reward_scales)

    for key in self.reward_scales.keys():
      self.reward_scales[key] *= self.dt

    for name in self.reward_scales.keys():
      if name == "termination":
        continue
      if name == "gail":
        continue
      self.reward_names.append(name)
      name = '_reward_' + name
      self.reward_functions.append(getattr(self, name))

    # if "gail" in self.reward_scales:
    #   self.gail_rew_buf = torch.zeros(
    #       self.num_envs, device=self.device, dtype=torch.float)

    self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                         for name in self.reward_scales.keys()}
    if self.task_wrapper.task_name == "following_command":
      # command ranges
      self.command_x_range = self.task_wrapper.task_cfg["randomCommandRanges"]["linear_x"]
      self.command_y_range = self.task_wrapper.task_cfg["randomCommandRanges"]["linear_y"]
      self.command_yaw_range = self.task_wrapper.task_cfg["randomCommandRanges"]["yaw"]

  def compute_reward(self):
    """ Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
    """
    self.rew_buf[:] = 0.
    # print("----")
    for i in range(len(self.reward_functions)):
      name = self.reward_names[i]
      rew = self.reward_functions[i]() * self.reward_scales[name]
      # if name == "gail":
      #   self.gail_rew_buf.copy_(rew)
      # if name == "torques" or name == "moving_forward":
      #   print(name, rew)
      self.rew_buf += rew
      self.episode_sums[name] += rew
      # print("reward:", name, rew)
    # self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
    # add termination reward after clipping
    if "termination" in self.reward_scales:
      rew = self._reward_termination(
      ) * self.reward_scales["termination"]
      self.rew_buf += rew
      self.episode_sums["termination"] += rew

  def _reward_lin_vel_z(self):
    # Penalize z axis base linear velocity
    return torch.square(self.base_lin_vel[:, 2])

  def _reward_ang_vel_xy(self):
    # Penalize xy axes base angular velocity
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

  def _reward_orientation(self):
    # Penalize non flat base orientation
    # projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
    # print(projected_gravity[:, :2])
    # print(torch.sum(torch.abs(projected_gravity[:, :2]), dim=1))
    return torch.sum(torch.abs(self.projected_gravity[:, :2]), dim=1)

  def _reward_base_height(self):
    # Penalize base height away from target
    if self.use_height_map:
      # if self.use_wide_height_map:
      #   print(self.measured_heights.shape)
      #   filtered_height = self.measured_heights.reshape(
      #       -1, 19, 18)[:, 8:13, 7:11].reshape(-1, 20)
      # else:
      # print(self.measured_heights.shape)
      filtered_height = self.measured_heights[:, :546].reshape(
          -1, 21, 26)[:, 8:13, 8:13].reshape(-1, 25)
      diff = self.root_states[:, 2].unsqueeze(1) - filtered_height
      base_height = torch.mean(
          #
          diff * (diff <= 0.37) +
          (self.root_states[:, 2].unsqueeze(1)) * (diff > 0.37), dim=1
      )  # / (torch.sum((filtered_height > -2), dim=1) + 1e-5)
      # print(base_height)
    else:
      base_height = self.root_states[:, 2]
    # print(base_height)
    return torch.abs(base_height - self.reward_params["base_height_target"])

  def _reward_torques(self):
    # Penalize torques
    return torch.sum(torch.abs(self.computed_torques * self.dof_vel), dim=1)

  def _reward_dof_vel(self):
    # Penalize dof velocities
    return torch.sum(torch.square(self.dof_vel), dim=1)

  def _reward_dof_acc(self):
    # Penalize dof accelerations
    return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

  def _reward_action_rate(self):
    # Penalize changes in actions
    return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

  def _reward_collision(self):
    # Penalize collisions on selected bodies
    return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

  def _reward_termination(self):
    # Terminal reward / penalty
    return self.reset_buf * ~self.time_out_buf

  def _reward_dof_pos_limits(self):
    # Penalize dof positions too close to the limit
    out_of_limits = - \
        (self.dof_pos -
         self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
    out_of_limits += (self.dof_pos -
                      self.dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

  def _reward_dof_vel_limits(self):
    # Penalize dof velocities too close to the limit
    return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["env"]["learn"]["soft_dof_vel_limit"]).clip(min=0.), dim=1)

  def _reward_torque_limits(self):
    # penalize torques too close to the limit
    return torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg["env"]["learn"]["soft_torque_limit"]).clip(min=0.), dim=1)

  def _reward_moving_forward(self):
    # encourage moving forward as fast as possible
    base_lin_val_in_world_frame = self.root_states[self.a1_indices, 7].clone(
    )
    # print(base_lin_val_in_world_frame)
    return torch.clamp(
        base_lin_val_in_world_frame,
        min=self.reward_params["forward_vel_clip"],
        max=self.reward_params["forward_vel"]
    ) / self.reward_params["forward_vel"]
    # Tracking of linear velocity commands (xy axes)
    # lin_vel_error = torch.square(
    #     self.reward_params["forward_vel"] - base_lin_val_in_world_frame
    # )
    # # print(base_lin_val_in_world_frame,
    # #       torch.exp(-lin_vel_error / self.reward_params["tracking_sigma"]))
    # return torch.exp(-lin_vel_error / self.reward_params["tracking_sigma"])

  # def _reward_moving_forward(self):
  #   # encourage moving forward as fast as possible
  #   base_lin_val_in_world_frame = self.root_states[self.a1_indices, 7].clone(
  #   )
  #   moving_forward_err = torch.square(
  #     base_lin_val_in_world_frame - self.reward_params["forward_vel"])
  #   return torch.exp(-moving_forward_err / self.reward_params["tracking_sigma"])

  def _reward_tracking_lin_vel(self):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.sum(torch.square(
        self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / self.reward_params["tracking_sigma"])

  def _reward_tracking_ang_vel(self):
    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.square(
        self.commands[:, 2] - self.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / self.reward_params["tracking_sigma"])

  def _reward_feet_air_time(self):
    # Reward long steps
    # contact = self.contact_forces[:, self.feet_indices + 1, 2] > 0.5
    contact = self.contact_forces[:, self.feet_indices, 2] > 0.5
    # print(contact)
    first_contact = (self.feet_air_time > 0.) * contact
    # print(first_contact)
    self.feet_air_time += self.dt
    # reward only on first contact with the ground
    rew_airTime = torch.sum(
        (self.feet_air_time - 0.5) * first_contact, dim=1
    )
    # no reward for zero command
    # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.5
    self.feet_air_time *= ~contact
    # print(rew_airTime)
    return rew_airTime

  def _reward_stumble(self):
    # Penalize feet hitting vertical surfaces
    return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
                     5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

  def _reward_stand_still(self):
    # Penalize motion at zero commands
    return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

  def _reward_feet_contact_force(self):
    # penalize high contact force
    return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.reward_params["max_contact_force"]).clip(min=0.), dim=1)

  def _reward_gail(self):
    # GAIL reward
    # * self.reward_scales["gail"]
    return self.learners["gail"].reward(self.last_dof_pos, self.dof_pos)

  def compute_proprioceptive_observations(self):

    # commands_scaled = self.commands[..., :3] * \
    #     torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
    #                  requires_grad=False, device=self.commands.device)

    if self.historical_step > 1:
      dof_pos = (self.dof_pos_buf -
                 self.default_dof_pos.unsqueeze(1)).view(self.num_envs, -1)
      actions = self.actions_buf.view(
          self.num_envs, -1) * self.action_scale.repeat(self.historical_step)
    else:
      dof_pos = self.dof_pos
      actions = self.actions_buf * self.action_scale

    _obs = [
        # torch.zeros_like(self.base_lin_vel * self.lin_vel_scale),
        # self.base_ang_vel * self.ang_vel_scale,
        dof_pos * self.dof_pos_scale,
        self.dof_vel * self.dof_vel_scale,
        self.projected_gravity,
        # commands_scaled,
        # actions,
    ]
    if self.use_prev_action:
      _obs.append(actions)
    # _obs = [
    #     # torch.zeros_like(self.base_lin_vel * self.lin_vel_scale),
    #     self.base_lin_vel * self.lin_vel_scale,
    #     self.base_ang_vel * self.ang_vel_scale,
    #     dof_pos * self.dof_pos_scale,
    #     self.dof_vel * self.dof_vel_scale,
    #     projected_gravity,
    #     commands_scaled,
    #     actions,
    # ]
    # print("--------------------------------------------")
    # print("base lin", self.base_lin_vel * self.lin_vel_scale)
    # print("base ang vel", self.base_ang_vel * self.ang_vel_scale)
    # print("dof pos", dof_pos * self.dof_pos_scale)
    # print("dof vel", self.dof_vel * self.dof_vel_scale)
    # print("projected g", projected_gravity)
    # print("commands_scaled", commands_scaled)
    # print("actions", actions)
    # print("actions raw", self.actions_buf.view(
    #       self.num_envs, -1))
    # print("action scale", self.action_scale.repeat(self.historical_step))

    # if self.leg_controller.mode in ["cpg", "pd_foot"]:
    #   if self.use_aug_time_phase:
    #     _obs.append(
    #         self.leg_controller.phase_info.view(self.num_envs, -1)
    #     )
    #   # print("Phase Info", self.leg_controller.phase_info.view(self.num_envs, -1))
    #   if self.use_original_time_phase:
    #     _obs.append(
    #         self.leg_controller.time_phase_info.view(self.num_envs, -1)
    #     )
    #     # print("Time Phase Info",
    #     #       self.leg_controller.time_phase_info.view(self.num_envs, -1))

    if self.save_historical_buffer:
      nonfilled_indices = torch.nonzero(
          (self.progress_buf < self.historical_buffer_size), as_tuple=True)[0]
      filled_indices = torch.nonzero(
          (self.progress_buf >= self.historical_buffer_size), as_tuple=True)[0]
      state_in_step = torch.cat(_obs, dim=-1)
      if nonfilled_indices.shape[0] > 0:
        self.historical_states_buffer[nonfilled_indices,
                                      self.progress_buf[nonfilled_indices]] = state_in_step[nonfilled_indices]
      if filled_indices.shape[0] > 0:
        self.historical_states_buffer[filled_indices] = torch.cat(
            [self.historical_states_buffer[filled_indices, 1:], state_in_step[filled_indices].unsqueeze(1)], dim=1)

    # ================================================================
    # Priviledged observations as follows:
    if self.use_lin_vel:
      _obs.append(
          self.base_lin_vel * self.lin_vel_scale
      )
    if self.use_ang_vel:
      _obs.append(
          self.base_ang_vel * self.ang_vel_scale
      )
    if self.use_feet_observation:
      # dim 12
      # Use absolute position to compute with height points
      feet_points = self.feet_pos_buf.clone().view(self.num_envs, 4, 3)
      for j in range(feet_points.shape[1]):
        feet_points[:, j] = quat_rotate(self.base_quat, feet_points[:, j])
      _obs.append(feet_points.view(self.num_envs, -1))
    if self.use_contact_force:
      # dim 12
      _obs.append(
          # self.contact_forces[:, self.feet_indices + 1].view(self.num_envs, -1) * self.contact_forces_scale)
          self.contact_forces[:, self.feet_indices].view(self.num_envs, -1) * self.contact_forces_scale)
    if self.use_contact_state:
      # dim 4
      _obs.append(self._getContactFootState().view(
          self.num_envs, -1) * self.contact_state_scale)
    if self.use_feet_air_time:
      # dim 4
      _obs.append(self.feet_air_time.view(
          self.num_envs, -1) * self.feet_air_time_scale)
    # for _ob in _obs:
    #   print(_ob.shape)

    obs = torch.cat(_obs, dim=-1)
    if self.use_stacked_state:
      # print(obs.unsqueeze(-2).shape)
      # print(self.stacked_obs_buf[:, :-1].shape)
      self.stacked_obs_buf = torch.cat([
          obs.unsqueeze(-2), self.stacked_obs_buf[:, :-1]
      ], dim=1)
      obs = self.stacked_obs_buf[:, self.state_skip_indexes, :].flatten(1)
    return obs

  def log_robot_cameras(self, env_id=0):

    log_images = []
    if self.camera_number == 2:
      self.head_depth_image[env_id][torch.isneginf(
          self.head_depth_image[env_id])] = -5
      head_depth = -255.0 * \
          (self.head_depth_image[env_id] /
           torch.max(self.head_depth_image[env_id] + 1e-4))
      log_images.append(head_depth)
      self.front_depth_image[env_id][torch.isneginf(
          self.front_depth_image[env_id])] = -5
      front_depth = -255.0 * \
          (self.front_depth_image[env_id] /
           torch.max(self.front_depth_image[env_id] + 1e-4))
      log_images.append(front_depth)
    elif self.camera_number == 1:
      self.depth_image[env_id][torch.isneginf(
          self.depth_image[env_id])] = -5
      # depth = -255.0 * \
      #     (self.depth_image[env_id] /
      #      torch.max(self.depth_image[env_id] + 1e-4))
      depth = 255.0 * \
          (self.depth_image[env_id] / -5)
      depth = torch.clamp(depth, 0, 255)
      # print(depth)
      if self.reshape_vision:
        print(depth.shape)
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(
                0)[:, :, self.horizontal_clip:-self.horizontal_clip],
            size=self.target_shape
        ).squeeze(0).squeeze(0)
      log_images.append(depth)
    log_images = torch.cat(log_images, dim=0)
    return log_images


@torch.jit.script
def depth_image_processes(
    image_vectors: Tensor, image_buf: Tensor, camera_number: int,  # , nolinear: bool
    mean: float = 1.25, std: float = 0.45, input_inverse_depth=False, input_original_depth=False
):
  image_vectors[torch.isneginf(image_vectors)] = -100.0
  # image_vectors[torch.isinf(image_vectors)] = -100.0
  image_vectors[image_vectors < -5] = -5
  # image_vectors[image_vectors > -1e-3] = -5
  # if nolinear:
  image_vectors = image_vectors * -1

  image_vectors = torch.clamp(image_vectors, 0.1, 3.0)
  if input_inverse_depth:
    image_vectors = 1 / image_vectors
  elif input_original_depth:
    image_vectors = image_vectors
  else:
    image_vectors = torch.sqrt(torch.log(image_vectors + 1))
  return torch.cat([
      image_vectors, image_buf[:, :-camera_number]
  ], dim=1)


@torch.jit.script
def concat_3cam_images(head_depth_image: List[Tensor], front_depth_image: List[Tensor], bottom_depth_image: List[Tensor]):
  head_depth_image = torch.stack(
      head_depth_image, dim=0)
  front_depth_image = torch.stack(
      front_depth_image, dim=0)
  bottom_depth_image = torch.stack(
      bottom_depth_image, dim=0)
  return torch.stack(
      [head_depth_image, front_depth_image, bottom_depth_image], dim=1).flatten(-2)


@torch.jit.script
def concat_2cam_images(head_depth_image: List[Tensor], front_depth_image: List[Tensor]):
  head_depth_image = torch.stack(
      head_depth_image, dim=0)
  front_depth_image = torch.stack(
      front_depth_image, dim=0)
  return torch.stack(
      [head_depth_image, front_depth_image], dim=1).flatten(-2)


@torch.jit.script
def quat_apply_yaw(quat, vec):
  quat_yaw = quat.clone().view(-1, 4)
  quat_yaw[:, :2] = 0.
  quat_yaw = normalize(quat_yaw)
  return quat_apply(quat_yaw, vec)
