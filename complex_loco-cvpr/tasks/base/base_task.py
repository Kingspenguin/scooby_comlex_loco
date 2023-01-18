# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
from isaacgym import gymapi
import torch


# Base class for RL tasks
class BaseTask():

  def __init__(self, cfg, enable_camera_sensors=False):
    self.gym = gymapi.acquire_gym()

    self.device_type = cfg.get("device_type", "cuda")
    self.device_id = cfg.get("device_id", 0)

    self.device = "cpu"
    if self.device_type == "cuda" or self.device_type == "GPU":
      self.device = "cuda" + ":" + str(self.device_id)

    self.headless = cfg["headless"]

    # double check!
    self.graphics_device_id = self.device_id
    if enable_camera_sensors == False and self.headless == True and not self.get_image:
      self.graphics_device_id = -1

    self.num_envs = cfg["env"]["numEnvs"]
    self.num_obs = cfg["env"]["numObservations"]
    self.num_states = cfg["env"].get("numStates", 0)
    self.num_actions = cfg["env"]["numActions"]

    # self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

    # optimization flags for pytorch JIT
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)

    # allocate buffers
    self.obs_buf = torch.zeros(
      (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
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
    self.extras = {}

    self.original_props = {}
    self.dr_randomizations = {}
    self.first_randomization = True
    self.actor_params_generator = None
    self.extern_actor_params = {}
    for env_id in range(self.num_envs):
      self.extern_actor_params[env_id] = None

    self.last_step = -1
    self.last_rand_step = -1
    self.last_rew_rand_step = -1

    # create envs, sim and viewer
    self.create_sim()
    self.gym.prepare_sim(self.sim)

    # todo: read from config
    self.enable_viewer_sync = True
    self.viewer = None

    # if running with a viewer, set up keyboard shortcuts and camera
    if self.headless == False:
      # subscribe to keyboard shortcuts
      self.viewer = self.gym.create_viewer(
        self.sim, gymapi.CameraProperties())
      self.gym.subscribe_viewer_keyboard_event(
        self.viewer, gymapi.KEY_ESCAPE, "QUIT")
      self.gym.subscribe_viewer_keyboard_event(
        self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

      # set the camera position based on up axis
      sim_params = self.gym.get_sim_params(self.sim)
      if sim_params.up_axis == gymapi.UP_AXIS_Z:
        cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
        cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
      else:
        cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
        cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

      self.gym.viewer_camera_look_at(
        self.viewer, None, cam_pos, cam_target)

  # set gravity based on up axis and return axis index
  def set_sim_params_up_axis(self, sim_params, axis):
    if axis == 'z':
      sim_params.up_axis = gymapi.UP_AXIS_Z
      sim_params.gravity.x = 0
      sim_params.gravity.y = 0
      sim_params.gravity.z = -9.81
      return 2
    return 1

  def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
    sim = self.gym.create_sim(
      compute_device, graphics_device, physics_engine, sim_params)
    if sim is None:
      print("*** Failed to create sim")
      quit()

    return sim

  def step(self, actions):
    if self.dr_randomizations.get('actions', None):
      actions = self.dr_randomizations['actions']['noise_lambda'](
        actions)

    # apply actions
    self.pre_physics_step(actions)

    # step physics and render each frame
    for _ in range(self.control_freq_inv):
      self.render()
      self.gym.simulate(self.sim)

    # to fix!
    if self.device == 'cpu':
      self.gym.fetch_results(self.sim, True)

    # compute observations, rewards, resets, ...
    self.post_physics_step()

    if self.dr_randomizations.get('observations', None):
      self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](
        self.obs_buf)

  def get_states(self):
    return self.states_buf

  def render(self, sync_frame_time=True):
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

      # fetch results
      if self.device != 'cpu':
        self.gym.fetch_results(self.sim, True)

      # step graphics
      if self.enable_viewer_sync:
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        if sync_frame_time:
          self.gym.sync_frame_time(self.sim)
      else:
        self.gym.poll_viewer_events(self.viewer)

  def pre_physics_step(self, actions):
    raise NotImplementedError

  def post_physics_step(self):
    raise NotImplementedError
