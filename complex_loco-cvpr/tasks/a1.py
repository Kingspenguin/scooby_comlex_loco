# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
from utils.torch_jit_utils import *
from tasks.base.base_task import BaseTask
from utils.terrain import Terrain
from isaacgym import gymtorch
from isaacgym import gymapi
from pytorch3d.transforms import matrix_to_euler_angles, quaternion_to_matrix
import torch
from torch import Tensor
from typing import Tuple, Dict


class A1(BaseTask):

  def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

    self.cfg = cfg
    self.sim_params = sim_params
    self.physics_engine = physics_engine

    # use RL or mpc controller
    self.use_controller = self.cfg["env"]["controller"]

    # normalization
    self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
    self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
    self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
    self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
    self.action_scale = self.cfg["env"]["control"]["actionScale"]
    self.hip_action_scale = self.cfg["env"]["control"]["hipActionScale"]
    self.get_image = self.cfg["env"]["vision"]["get_image"]
    if self.get_image:
      # vision-guided mode
      self.frame_stack = self.cfg["env"]["vision"]["frame_stack"]
      self.vision_update_freq = self.cfg["env"]["vision"]["update_freq"]
      self.image_type = self.cfg["env"]["vision"]["image_type"]
      self.width = self.cfg["env"]["vision"]["width"]
      self.height = self.cfg["env"]["vision"]["height"]
      self.camera_angle = self.cfg["env"]["vision"]["camera_angle"]

    self.frame_count = 0
    # reward scales
    self.rew_scales = {}
    self.rew_scales["linearVelocityXYRewardScale"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
    self.rew_scales["angularVelocityZRewardScale"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
    self.rew_scales["linearVelocityZRewardScale"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
    self.rew_scales["torqueRewardScale"] = self.cfg["env"]["learn"]["torqueRewardScale"]
    self.rew_scales["actionSmoothingRewardScale"] = self.cfg["env"]["learn"]["actionSmoothingRewardScale"]
    self.rew_scales["dof_vel_limit"] = self.cfg["env"]["learn"]["dof_vel_limit"]
    self.rew_scales["torque_limit"] = self.cfg["env"]["learn"]["torque_limit"]
    self.rew_scales["stumble"] = self.cfg["env"]["learn"]["stumble"]
    self.rew_scales["feet_air_time"] = self.cfg["env"]["learn"]["feet_air_time"]
    self.rew_scales["collision"] = self.cfg["env"]["learn"]["collision"]

    self.soft_limits = self.cfg["env"]["learn"]["soft_limits"]

    # use diagonal action
    self.diagonal_act = self.cfg["env"]["learn"]["diagonal_act"]

    # randomization
    self.randomization_params = self.cfg["task"]["randomization_params"]
    self.randomize = self.cfg["task"]["randomize"]
    self.randomize_reward = self.cfg["randomize_reward"]["randomize"]
    self.reward_randomization_params = self.cfg["randomize_reward"]["randomization_params"]

    # commands
    self.command_type = self.cfg["env"]["command"]
    self.command_change_step = self.cfg["env"]["commandChangeStep"]

    # command ranges
    self.command_x_range = self.cfg["env"]["randomCommandRanges"]["linear_x"]
    self.command_y_range = self.cfg["env"]["randomCommandRanges"]["linear_y"]
    self.command_yaw_range = self.cfg["env"]["randomCommandRanges"]["yaw"]

    # terrain
    self.terrain_name = self.cfg["env"]["terrain"]["terrain_name"]

    if self.terrain_name == "play_ground":
      self.play_ground = Terrain(
        self.cfg["env"]["terrain"]["play_ground_attr"], self.cfg["env"]["numEnvs"])

    # plane params
    self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
    self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
    self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

    # base init state
    pos = self.cfg["env"]["baseInitState"]["pos"]
    rot = self.cfg["env"]["baseInitState"]["rot"]
    v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
    v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
    pos[0] += self.cfg["env"]["terrain"]["robot_origin"][0]
    pos[1] += self.cfg["env"]["terrain"]["robot_origin"][1]
    pos[2] += self.cfg["env"]["terrain"]["robot_origin"][2]
    state = pos + rot + v_lin + v_ang
    self.base_init_state = state

    # sensor settings
    self.historical_step = self.cfg["env"]["sensor"]["historical_step"]
    self.use_sys_information = self.cfg["env"]["sensor"]["sys_id"]

    self.refEnv = self.cfg["env"]["viewer"]["refEnv"]

    self.risk_reward = self.cfg["env"]["risk_reward"]
    self.rush_reward = self.cfg["env"]["rush_reward"]
    self.vel_reward_exp_coeff = self.cfg["env"]["vel_reward_exp_coeff"]

    self.compute_a1_reward = compute_a1_reward

    # default joint positions
    self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

    # other
    self.control_freq_inv = self.cfg["env"]["control"]["controlFrequencyInv"]
    self.dt = sim_params.dt
    self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
    self.max_episode_length = int(
      self.max_episode_length_s / (self.control_freq_inv * self.dt) + 0.5)
    self.Kp = self.cfg["env"]["control"]["stiffness"]
    self.Kd = self.cfg["env"]["control"]["damping"]

    for key in self.rew_scales.keys():
      self.rew_scales[key] *= self.dt

    extra_info_len = 3 if self.use_sys_information else 0
    if self.diagonal_act:
      self.cfg["env"]["numObservations"] = 18 * \
        self.historical_step + 24 + extra_info_len + 12
      self.cfg["env"]["numActions"] = 6
    else:
      self.cfg["env"]["numObservations"] = 24 * \
        (self.historical_step + 1) + extra_info_len + 12
      self.cfg["env"]["numActions"] = 12
    self.state_obs_size = self.cfg["env"]["numObservations"]
    if self.use_controller:
      self.cfg["env"]["numActions"] *= 2
    if self.get_image:
      if self.image_type == "depth":
        image_obs_size = self.width * \
          self.height * self.frame_stack
      elif self.image_type == "rgb":
        image_obs_size = self.width * \
          self.height * self.frame_stack * 3
      elif self.image_type == "rgbd":
        image_obs_size = self.width * \
          self.height * self.frame_stack * 4
      else:
        raise NotImplementedError

      self.cfg["env"]["numObservations"] += image_obs_size
      self.image_obs_size = image_obs_size

    self.cfg["device_type"] = device_type
    self.cfg["device_id"] = device_id
    self.cfg["headless"] = headless

    super().__init__(cfg=self.cfg)

    if self.viewer is not None:
      p = self.cfg["env"]["viewer"]["pos"]
      lookat = self.cfg["env"]["viewer"]["lookat"]
      self.camera_distance = [
        _lookat - _p for _lookat, _p in zip(lookat, p)]
      cam_pos = gymapi.Vec3(p[0], p[1], p[2])
      cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
      self.gym.viewer_camera_look_at(
        self.viewer, None, cam_pos, cam_target)

    # get gym state tensors
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
      if self.diagonal_act:
        self.actions_buf = torch.zeros(
          (self.num_envs, self.historical_step, self.num_dof // 2), device=self.device)
        self.last_actions = torch.zeros(
          self.num_envs, self.num_dof // 2, dtype=torch.float, device=self.device, requires_grad=False)
      else:
        self.actions_buf = torch.zeros(
          (self.num_envs, self.historical_step, self.num_dof), device=self.device)
        self.last_actions = torch.zeros(

          self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

      self.torques_buf = torch.zeros(
        (self.num_envs, self.historical_step, self.num_dof), device=self.device)

    if self.get_image:
      if self.frame_stack > 1:
        if self.image_type == "depth":
          self.image_buf = torch.zeros(
            (self.num_envs, self.frame_stack, self.width * self.height), device=self.device
          )
        elif self.image_type == "rgb":
          self.image_buf = torch.zeros(
            (self.num_envs, self.frame_stack, self.width * self.height * 3), device=self.device
          )
        elif self.image_type == "rgbd":
          self.image_buf = torch.zeros(
            (self.num_envs, self.frame_stack, self.width * self.height * 4), device=self.device
          )
        else:
          raise NotImplementedError

    self.commands = torch.zeros(
      self.num_envs, 3 + extra_info_len, dtype=torch.float, device=self.device, requires_grad=False)
    self.commands_y = self.commands.view(
      self.num_envs, 3 + extra_info_len)[..., 1]
    self.commands_x = self.commands.view(
      self.num_envs, 3 + extra_info_len)[..., 0]
    self.commands_yaw = self.commands.view(
      self.num_envs, 3 + extra_info_len)[..., 2]

    self.default_dof_pos = torch.zeros_like(
      self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
    for i in range(self.num_dof):
      name = self.dof_names[i]
      angle = self.named_default_joint_angles[name]
      self.default_dof_pos[:, i] = angle

    self.num_legs = 4
    self._com_offset = - \
      torch.as_tensor([0.012731, 0.002186, 0.000515],
                      device=self.device)
    self._hip_offset = torch.as_tensor([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                                        ], device=self.device) + self._com_offset
    self._default_hip_positions = torch.as_tensor([
      [0.17, -0.14, 0],
      [0.17, 0.14, 0],
      [-0.17, -0.14, 0],
      [-0.17, 0.14, 0],
    ], device=self.device)
    # initialize some data used later on
    self.extras = {}
    self.initial_root_states = self.root_states.clone()
    self.initial_root_states[self.a1_indices] = to_torch(
      self.base_init_state, device=self.device, requires_grad=False)
    self.gravity_vec = to_torch(
      get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
    self.actions = torch.zeros(self.num_envs, self.num_actions,
                               dtype=torch.float, device=self.device, requires_grad=False)
    self.time_out_buf = torch.zeros_like(self.reset_buf)

    jt = self.gym.acquire_jacobian_tensor(self.sim, "a1")
    self.jacobian_tensor = gymtorch.wrap_tensor(jt)
    self.feet_dof_pos = self.dof_pos[..., self.feet_indices]
    if self.diagonal_act:
      self.action_scale = torch.as_tensor(
        [self.hip_action_scale, self.action_scale, self.action_scale] * 2, device=self.device)
    else:
      self.action_scale = torch.as_tensor(
        [self.hip_action_scale, self.action_scale, self.action_scale] * 4, device=self.device)
    self.reset(torch.arange(self.num_envs, device=self.device))

  def create_sim(self):
    self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
    self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                  self.physics_engine, self.sim_params)
    if self.terrain_name == "play_ground":
      self._create_play_ground()
    else:
      self._create_ground_plane()
    self._create_envs(self.cfg["env"]['envSpacing'],
                      int(np.sqrt(self.num_envs)))

  def _create_ground_plane(self):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = self.plane_static_friction
    plane_params.dynamic_friction = self.plane_dynamic_friction
    self.gym.add_ground(self.sim, plane_params)

  def _create_play_ground(self):
    """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
    """
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = self.play_ground.vertices.shape[0]
    tm_params.nb_triangles = self.play_ground.triangles.shape[0]
    tm_params.transform.p.x = -self.play_ground.border_size
    tm_params.transform.p.y = -self.play_ground.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = self.cfg["env"]["terrain"]["play_ground_attr"]["static_friction"]
    tm_params.dynamic_friction = self.cfg["env"]["terrain"]["play_ground_attr"]["dynamic_friction"]
    tm_params.restitution = self.cfg["env"]["terrain"]["play_ground_attr"]["restitution"]
    # tm_params.texture_path = os.path.join(self.cwd, 'resources', 'terrain', 'textures', 'texture_concrete_seemless.jpg') TODO: not working at the moment

    self.gym.add_triangle_mesh(self.sim, self.play_ground.vertices.flatten(
      order='C'), self.play_ground.triangles.flatten(order='C'), tm_params)
    self.height_samples = torch.tensor(self.play_ground.heightsamples).view(
      self.play_ground.tot_cols, self.play_ground.tot_rows).to(self.device)

  def _create_terrain(self, env_ptr, env_id):

    pose = gymapi.Transform()

    pose.p.x = self.cfg["env"]["terrain"]["terrain_origin"][0]
    pose.p.y = self.cfg["env"]["terrain"]["terrain_origin"][1]
    pose.p.z = self.cfg["env"]["terrain"]["terrain_origin"][2]

    handle = self.gym.create_actor(
      env_ptr, self.terrain_asset, pose, "tm", env_id, 2, 0)
    if self.cfg["env"]["terrain"]["texture"] != "none":
      th = self.gym.create_texture_from_file(
        self.sim, "../assets/textures/{}".format(self.cfg["env"]["terrain"]["texture"]))
      self.gym.set_rigid_body_texture(
        env_ptr, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, th)
    return handle

  def _load_terrain(self):
    if self.terrain_name not in ["plane", "play_ground"]:
      terrain_asset_root = "../assets"
      terrain_asset_file = "terrains/{}/{}.urdf".format(
        self.terrain_name, self.terrain_name)
      terrain_asset_path = os.path.join(
        terrain_asset_root, terrain_asset_file)
      terrain_asset_root = os.path.dirname(terrain_asset_path)
      terrain_asset_file = os.path.basename(terrain_asset_path)

      terrain_asset_options = gymapi.AssetOptions()
      terrain_asset_options.fix_base_link = True
      terrain_asset_options.vhacd_enabled = True
      terrain_asset_options.vhacd_params.resolution = 3000000
      terrain_asset_options.vhacd_params.max_convex_hulls = 20
      terrain_asset_options.vhacd_params.max_num_vertices_per_ch = 256

      self.terrain_asset = self.gym.load_asset(
        self.sim, terrain_asset_root, terrain_asset_file, terrain_asset_options)

  def _create_envs(self, spacing, num_per_row):
    asset_root = "../assets"
    asset_file = "urdf/a1/a1.urdf"
    asset_path = os.path.join(asset_root, asset_file)
    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    asset_options.collapse_fixed_joints = True
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
    start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

    body_names = self.gym.get_asset_rigid_body_names(a1_asset)
    self.dof_names = self.gym.get_asset_dof_names(a1_asset)
    feet_names = [(i, s) for i, s in enumerate(body_names) if "lower" in s]
    self.feet_indices_in_bodies = torch.as_tensor(
      [fn[0] for fn in feet_names], device=self.device)
    self.feet_indices = torch.zeros(
      len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
    knee_names = [s for s in body_names if "upper" in s]
    self.knee_indices = torch.zeros(
      len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
    self.base_index = 0
    self._load_terrain()
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    self.a1_indices = []
    self.a1_handles = []
    self.terrain_handles = []
    self.terrain_indices = []
    self.box_handles = []
    self.envs = []
    if self.get_image:
      self.camera_handles = []
      self.depth_image = []

    for i in range(self.num_envs):
      # create env instances
      env_ptr = self.gym.create_env(
        self.sim, env_lower, env_upper, num_per_row)
      a1_handle = self.gym.create_actor(
        env_ptr, a1_asset, start_pose, "a1", i, 1, 0)
      dof_props = self._process_dof_props(dof_props_asset, i)
      self.gym.set_actor_dof_properties(
        env_ptr, a1_handle, dof_props)
      self.gym.enable_actor_dof_force_sensors(env_ptr, a1_handle)
      self.envs.append(env_ptr)
      self.a1_handles.append(a1_handle)
      a1_idx = self.gym.get_actor_index(
        env_ptr, a1_handle, gymapi.DOMAIN_SIM)
      self.a1_indices.append(a1_idx)
      if self.terrain_name not in ["plane", "play_ground"]:
        terrain_handle = self._create_terrain(env_ptr, i)
        terrain_idx = self.gym.get_actor_index(
          env_ptr, terrain_handle, gymapi.DOMAIN_SIM)
        self.terrain_indices.append(terrain_idx)

      if self.get_image:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.width
        camera_properties.height = self.height
        camera_properties.enable_tensors = True
        head_camera = self.gym.create_camera_sensor(
          env_ptr, camera_properties)
        camera_offset = gymapi.Vec3(0.3, 0, 0)
        camera_rotation = gymapi.Quat.from_axis_angle(
          gymapi.Vec3(0, 1, 0), np.deg2rad(self.camera_angle))
        body_handle = self.gym.get_actor_rigid_body_handle(
          env_ptr, a1_handle, 0)
        self.gym.attach_camera_to_body(head_camera, env_ptr, body_handle, gymapi.Transform(
          camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
        self.camera_handles.append(head_camera)

    for i in range(len(feet_names)):
      self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
        self.envs[0], self.a1_handles[0], feet_names[i][1]) - 1
    for i in range(len(knee_names)):
      self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(
        self.envs[0], self.a1_handles[0], knee_names[i]) - 1
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
    if self.terrain_name != "plane":
      self.terrain_indices = to_torch(
        self.terrain_indices, dtype=torch.long, device=self.device)

  def pre_physics_step(self, actions):
    self.actions = actions.clone().to(self.device)
    self.last_root_states = self.root_states.clone()
    self.last_actions[:] = self.actions[:]
    if self.historical_step > 1:
      self.actions_buf = torch.cat(
        [self.actions_buf[:, 1:], self.actions.unsqueeze(1)], dim=1)

  def controller_step(self, actions):
    self.actions = actions.clone().to(self.device)
    position_control, torque_control = torch.chunk(self.actions, 2, dim=-1)
    position_control += self.default_dof_pos
    torque_control *= 100
    self.gym.set_dof_position_target_tensor(
      self.sim, gymtorch.unwrap_tensor(position_control.contiguous()))
    self.gym.set_dof_actuation_force_tensor(
      self.sim, gymtorch.unwrap_tensor(torque_control.contiguous()))
    # self.gym.set_dof_position_target_tensor_indexed(
    #     self.sim, gymtorch.unwrap_tensor(position_control.contiguous()))
    # self.gym.set_dof_actuation_force_tensor_indexed(
    #     self.sim, gymtorch.unwrap_tensor(torque_control.contiguous()))

    # self.gym.set_dof_position_target_tensor(
    #     self.sim, gymtorch.unwrap_tensor(torch.zeros_like(position_control)))
    # self.gym.set_dof_actuation_force_tensor(
    #     self.sim, gymtorch.unwrap_tensor(torch.zeros_like(torque_control)))
    # for _ in range(self.control_freq_inv):
    #     self.gym.simulate(self.sim)
    self.gym.simulate(self.sim)

    self.render()
    if self.device == 'cpu':
      self.gym.fetch_results(self.sim, True)
    self.post_physics_step()

  def step(self, actions):
    if self.use_controller:
      return self.controller_step(actions)

    if self.dr_randomizations.get('actions', None):
      actions = self.dr_randomizations['actions']['noise_lambda'](
        actions)

    # apply actions
    self.pre_physics_step(actions)

    # step physics and render each frame
    if self.diagonal_act:
      right_action, left_action = torch.chunk(
        self.action_scale * self.actions, 2, dim=-1)
      whole_action = torch.cat(
        [right_action, left_action, left_action, right_action], dim=-1)
      targets_pos = whole_action + self.default_dof_pos
    else:
      targets_pos = self.action_scale * self.actions + self.default_dof_pos

    # self.gym.set_dof_position_target_tensor(
    #     self.sim, gymtorch.unwrap_tensor(targets_pos))
    self.render()
    # this is the correct way to use action repeat with position control!
    for _ in range(self.control_freq_inv):
      self.gym.set_dof_position_target_tensor(
        self.sim, gymtorch.unwrap_tensor(targets_pos))
      self.gym.simulate(self.sim)

    # to fix!
    if self.device == 'cpu':
      self.gym.fetch_results(self.sim, True)

    # compute observations, rewards, resets, ...
    self.post_physics_step()

    if self.dr_randomizations.get('observations', None):
      self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](
        self.obs_buf)

  def post_physics_step(self):
    self.frame_count += 1
    self.progress_buf += 1

    change_commmand_env_ids = (torch.fmod(
      self.progress_buf, self.command_change_step) == 0).float().nonzero(as_tuple=False).squeeze(-1)
    if len(change_commmand_env_ids) > 0:
      self.reset_command(change_commmand_env_ids)
    self.check_termination()
    self.compute_reward()
    env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(env_ids) > 0:
      self.reset(env_ids)
    # self._update_viewer()
    if self.historical_step > 1:
      self.dof_pos_buf = torch.cat(
        [self.dof_pos_buf[:, :-1], self.dof_pos.unsqueeze(1)], dim=1)
      self.torques_buf = torch.cat(
        [self.torques_buf[:, :-1], self.torques.unsqueeze(1)], dim=1)

    if self.get_image and self.frame_count % self.vision_update_freq == 0:
      if self.headless:
        self.gym.step_graphics(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.sync_frame_time(self.sim)
        # render the camera sensors
      self.gym.render_all_camera_sensors(self.sim)
      image_vectors = torch.stack(
        [self.update_image(i) for i in range(self.num_envs)], dim=0)
      self.image_buf = torch.cat(
        [image_vectors.unsqueeze(1), self.image_buf[:, :-1]], dim=1)
    if self.get_image:
      self.obs_buf[:, self.state_obs_size:] = self.image_buf.flatten(1)
    self.compute_observations()

  def check_termination(self):
    """ Check if environments need to be reset
    """
    self.reset_buf = torch.any(torch.norm(
      self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    self.time_out_buf = self.progress_buf > self.max_episode_length
    self.reset_buf |= self.time_out_buf

  def _reward_feet_air_time(self):
    # Reward long steps
    contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
    first_contact = (self.feet_air_time > 0.) * contact
    self.feet_air_time += self.dt
    # reward only on first contact with the ground
    rew_airTime = torch.sum(
      (self.feet_air_time - 0.5) * first_contact, dim=1)
    # no reward for zero command
    rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
    self.feet_air_time *= ~contact
    return rew_airTime * self.rew_scales["feet_air_time"]

  def compute_reward(self):
    if self.command_type == "vel":
      root_states = self.root_states
      commands = self.commands
    elif self.command_type == "acc":
      root_states = self.root_states - self.last_root_states
      commands = self.commands * self.control_freq_inv / 120
    self.rew_buf[:], self.reset_buf[:] = self.compute_a1_reward(
      # tensors
      root_states,
      commands,
      self.dof_vel,
      self.torques,
      self.contact_forces,
      self.feet_indices,
      self.knee_indices,
      self.penalised_contact_indices,
      self.progress_buf,
      self.rew_scales,
      self.base_index,
      self.max_episode_length,
      self.a1_indices,
      self.vel_reward_exp_coeff,
      self.dof_vel_limits,
      self.torque_limits,
      self.actions,
      self.last_actions
    )
    self.rew_buf += self._reward_feet_air_time()

  def compute_observations(self):
    self.gym.refresh_dof_state_tensor(self.sim)  # done in step
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_jacobian_tensors(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)
    self.gym.refresh_dof_force_tensor(self.sim)
    contact_forces = (
      self.contact_forces[:, self.feet_indices_in_bodies]).flatten(1)
    if self.historical_step > 1:
      dof_pos = (self.dof_pos_buf -
                 self.default_dof_pos.unsqueeze(1)).view(self.num_envs, -1)
      actions = self.actions_buf.view(self.num_envs, -1)
      self.obs_buf[:, :self.state_obs_size] = compute_a1_observations(  # tensors
        self.root_states,
        self.commands,
        dof_pos,
        self.dof_vel,
        self.gravity_vec,
        self.action_scale.repeat(self.historical_step) * actions,
        # scales
        self.lin_vel_scale,
        self.ang_vel_scale,
        self.dof_pos_scale,
        self.dof_vel_scale,
        self.a1_indices,
        contact_forces,
      )
    else:
      self.obs_buf[:, :self.state_obs_size] = compute_a1_observations(  # tensors
        self.root_states,
        self.commands,
        self.dof_pos - self.default_dof_pos,
        self.dof_vel,
        self.gravity_vec,
        self.action_scale * self.actions,
        # scales
        self.lin_vel_scale,
        self.ang_vel_scale,
        self.dof_pos_scale,
        self.dof_vel_scale,
        self.a1_indices,
        contact_forces,
      )

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
          self.soft_limits["soft_dof_vel_limit"]
        self.dof_pos_limits[i, 1] = m + 0.5 * r * \
          self.soft_limits["soft_dof_vel_limit"]
    for i in range(self.num_dof):
      props['driveMode'][i] = gymapi.DOF_MODE_POS
      props['stiffness'][i] = self.Kp
      props['damping'][i] = self.Kd
    return props

  def apply_reward_randomizations(self, rr_params):
    rand_freq = rr_params.get("frequency", 1)

    self.last_step = self.gym.get_frame_count(self.sim)

    do_rew_randomize = (
      self.last_step - self.last_rew_rand_step) >= rand_freq
    if do_rew_randomize:
      self.last_rew_rand_step = self.last_step

    scale_params = rr_params["reward_scale"]
    for k, v in scale_params.items():
      v_range = v["range"]
      self.rew_scales[k] = np.random.uniform(
        low=v_range[0], high=v_range[1]) * self.dt

  def reset(self, env_ids):
    # Randomization can happen only at reset time, since it can reset actor positions on GPU
    if self.randomize:
      self.apply_randomizations(self.randomization_params)

    if self.randomize_reward:
      self.apply_reward_randomizations(self.reward_randomization_params)

    # positions_offset = torch_rand_float(
    #     0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
    velocities = torch_rand_float(-0.05, 0.05,
                                  (len(env_ids), self.num_dof), device=self.device)

    self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + 0.1 * \
      (torch.rand_like(
        self.default_dof_pos[env_ids], device=self.device) - 0.5)
    self.dof_vel[env_ids] = velocities
    # self.dof_vel[env_ids] = 0

    env_ids_int32 = env_ids.to(dtype=torch.int32)
    a1_indices = self.a1_indices[env_ids].to(torch.int32)
    self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                 gymtorch.unwrap_tensor(
                                                   self.initial_root_states),
                                                 gymtorch.unwrap_tensor(a1_indices), len(env_ids_int32))
    self.gym.set_dof_state_tensor_indexed(self.sim,
                                          gymtorch.unwrap_tensor(
                                            self.dof_state),
                                          gymtorch.unwrap_tensor(a1_indices), len(env_ids_int32))

    self.commands_x[env_ids] = torch_rand_float(
      self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
    self.commands_y[env_ids] = torch_rand_float(
      self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
    self.commands_yaw[env_ids] = torch_rand_float(
      self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

    if self.historical_step > 1:
      self.dof_pos_buf[env_ids] = 0
      self.actions_buf[env_ids] = 0
      self.torques_buf[env_ids] = 0
    self.progress_buf[env_ids] = 0
    self.reset_buf[env_ids] = 1
    self.feet_air_time[env_ids] = 0

  def update_image(self, env_ids):
    # output image and then write it to disk using Pillow
    # communicate physics to graphics system
    image_vec = []
    if self.image_type != "rgb":
      depth_image = self.gym.get_camera_image_gpu_tensor(
        self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_DEPTH)
      depth_image = gymtorch.wrap_tensor(depth_image)

      # -inf implies no depth value, set it to zero. output will be black.
      depth_image[torch.isneginf(depth_image)] = 0

      # clamp depth image to 10 meters to make output image human friendly
      depth_image[depth_image < -5] = -5

      # depth_image = (depth_image - torch.mean(depth_image)) / \
      #     (torch.std(depth_image) + 1e-5)
      image_vec.append(depth_image.unsqueeze(0))

    if self.image_type != "depth":
      rgba_image = self.gym.get_camera_image_gpu_tensor(
        self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_COLOR)
      rgb_image = gymtorch.wrap_tensor(rgba_image)[:3].float()
      rgb_image = rgba_image / 255.

      image_vec.append(rgb_image)

    image_vec = torch.cat(image_vec, dim=0).flatten()
    # # flip the direction so near-objects are light and far objects are dark
    # normalized_depth = -255.0 * \
    #     (depth_image / torch.min(depth_image + 1e-4))
    # normalized_depth = normalized_depth.cpu().numpy()
    # normalized_depth_image = im.fromarray(
    #     normalized_depth.astype(np.uint8), mode="L")
    # normalized_depth_image.save(
    #     "output_images/depth_{}.png".format(self.frame_count))
    # self.gym.write_camera_image_to_file(
    #     self.sim, self.envs[env_ids], self.camera_handles[env_ids], gymapi.IMAGE_COLOR, "output_images/rgb_{}.png".format(self.frame_count))
    # self.frame_count += 1
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

    elif self.command_type == "acc":
      self.commands_x[env_ids] = - self.commands_x[env_ids]
      self.commands_y[env_ids] = - self.commands_y[env_ids]
      self.commands_yaw[env_ids] = - self.commands_yaw[env_ids]

  def _update_viewer(self):
    if self.viewer is not None:
      lookat = self.root_states[self.a1_indices[self.refEnv], 0:3]
      p = [_lookat - _camera_distance for _camera_distance,
           _lookat in zip(self.camera_distance, lookat)]
      cam_pos = gymapi.Vec3(p[0], p[1], p[2])
      cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
      self.gym.viewer_camera_look_at(
        self.viewer, None, cam_pos, cam_target)

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

  def _footPositionsInBaseFrame(self):
    """Get the robot's foot position in the base frame."""
    angles = self.dof_pos
    angles = angles.reshape(self.num_envs, self.num_legs, 3)
    foot_positions = torch.zeros_like(angles, device=self.device)
    for i in range(self.num_legs):
      foot_positions[:, i] = self._foot_position_in_hip_frame(
        angles[:, i], l_hip_sign=(-1)**(i + 1))
    return foot_positions + self._hip_offset

  def _footPositionsToJointAngles(self, foot_positions):
    fjt = self.jacobian_tensor[:, :, :, 6:]
    fjt = fjt[:, self.feet_indices + 1]
    # solve damped least squares
    fjt_T = torch.transpose(fjt, -1, -2)
    feet_pos = self._footPositionsInBaseFrame()
    feet_pos_err, feet_rot_err = torch.zeros_like(
      feet_pos), torch.zeros_like(feet_pos)
    feet_pos_err = foot_positions - feet_pos
    dof_err = torch.cat([feet_pos_err, feet_rot_err],
                        dim=-1).unsqueeze_(-1)
    d = 0.05  # damping term
    lmbda = torch.eye(6).unsqueeze_(0).unsqueeze_(0).repeat(
      self.num_envs, 4, 1, 1).to(self.device) * (d ** 2)
    u = (fjt_T @ torch.inverse(fjt @ fjt_T + lmbda) @ dof_err).squeeze(-1)
    # u = (fjt_T @ dof_err).squeeze(-1)
    delta_pos = []
    for i in range(self.num_legs):
      delta_pos.append(u[:, i, 3 * i: 3 * (i + 1)])
    delta_pos = torch.cat(delta_pos, dim=-1)
    pos_target = self.dof_pos + delta_pos
    return pos_target

  def _getBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    Returns:
    A tuple (roll, pitch, yaw) of the base in world frame.
    """
    base_quat = self.root_states[self.a1_indices, 3:7]
    roll_pitch_yaw = matrix_to_euler_angles(
      quaternion_to_matrix(base_quat), "XYZ")
    return roll_pitch_yaw

  def _getBaseRollPitchYawRate(self):
    quat = self.root_states[self.a1_indices, 3:7]
    ang_vel = quat_rotate_inverse(
      quat, self.root_states[self.a1_indices, 10:13])
    return ang_vel

  def _getHipPositionsInBaseFrame(self):
    return self._default_hip_positions

  def _getContactFootState(self):
    contact_forces = self.contact_forces[:, self.feet_indices].sum(-1)
    contact_states = (contact_forces != 0)
    return contact_states

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_a1_reward(
    root_states: Tensor,
    commands: Tensor,
    dof_vel: Tensor,
    torques: Tensor,
    contact_forces: Tensor,
    feet_indices: Tensor,
    knee_indices: Tensor,
    penalised_contact_indices: Tensor,
    episode_lengths: Tensor,
    rew_scales: Dict[str, float],
    base_index: int,
    max_episode_length: int,
    a1_indices: Tensor,
    vel_reward_exp_coeff: float,
    dof_vel_limits: Tensor,
    torque_limits: Tensor,
    actions: Tensor,
    last_actions: Tensor
) -> Tuple[Tensor, Tensor]:

  base_quat = root_states[a1_indices, 3:7]
  base_lin_vel = quat_rotate_inverse(
    base_quat, root_states[a1_indices, 7:10])
  base_ang_vel = quat_rotate_inverse(
    base_quat, root_states[a1_indices, 10:13])

  lin_vel_error = torch.sum(torch.square(
    commands[:, :2] - base_lin_vel[:, :2]), dim=1)
  ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
  rew_lin_vel_xy = torch.exp(-lin_vel_error / vel_reward_exp_coeff) * \
    rew_scales["linearVelocityXYRewardScale"]
  rew_ang_vel_z = torch.exp(-ang_vel_error / vel_reward_exp_coeff) * \
    rew_scales["angularVelocityZRewardScale"]
  # z velocity penalty
  rew_z_vel = torch.square(
    base_lin_vel[:, 2]) * rew_scales["linearVelocityZRewardScale"]

  # torque penalty
  rew_torque = torch.sum(torch.square(torques), dim=1) * \
    rew_scales["torqueRewardScale"]

  # penalize the actions that are closer to hardware limits

  dof_vel_limit_rew = rew_scales["dof_vel_limit"] * torch.sum((torch.abs(
    dof_vel) - dof_vel_limits * rew_scales["dof_vel_limit"]).clip(min=0.), dim=1)
  torque_limit_rew = rew_scales["torque_limit"] * torch.sum((torch.abs(
    torques) - torque_limits * rew_scales["torque_limit"]).clip(min=0.), dim=1)

  # smooth action
  rew_action_smoothing = torch.sum(torch.square(
    last_actions - actions), dim=1) * rew_scales["actionSmoothingRewardScale"]

  # Penalize feet stumble
  rew_stumble = torch.any(torch.norm(contact_forces[:, feet_indices, :2], dim=2) > 5 * torch.abs(
    contact_forces[:, feet_indices, 2]), dim=1) * rew_scales["stumble"]

  # Penalize collision on specific dofs
  rew_collision = torch.sum(
    (torch.norm(contact_forces[:, penalised_contact_indices, :], dim=-1) > 0.1), dim=1) * rew_scales["collision"]

  total_reward = rew_lin_vel_xy + rew_ang_vel_z + \
    rew_torque + rew_z_vel + rew_action_smoothing + rew_stumble + rew_collision + \
    dof_vel_limit_rew + torque_limit_rew
  total_reward = torch.clip(total_reward, 0., None)

  # reset agents
  reset = torch.norm(contact_forces[:, base_index, :], dim=1) > 1.
  reset = reset | torch.any(torch.norm(
    contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)

  # reset due to fall
  reset = reset | (base_quat[:, -1] < 0.6)

  # no terminal reward for time-outs
  time_out = episode_lengths > max_episode_length
  reset = reset | time_out

  return total_reward.detach(), reset


@torch.jit.script
def compute_a1_observations(root_states: Tensor,
                            commands: Tensor,
                            dof_pos: Tensor,
                            dof_vel: Tensor,
                            gravity_vec: Tensor,
                            actions: Tensor,
                            lin_vel_scale: float,
                            ang_vel_scale: float,
                            dof_pos_scale: float,
                            dof_vel_scale: float,
                            a1_indices: Tensor,
                            contact_forces: Tensor
                            ) -> Tensor:

  # base_position = root_states[:, 0:3]
  base_quat = root_states[a1_indices, 3:7]
  base_lin_vel = quat_rotate_inverse(
    base_quat, root_states[a1_indices, 7:10]) * lin_vel_scale
  base_ang_vel = quat_rotate_inverse(
    base_quat, root_states[a1_indices, 10:13]) * ang_vel_scale
  projected_gravity = quat_rotate(base_quat, gravity_vec)

  dof_pos_scaled = dof_pos * dof_pos_scale

  commands_scaled = commands[..., :3] * \
    torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale],
                 requires_grad=False, device=commands.device)

  obs = torch.cat((base_lin_vel,
                   base_ang_vel,
                   projected_gravity,
                   commands_scaled,
                   commands[..., 3:],
                   dof_pos_scaled,
                   dof_vel * dof_vel_scale,
                   actions,
                   contact_forces,
                   ), dim=-1)

  return obs
