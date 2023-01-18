from a1_utilities.a1_sensor_histories import NormedStateHistory
from a1_utilities.a1_sensor_process import convert_order_from_isaac_to_ros, convert_order_from_ros_to_isaac, observation_to_joint_position_isaac, observation_to_joint_vel, observation_to_torque
from a1_utilities.logger import StateLogger
from a1_utilities.a1_sensor_histories import VisualHistory
import numpy as np
import torch
from torch import Tensor
import math
import time

from a1_utilities.cpg.gait_generator import GaitGenerator
from a1_utilities.cpg.A1_inverse_kinematics import multiple_leg_inverse_kinematics_isaac



@torch.jit.script
def quat_rotate(q, v):
  shape = q.shape
  q_w = q[:, -1]
  q_vec = q[:, :3]
  a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
  b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
  c = q_vec * \
      torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
          shape[0], 3, 1)).squeeze(-1) * 2.0
  return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
  shape = q.shape
  q_w = q[:, -1]
  q_vec = q[:, :3]
  a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
  b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
  c = q_vec * \
      torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
          shape[0], 3, 1)).squeeze(-1) * 2.0
  return a - b + c


class LSTMCPGPolicyWrapperV2():
  def __init__(
      self,
      policy,
      obs_scale,
      get_image_interval,
      save_dir_name,
      sliding_frames=False, no_tensor=False,
      default_joint_angle=None,
      twoview=False,
      vis_only=False,
      use_com_vel=False,
      use_command=False,
      state_only=False,
      use_foot_contact=False,
      save_log=False,
      num_hist=3,
      phase_scale=0.1,
      action_scale=[1.0, 1.0, 1.0],
      # CPG settings
      stance_duration=0.2,
      foot_clearance=0.025,
      Kp=60, Kd=0.8,
      phase_dt=0.05
  ):

    self.pf = policy
    self.no_tensor = no_tensor

    self.get_image_interval = get_image_interval

    self.vis_only = vis_only
    self.state_only = state_only

    self.obs_scale = obs_scale
    self.device = "cpu"
    #
    self.action_scale = action_scale * 4
    self.phase_scale = [phase_scale] * 4
    self.action_scale = np.array(
        self.action_scale + self.phase_scale, dtype=np.float32)
    self.hist_action_scale = np.hstack([self.action_scale] * num_hist)
    self.action_scale = torch.tensor(
        self.action_scale, device="cuda:0"
    )

    self.gait_generator = GaitGenerator(
        stance_duration=stance_duration
    )

    self.default_dof_pos = np.array([
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7]
    ])
    self.flatten_default_dof_pos = self.default_dof_pos.reshape(-1)
    self.Kp = Kp
    self.Kd = Kd

    # FIXME: need to see what is this nominal residue phase
    self.nominal_residual_phase = 1.0 * np.array([-0.2, 0.2, 0.2, -0.2])

    self.twoview = twoview

    self.phase_time = 0
    self.phase_dt = phase_dt
    self.cpg_mode = False

    if not self.vis_only:
      self.last_action_historical_data = NormedStateHistory(
          input_dim=16,
          num_hist=num_hist,
          mean=np.zeros(num_hist * 16),
          var=np.ones(num_hist * 16)
      )

      self.joint_angle_historical_data = NormedStateHistory(
          input_dim=12,
          num_hist=num_hist,
          mean=np.zeros(num_hist * 12),
          var=np.ones(num_hist * 12)
      )

    # self.frames_historical_data = VisualHistory(
    #     frame_shape=(64, 64),
    #     num_hist=1,
    #     mean=1.25 * np.ones(
    #         64 * 64 * 1
    #     ),   # FIXME: Mean measured = 1.02
    #     var=(0.45 ** 2) * np.ones(
    #         64 * 64 * 1
    #     ),    # FIXME: Variance measured = 0.11
    #     sliding_frames=sliding_frames
    # )  # variance, not std!

    self.frames_historical_data = VisualHistory(
        frame_shape=(64, 64),
        num_hist=1,
        mean=np.zeros(
            64 * 64 * 1
        ),   # FIXME: Mean measured = 1.02
        var=np.ones(
            64 * 64 * 1
        ),    # FIXME: Variance measured = 0.11
        sliding_frames=sliding_frames
    )  # variance, not std!

    if self.twoview:
      self.secview_frames_historical_data = VisualHistory(
          frame_shape=(64, 64),
          num_hist=get_image_interval * 3 + 1,
          mean=1.25 * np.ones(
              64 * 64 * (get_image_interval * 3 + 1)
          ),   # FIXME: Mean measured = 1.02
          var=0.425 ** 2 * np.ones(
              64 * 64 * (get_image_interval * 3 + 1)
          ),    # FIXME: Variance measured = 0.11
          sliding_frames=sliding_frames
      )  # variance, not std!

    self.save_log = save_log
    if self.save_log:
      # array savers
      self.ob_tensor_saver = StateLogger(
          np.zeros((4184), dtype=np.float32),
          duration=60,
          frequency=25,
          data_save_name=save_dir_name + "ob_t.npz"
      )

      self.policy_action_saver = StateLogger(
          np.zeros((16), dtype=np.float32),
          duration=60,
          frequency=25,
          data_save_name=save_dir_name + "policy_action.npz"
      )

  def process_obs(
      self,
      observation, com_vel,
      depth_frame,
      depth_scale,
      last_action
  ):
    if not self.vis_only:

      # joint angle
      joint_angle = convert_order_from_ros_to_isaac(np.array(observation.q))
      joint_vel = convert_order_from_ros_to_isaac(np.array(observation.dq))
      self.current_joint_angle = joint_angle
      # print("Joint Time:", time.time()-joint_start)
      joint_angle_hist_normalized = self.joint_angle_historical_data.record(
          joint_angle - self.flatten_default_dof_pos,
          backwards=True
      ) * self.obs_scale["joint_angle"]

      self.current_joint_vel = joint_vel
      joint_vel_hist_normalized = joint_vel * self.obs_scale["joint_vel"]

      # Phase info in gait generator is already
      phase_info = self.gait_generator.phase_info.reshape(-1)  # .cpu().numpy()
      phase_info_hist_normalized = phase_info * self.obs_scale["phase_info"]

      last_action = last_action
      last_action_normalized = self.last_action_historical_data.record(
          last_action,
          backwards=True
      )  # * self.hist_action_scale

      # proj_start = time.time()
      quat = observation.imu.quaternion
      quat = np.concatenate(
          [quat[1:], quat[0:1]]
      )
      proj_g = quat_rotate_inverse(
          torch.tensor(quat).float().unsqueeze(0),
          torch.tensor([0, 0, -1.0]).float().unsqueeze(0)
      ).squeeze().cpu().numpy()
      proj_g_normalized = proj_g * self.obs_scale["proj_g"]

      # append new frame everytime and give index 0,4,8,12
      normalized_visual_history = self.frames_historical_data.record_and_normalize(
          depth_frame, depth_scale, backwards=False)
      # print(normalized_visual_history)

    # concatnate all observations and feed into network
    if self.vis_only:
      obs_normalized_np = normalized_visual_history.reshape(-1)
    elif self.state_only:
      # assert False
      obs_list = []
      obs_list = [
          joint_angle_hist_normalized,  # 12
          joint_vel_hist_normalized,  # 12 * 2
          proj_g_normalized,  # 3
          last_action_normalized,  # 16 * 2
          phase_info_hist_normalized,  # 8
      ]
      obs_normalized_np = np.hstack(obs_list)
    else:
      obs_list = []
      obs_list = [
          joint_angle_hist_normalized,  # 12
          joint_vel_hist_normalized,  # 12 * 2
          proj_g_normalized,  # 3
          # last_action_normalized,  # 16 * 2
          phase_info_hist_normalized,  # 8
          normalized_visual_history.reshape(-1)
      ]
      # print(normalized_visual_history)
      obs_normalized_np = np.hstack(obs_list)
      # print(obs_normalized_np.shape)

    if self.save_log:
      self.ob_tensor_saver.record(obs_normalized_np)

    # TODO: move below part to GPU and profile
    if not self.no_tensor:
      ob_t = torch.Tensor(obs_normalized_np).unsqueeze(
          0).unsqueeze(0).to("cuda:0")
    else:
      ob_t = obs_normalized_np[np.newaxis, :]
    return ob_t

  def process_obs_twoview(
      self,
      observation,
      front_depth_frame, front_depth_scale,
      neck_depth_frame, neck_depth_scale,
      last_action
  ):
    if not self.vis_only:
      # IMU
      base_ang_hist_normalized = self.base_ang_historical_data.record_and_normalize(
          np.array(
              observation["imu_rpy"][:2]
          )  # R, P, dR, dP; dR,dP are not literally speed of roll, but angular velocity on corresponding axis in body frame.
      ) * self.obs_scale["base_ang"]

      base_ang_vel_hist_normalized = self.base_ang_vel_historical_data.record_and_normalize(
          np.array(
              observation["imu_gyroscope"][:3]
          )  # R, P, dR, dP; dR,dP are not literally speed of roll, but angular velocity on corresponding axis in body frame.
      ) * self.obs_scale["base_ang_vel"]

      # joint angle
      # joint_angle = observation_to_joint_position(observation)
      joint_angle = convert_order_from_ros_to_isaac(
          observation["joint_position"])
      self.current_joint_angle = joint_angle
      joint_angle_hist_normalized = self.joint_angle_historical_data.record_and_normalize(
          joint_angle
      ) * self.obs_scale["joint_angle"]

      joint_vel = convert_order_from_ros_to_isaac(observation["joint_vel"])
      joint_vel_hist_normalized = self.joint_vel_historical_data.record_and_normalize(
          joint_vel
      ) * self.obs_scale["joint_vel"]

      # last action
      # last_action = convert_order_from_ros_to_isaac(last_action)
      last_action = last_action
      last_action_normalized = self.last_action_historical_data.record_and_normalize(
          last_action
      ) * self.obs_scale["last_action"]
      if self.use_foot_contact:
        foot_contact = np.array(observation.footForce) > 20
        # print(np.array(observation.footForce))
        foot_contact = convert_order_from_ros_to_isaac(foot_contact)
        # foot_contact_normalized = foot_contact
        foot_contact_normalized = self.foot_contact_historical_data.record_and_normalize(
            foot_contact
        ) * self.obs_scale["foot_contact"]
      # append new frame everytime and give index 0,4,8,12
      front_normalized_visual_history = self.front_frames_historical_data.record_and_normalize(
          front_depth_frame, front_depth_scale, backwards=True
      )
      neck_normalized_visual_history = self.neck_frames_historical_data.record_and_normalize(
          neck_depth_frame, neck_depth_scale, backwards=True
      )

    # concatnate all observations and feed into network
    if self.vis_only:
      obs_list = [
          front_normalized_visual_history.reshape(-1),
          neck_normalized_visual_history.reshape(-1)
      ]
      obs_normalized_np = np.hstack(obs_list)
    elif self.state_only:
      obs_list = []
      obs_list += [
          joint_angle_hist_normalized,  # 12 * 2
          joint_vel_hist_normalized,  # 12 * 2
          base_ang_hist_normalized,  # 2 * 2
          base_ang_vel_hist_normalized,  # 2 * 2
          last_action_normalized,  # 12 * 2
      ]
      if self.use_foot_contact:
        obs_list.append(foot_contact_normalized)  # 12
      obs_normalized_np = np.hstack(obs_list)
      print("State Obs")
      print(obs_normalized_np)
    else:
      obs_list = []
      if self.use_foot_contact:
        obs_list.append(foot_contact_normalized)
      obs_list += [
          joint_angle_hist_normalized,
          joint_vel_hist_normalized,
          base_ang_hist_normalized,
          base_ang_vel_hist_normalized,
          last_action_normalized,
          front_normalized_visual_history.reshape(-1),
          neck_normalized_visual_history.reshape(-1)
      ]
      obs_normalized_np = np.hstack(obs_list)

    if self.save_log:
      self.ob_tensor_saver.record(obs_normalized_np)

    # TODO: move below part to GPU and profile
    if not self.no_tensor:
      import torch
      ob_t = torch.Tensor(obs_normalized_np).unsqueeze(0).to("cuda:0")
    else:
      ob_t = obs_normalized_np[np.newaxis, :]
    return ob_t

  def process_act(self, action):
    if self.vis_only:
      return action
    else:
      # print(action.shape)
      # action = action.squeeze().cpu().numpy()
      action = action.detach()
      action_normalized = action
      # action_normalized = np.clip(action_normalized, -1, 1)
      action_normalized = torch.clamp(action_normalized, -1, 1)
      # action = self.action_scale * action_normalized.flatten()
      action = self.action_scale * action_normalized.reshape(-1)
      action = action.cpu().numpy()
    return action

  def get_action(
          self, observation, com_vel,
          hidden_state, mask, depth_frame, depth_scale, last_action):
    '''
    This function process raw observation, fed normalized observation into
    the network, de-normalize and output the action.
    '''
    ob_t = self.process_obs(
        observation, com_vel,
        depth_frame, depth_scale, last_action)
    ob_t = torch.clamp(ob_t, -5, 5)
    c_t = time.time()
    with torch.no_grad():
      action, hidden_state = self.pf.act_inference(ob_t, hidden_state, mask)
    print("inference time:", time.time() - c_t)
    action = self.process_act(action)
    if self.cpg_mode:
      self.residual_joint_angle = action[..., :12]
      self.residual_phase = action[..., 12:]
      self.phase_time += self.phase_dt
      self.gait_generator.update(
          self.phase_time - self.phase_dt + 0.02,
          self.residual_phase + self.nominal_residual_phase
      )
    if self.save_log:
      self.policy_action_saver.record(action)
    return action, hidden_state, self.phase_time, self.current_joint_angle

  def write(self):
    if self.save_log:
      self.ob_tensor_saver.write()
      self.policy_action_saver.write()

  def set_cpg_mode(self, mode):
    if mode:
      self.phase_start_time = time.time()
      self.previous_end_time = self.phase_start_time
    self.cpg_mode = mode
