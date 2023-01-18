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


class LSTMCPGTRTPolicyWrapper():
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

    self.gait_generator = GaitGenerator(
        stance_duration=stance_duration
    )
    self.foot_clearance = foot_clearance

    self.end_feet_pos = np.array(
        [0.05, 0.0, self.foot_clearance] * 4,
    )

    # from minghao's validation data
    self.default_feet_pos = np.array([
        [0.1478, -0.11459, -0.45576],
        [0.1478, 0.11688, -0.45576],
        [-0.2895, -0.11459, -0.45576],
        [-0.2895, 0.11688, -0.45576]
    ])

    self.foot_target_position = self.end_feet_pos.reshape(
        4, 3) + self.default_feet_pos

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
      observation,
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

      time_phase_info = self.gait_generator.time_phase_info.reshape(-1)
      print(time_phase_info)

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
      # if self.use_com_vel:
      #   obs_list.append(com_vel_normalized)  # 3
      # if self.use_foot_contact:
      #   obs_list.append(foot_contact_normalized)
      obs_list = [
          joint_angle_hist_normalized,  # 12
          joint_vel_hist_normalized,  # 12 * 2
          proj_g_normalized,  # 3
          last_action_normalized,  # 16 * 2
          phase_info_hist_normalized,  # 8
          time_phase_info,
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
          time_phase_info,
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

  def process_act(self, action):
    if self.vis_only:
      return action
    else:
      action_normalized = action
      action_normalized = np.clip(action_normalized, -1, 1)
      action = self.action_scale * action_normalized.flatten()
    return action

  def get_action(
      self, observation,
      hidden_state, depth_frame, depth_scale, last_action
  ):
    '''
    This function process raw observation, fed normalized observation into
    the network, de-normalize and output the action.
    '''
    ob_t = self.process_obs(
        observation,
        depth_frame, depth_scale, last_action
    )
    ob_t = np.clip(ob_t, -5, 5)
    action, hidden_state = self.pf.act_inference(ob_t, hidden_state)
    print(action.shape, hidden_state.shape)
    action = self.process_act(action)
    # action = torch.cat([
    #   torch.zeros(12),
    #   -1 * torch.Tensor(self.nominal_residual_phase)
    # ])
    target_joint_pos = self.get_cpg_action(action)
    if self.save_log:
      self.policy_action_saver.record(action)
    return action, target_joint_pos, hidden_state

  def get_cpg_action(self, action):
    self.residual_joint_angle = action[..., :12]
    self.residual_phase = action[..., 12:]
    if not self.cpg_mode:
      return None
    # t = time.time() - self.phase_start_time# - self.phase_dt
    # t = self.previous_end_time - self.phase_start_time
    # self.previous_end_time = time.time()
    self.phase_time += self.phase_dt
    # calculate joint position command from stored cpg command
    # 1. update gait generator to generate phase
    self.gait_generator.update(
        self.phase_time - self.phase_dt + 0.02,
        # self.phase_time,
        # t + 0.02,
        self.residual_phase + self.nominal_residual_phase
    )
    # 2. calculate swing_feet_pos
    #    based on "gait_generator", "default_feet_pos", "end_feet_pos"
    swing_feet_pos = self._gen_swing_foot_trajectory(
        self.gait_generator._normalized_phase, self.default_feet_pos, self.foot_target_position)

    # 3. select from "swing_feet_pos" or "stance_feet_pos" based on gait generator
    stance_feet_pos = self.default_feet_pos

    targets_feet_pos = (
        self.gait_generator.desired_leg_state[..., np.newaxis] != 0
    ).astype(np.float) * stance_feet_pos + (
        self.gait_generator.desired_leg_state[..., np.newaxis] == 0
    ).astype(np.float) * swing_feet_pos

    # # 4. use Inverse Kinematics to calculate the desired joint position
    # target_joint_pos = multiple_leg_inverse_kinematics_isaac(
    #     targets_feet_pos, leg_length=torch.tensor(
    #         [0.08505, 0.2, 0.2], device=self.device)
    # )[0]
    # # 5. add residual joint position command to calculated joint position
    # target_joint_pos = ik_joint_pos + self.residual_joint_angle
    # target_joint_pos = target_joint_pos.detach().cpu().numpy()
    # target_joint_pos = convert_order_from_isaac_to_ros(target_joint_pos)
    target_joint_pos = self._footPositionsToJointAngles(
        targets_feet_pos, self.current_joint_angle)
    target_joint_pos = target_joint_pos.reshape(-1) + self.residual_joint_angle
    target_joint_pos = convert_order_from_isaac_to_ros(target_joint_pos)
    return target_joint_pos

  def _footPositionsToJointAngles(self, foot_positions, dof_pos):
    # dof_pos = torch.Tensor(dof_pos).to(self.device)
    feet_err = foot_positions - \
        self.default_feet_pos[np.newaxis, ...]
    self._compute_leg_jacobian(dof_pos)
    u = np.squeeze((np.linalg.inv(self.leg_jacobian) @
                    feet_err[..., np.newaxis]), -1)
    pos_target = u.reshape(4, 3) + \
        self.default_dof_pos
    return pos_target

  def _compute_leg_jacobian(self, dof_pos):
    l1 = 0.0838
    l2 = 0.2
    l3 = 0.2

    # leg_length = torch.tensor([0.08505, 0.2, 0.2]).to(device)

    # Only for A1
    joint_pos = dof_pos.reshape(
        4, 3
    )

    self.leg_jacobian = np.zeros(
        (4, 3, 3)
    )
    side_sign = np.array(
        [1, -1, 1, -1]
    ).reshape((1, 4))

    s1 = np.sin(joint_pos[..., 0])
    s2 = np.sin(joint_pos[..., 1])
    s3 = np.sin(joint_pos[..., 2])

    c1 = np.cos(joint_pos[..., 0])
    c2 = np.cos(joint_pos[..., 1])
    c3 = np.cos(joint_pos[..., 2])

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

  def _gen_parabola(self, phase, start, mid, end):
    """Gets a point on a parabola y = a x^2 + b x + c.

    The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
    the plane.

    Args:
      phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
      start: The y value at x == 0.
      mid: The y value at x == 0.5.
      end: The y value at x == 1.

    Returns:
      The y value at x == phase.
    """
    mid_phase = 0.5
    delta_1 = mid - start
    delta_2 = end - start
    delta_3 = mid_phase**2 - mid_phase
    coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
    coef_b = (delta_2 * mid_phase**2 - delta_1) / delta_3
    coef_c = start

    return coef_a * phase**2 + coef_b * phase + coef_c

  # @torch.jit.script

  def _gen_swing_foot_trajectory(self, input_phase, start_pos,
                                 end_pos):
    """Generates the swing trajectory using a parabola.

    Args:
      input_phase: the swing/stance phase value between [0, 1].
      start_pos: The foot's position at the beginning of swing cycle.
      end_pos: The foot's desired position at the end of swing cycle.

    Returns:
      The desired foot position at the current phase.
    """
    # We augment the swing speed using the below formula. For the first half of
    # the swing cycle, the swing leg moves faster and finishes 80% of the full
    # swing trajectory. The rest 20% of trajectory takes another half swing
    # cycle. Intuitely, we want to move the swing foot quickly to the target
    # landing location and stay above the ground, in this way the control is more
    # robust to perturbations to the body that may cause the swing foot to drop
    # onto the ground earlier than expected. This is a common practice similar
    # to the MIT cheetah and Marc Raibert's original controllers.
    # print(input_phase)
    input_phase = np.clip(input_phase, 0, 1)
    phase = np.zeros_like(input_phase)
    indices = np.nonzero(input_phase <= 0.5)
    non_indices = np.nonzero(input_phase > 0.5)
    phase[indices] = 0.8 * np.sin(input_phase[indices] * math.pi)
    phase[non_indices] = 0.8 + (input_phase[non_indices] - 0.5) * 0.4
    x = (1 - phase) * start_pos[..., 0] + phase * end_pos[..., 0]
    y = (1 - phase) * start_pos[..., 1] + phase * end_pos[..., 1]
    max_clearance = 0.1
    mid = np.maximum(end_pos[..., 2], start_pos[..., 2]) + max_clearance
    z = self._gen_parabola(phase, start_pos[..., 2], mid, end_pos[..., 2])

    return np.stack([x, y, z], axis=-1)

  def write(self):
    if self.save_log:
      self.ob_tensor_saver.write()
      self.policy_action_saver.write()

  def set_cpg_mode(self, mode):
    if mode:
      self.phase_start_time = time.time()
      self.previous_end_time = self.phase_start_time
    self.cpg_mode = mode
