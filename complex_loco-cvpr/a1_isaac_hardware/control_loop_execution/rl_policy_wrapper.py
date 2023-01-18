from a1_utilities.a1_sensor_histories import NormedStateHistory
from a1_utilities.a1_sensor_process import convert_order_from_isaac_to_ros, convert_order_from_ros_to_isaac, observation_to_joint_position_isaac, observation_to_joint_vel, observation_to_torque
from a1_utilities.logger import StateLogger
from a1_utilities.a1_sensor_histories import VisualHistory
import numpy as np
import torch


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


class PolicyWrapper():
  def __init__(
      self,
      policy,
      obs_scale,
      get_image_interval,
      save_dir_name,
      sliding_frames=True, no_tensor=False,
      default_joint_angle=None,
      twoview=False,
      action_range=None,
      vis_only=False,
      state_only=False,
      save_log=False,
      num_hist=3,
      use_inverse_depth=False,
      use_original_depth=False
  ):

    self.process_count = 0
    self.pf = policy
    self.no_tensor = no_tensor

    self.get_image_interval = get_image_interval

    self.vis_only = vis_only
    self.twoview = twoview
    self.state_only = state_only

    if default_joint_angle == None:
      default_joint_angle = [0.0, 0.9, -1.7]
    self.default_joint_angle = np.array(default_joint_angle * 4)
    self.default_dof_pos = np.array([
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7]
    ])
    self.flatten_default_dof_pos = self.default_dof_pos.reshape(-1)

    self.current_joint_angle = default_joint_angle
    if action_range == None:
      action_range = [0.05, 0.5, 0.5]
    self.action_range = np.array(action_range * 4)

    self.hist_action_scale = np.array(action_range * num_hist * 4)

    self.action_lb = self.default_joint_angle - self.action_range
    self.action_ub = self.default_joint_angle + self.action_range

    self.obs_scale = obs_scale

    if not self.vis_only:
      # self.proj_g_historical_data = NormedStateHistory(
      #     input_dim=3,
      #     num_hist=1,
      #     mean=np.zeros(1 * 3),
      #     var=np.ones(1 * 3)
      # )

      self.last_action_historical_data = NormedStateHistory(
          input_dim=12,
          num_hist=num_hist,
          mean=np.zeros(num_hist * 12),
          var=np.ones(num_hist * 12)
      )

      self.joint_angle_historical_data = NormedStateHistory(
          input_dim=12,
          num_hist=num_hist,
          mean=np.zeros(num_hist * 12),
          var=np.ones(num_hist * 12)
      )

      # self.joint_vel_historical_data = NormedStateHistory(
      #     input_dim=12,
      #     num_hist=1,
      #     mean=np.zeros(1 * 12),
      #     var=np.ones(1 * 12)
      # )

    self.frames_historical_data = VisualHistory(
        frame_shape=(64, 64),
        num_hist=17,
        mean=0 * np.ones(
            64 * 64 * 17
        ),   # FIXME: Mean measured = 1.02
        var=1 * np.ones(
            64 * 64 * 17
        ),    # FIXME: Variance measured = 0.11
        sliding_frames=sliding_frames,
        use_inverse_depth=use_inverse_depth,
        use_original_depth=use_original_depth
    )  # variance, not std!

    if self.twoview:
      self.secview_frames_historical_data = VisualHistory(
          frame_shape=(64, 64),
          num_hist=get_image_interval * 3 + 1,
          mean=1.25 * np.ones(
              64 * 64 * (get_image_interval * 3 + 1)
          ),   # FIXME: Mean measured = 1.02
          var=0.45 ** 2 * np.ones(
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

      # last action
      # last_action = convert_order_from_ros_to_isaac(last_action)
      # last_action = last_action.detach().cpu().numpy()
      last_action_normalized = self.last_action_historical_data.record(
          last_action,
          backwards=True
      ) * self.hist_action_scale

      # proj_start = time.time()
      import torch
      quat = observation.imu.quaternion
      quat = np.concatenate(
          [quat[1:], quat[0:1]]
      )
      proj_g = quat_rotate_inverse(
          torch.tensor(quat).float().unsqueeze(0),
          torch.tensor([0, 0, -1.0]).float().unsqueeze(0)
      ).squeeze().cpu().numpy()
      proj_g_normalized = proj_g * self.obs_scale["proj_g"]
      # print("Proj Time:", time.time() - proj_start)

      # command_normalized = self.com_vel_historical_data.record_and_normalize(
      #     command,
      #     backwards=True
      # ) * self.obs_scale["command"]

      if self.process_count % 2 == 0:
        # append new frame everytime and give index 0,4,8,12
        normalized_visual_history = self.frames_historical_data.record_and_normalize(
            depth_frame, depth_scale, backwards=False)
        self.normalized_visual_history = normalized_visual_history
      else:
        normalized_visual_history = self.normalized_visual_history
      self.process_count += 1
      # append new frame everytime and give index 0,4,8,12

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
          # com_vel_normalized,  # 3
          # base_ang_vel_hist_normalized,  # 3
          joint_angle_hist_normalized,  # 12
          joint_vel_hist_normalized,  # 12 * 2
          proj_g_normalized,  # 3
          # command_normalized,  # 3
          last_action_normalized,  # 12 * 2
      ]
      obs_normalized_np = np.hstack(obs_list)
      # print("State Obs")
      # print(obs_normalized_np)
      # with np.printoptions(precision=4, suppress=True):
      #   print("----------------------------")
      #   print("State Obs")
      #   print(obs_normalized_np)
      #   print("Visual")
      #   print(normalized_visual_history.reshape(-1))
      #   print("Com Vel", com_vel_normalized)
      #   print("Base Ang Vel", base_ang_vel_hist_normalized)
      #   print("Joint Ang", joint_angle_hist_normalized)  # 12
      #   print("Joint Vel", joint_vel_hist_normalized)  # 12 * 2
      #   print("Proj G", proj_g_normalized)  # 3
      #   print("Command", command_normalized)  # 3
      #   print("Action", last_action_normalized)  # 16 * 2
    else:

      obs_list = [
          # com_vel_normalized,  # 3
          # base_ang_vel_hist_normalized,  # 3
          joint_angle_hist_normalized,  # 12
          joint_vel_hist_normalized,  # 12 * 2
          proj_g_normalized,  # 3
          # command_normalized,  # 3
          last_action_normalized,  # 16 * 2
          normalized_visual_history.reshape(-1)
      ]
      obs_normalized_np = np.hstack(obs_list)
      # print(obs_normalized_np)

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
      if self.no_tensor:
        action = action[0]
      else:
        action = action.squeeze().cpu().numpy()
      action_normalized = action
      # right_act_normalized, left_act_normalized = np.split(diagonal_action_normalized, 2)
      # action_normalized = np.concatenate(
      #     [right_act_normalized, left_act_normalized, left_act_normalized, right_act_normalized]
      # )
      # action_normalized = convert_order_from_isaac_to_ros(
      #     action_normalized
      # )
      action_normalized = np.clip(action_normalized, -1, 1)
      action = action_normalized * self.action_range
      # TODO: move above part to GPU and profile

      # action_ub = self.action_ub
      # action_lb = self.action_lb
      # action = 0.5 * (np.tanh(action_normalized) + 1) * (action_ub - action_lb) + action_lb

      # if self.clip_motor:
      #   action = np.clip(
      #       action,
      #       self.current_joint_angle - self.clip_motor_value,
      #       self.current_joint_angle + self.clip_motor_value
      # )
    return action

  def get_action(self, observation, depth_frame, depth_scale, last_action):
    '''
    This function process raw observation, fed normalized observation into
    the network, de-normalize and output the action.
    '''
    ob_t = self.process_obs(observation,
                            depth_frame, depth_scale, last_action)
    if self.no_tensor:
      ob_t = np.clip(ob_t, -5, 5)
    else:
      ob_t = torch.clamp(ob_t, -5, 5)
    if self.no_tensor:
      action = self.pf.act_inference(ob_t)
    else:
      with torch.no_grad():
        print(ob_t.shape)
        action = self.pf.act_inference(ob_t)
    action = self.process_act(action)
    if self.save_log:
      self.policy_action_saver.record(action)
    return action

  def write(self):
    if self.save_log:
      self.ob_tensor_saver.write()
      self.policy_action_saver.write()
