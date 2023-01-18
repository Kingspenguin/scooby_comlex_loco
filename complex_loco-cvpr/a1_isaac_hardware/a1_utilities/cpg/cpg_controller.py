from threading import Thread
import numpy as np
from a1_utilities.a1_sensor_process import convert_order_from_isaac_to_ros, prepare_position_cmd
from a1_utilities.cpg.gait_generator import GaitGenerator
import torch
from torch import Tensor
from a1_utilities.robot_controller import RobotController
from a1_utilities.cpg.A1_inverse_kinematics import multiple_leg_inverse_kinematics_isaac
import time
import math
from a1_utilities.a1_robot import A1Robot
import pybullet
from pybullet_utils import bullet_client

import multiprocessing as mp
from multiprocessing import Manager


def get_cpg_action(action):
  self.residual_joint_angle = action[..., :12].cpu().numpy()
  self.residual_phase = action[..., 12:].cpu().numpy()
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


def _footPositionsToJointAngles(
    foot_positions, dof_pos,
    default_feet_pos, default_dof_pos
):
  # dof_pos = torch.Tensor(dof_pos).to(self.device)
  feet_err = foot_positions - default_feet_pos[np.newaxis, ...]
  leg_jacobian = _compute_leg_jacobian(dof_pos)
  u = np.squeeze(
      (np.linalg.inv(leg_jacobian) @ feet_err[..., np.newaxis]), -1
  )
  pos_target = u.reshape(4, 3) + default_dof_pos
  return pos_target


def _compute_leg_jacobian(dof_pos):
  l1 = 0.0838
  l2 = 0.2
  l3 = 0.2

  # leg_length = torch.tensor([0.08505, 0.2, 0.2]).to(device)
  # Only for A1
  joint_pos = dof_pos.reshape(
      4, 3
  )

  leg_jacobian = np.zeros(
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

  leg_jacobian[..., 0, 0] = 0
  leg_jacobian[..., 1, 0] = -side_sign * \
      l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
  leg_jacobian[..., 2, 0] = side_sign * \
      l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
  leg_jacobian[..., 0, 1] = -l3 * c23 - l2 * c2
  leg_jacobian[..., 1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
  leg_jacobian[..., 2, 1] = l2 * s2 * c1 + l3 * s23 * c1
  leg_jacobian[..., 0, 2] = -l3 * c23
  leg_jacobian[..., 1, 2] = -l3 * s23 * s1
  leg_jacobian[..., 2, 2] = l3 * s23 * c1
  return leg_jacobian


def _gen_parabola(
    phase, start, mid, end
):
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


def _gen_swing_foot_trajectory(
    input_phase, start_pos, end_pos
):
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
  z = _gen_parabola(phase, start_pos[..., 2], mid, end_pos[..., 2])
  return np.stack([x, y, z], axis=-1)


class CPGController(RobotController):
  def __init__(
      self,
      control_freq=400,
      default_action=None,
      save_log=False,
      log_interval=10,
      state_save_path=None,
      # CPG settings
      stance_duration=0.2,
      foot_clearance=0.025,
      Kp=60, Kd=0.8
  ):
    super().__init__(
        control_freq=control_freq,
        default_action=default_action,
        use_high_level_command=False,
        save_log=save_log,
        log_interval=log_interval,
        state_save_path=state_save_path,
    )
    self.Kp = Kp
    self.Kd = Kd

    self.stance_duration = stance_duration
    self.foot_clearance = foot_clearance

    self.manager = Manager()
    self.shared_dict = self.manager.dict()
    self.cpg_start_condition = self.manager.Condition()
    self.cpg_end_condition = self.manager.Condition()

  def pass_cpg_command_thread(
      self,
  ):
    print("Pass Start")
    self.cpg_start_condition.acquire()
    print("Acquire")
    self.cpg_end_condition.acquire()
    print("Acquire")
    self.cpg_start_condition.notify_all()
    self.cpg_start_condition.release()
    self.cpg_end_condition.wait()
    self.set_action(
        (self.shared_dict["tart_joint_pos"], self.Kp, self.Kd)
    )
    print("Pass End")

  def set_cpg_action(
      self, action, phase_time, current_joint_angle
  ):
    # scaled_action = self.action_scale * action.flatten()
    self.shared_dict["phase_time"] = phase_time
    self.shared_dict["action"] = action
    self.shared_dict["current_joint_angle"] = current_joint_angle
    Thread(target=self.pass_cpg_command_thread).start()

  def start_thread(self):
    super().start_thread()
    self.shared_dict["cpg_process_running"] = True
    self.cpg_process = mp.Process(
        target=self.cpg_process_function,
        args=(
            self.stance_duration,
            self.foot_clearance,
            self.shared_dict, self.cpg_start_condition, self.cpg_end_condition
        )
    )
    self.cpg_process.start()

  def stop_thread(self):
    self.shared_dict["cpg_process_running"] = False
    self.cpg_start_condition.notify()
    self.cpg_process.join()
    super().stop_thread()

  @staticmethod
  def cpg_process_function(
      stance_duration, foot_clearance,
      shared_dict, start_condition, end_condition
  ):
    gait_generator = GaitGenerator(
        stance_duration=stance_duration
    )

    end_feet_pos = np.array(
        [0.05, 0.0, foot_clearance] * 4,
    )

    # from minghao's validation data
    default_feet_pos = np.array([
        [0.1478, -0.11459, -0.45576],
        [0.1478, 0.11688, -0.45576],
        [-0.2895, -0.11459, -0.45576],
        [-0.2895, 0.11688, -0.45576]
    ])

    foot_target_position = end_feet_pos.reshape(4, 3) + default_feet_pos

    default_dof_pos = np.array([
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7],
        [0.0, 0.9, -1.7]
    ])

    # FIXME: need to see what is this nominal residue phase
    nominal_residual_phase = 1.0 * np.array([-0.2, 0.2, 0.2, -0.2])

    while shared_dict["cpg_process_running"]:
      # single transcation
      start_condition.acquire()
      start_condition.wait()
      phase_time = shared_dict["phase_time"]
      action = shared_dict["action"]
      current_joint_angle = shared_dict["current_joint_angle"]

      residual_joint_angle = action[..., :12]
      residual_phase = action[..., 12:]

      gait_generator.update(
          phase_time,
          residual_phase + nominal_residual_phase
      )
      # 2. calculate swing_feet_pos
      #    based on "gait_generator", "default_feet_pos", "end_feet_pos"
      swing_feet_pos = _gen_swing_foot_trajectory(
          gait_generator._normalized_phase,
          default_feet_pos,
          foot_target_position
      )

      # 3. select from "swing_feet_pos" or "stance_feet_pos" based on gait generator
      stance_feet_pos = default_feet_pos
      targets_feet_pos = (
          gait_generator.desired_leg_state[..., np.newaxis] != 0
      ).astype(np.float) * stance_feet_pos + (
          gait_generator.desired_leg_state[..., np.newaxis] == 0
      ).astype(np.float) * swing_feet_pos
      target_joint_pos = _footPositionsToJointAngles(
          targets_feet_pos, current_joint_angle,
          default_feet_pos, default_dof_pos
      )
      target_joint_pos = target_joint_pos.reshape(-1) + residual_joint_angle
      target_joint_pos = convert_order_from_isaac_to_ros(target_joint_pos)
      shared_dict["tart_joint_pos"] = target_joint_pos
      end_condition.acquire()
      end_condition.notify()
      end_condition.release()