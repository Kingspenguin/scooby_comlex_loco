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

# copied from legged_controller.py
DEFAULT_FEET_POS = [0.1602, 0.0356, -0.2584, 0.1583, -0.0397, -0.2586, -0.1809, 0.0364,
                    -0.2563, -0.1803, -0.0405, -0.2565]

# deduce the coordinate axis is:
#    HEAD       default feet pos coordinate
# FL ---- FR         X
# |        |         ^
# |        |         |
# |        |         |
# RL ---- RR         Z(up)--> Y
# in the order of FL FR RL RR


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

    self.device = "cpu"
    self.gait_generator = GaitGenerator(
        self.device, stance_duration=stance_duration)
    self.foot_clearance = foot_clearance

    self.end_feet_pos = torch.as_tensor(
        [0.0, 0.0, -self.foot_clearance] * 4,
        device=self.device
    )

    # from minghao's validation data
    self.default_feet_pos = torch.tensor([
        [0.1478, -0.11459, -0.45576],
        [0.1478, 0.11688, -0.45576],
        [-0.2895, -0.11459, -0.45576],
        [-0.2895, 0.11688, -0.45576]
    ], device=self.device)

    # FIXME: need to see what is this nominal residue phase
    self.nominal_residual_phase = 1.0 * torch.as_tensor(
        [-0.1, 0.1, 0.1, -0.1], device=self.device)

    # order: FL, FR, RL, RR (isaac order).
    self.residue_phase = torch.zeros((4,), device=self.device)
    self.residue_joint_angle = torch.zeros((12,), device=self.device)

    self.Kp = Kp
    self.Kd = Kd

    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    self.robot = A1Robot(
        pybullet_client=p, obs_interval=self.control_interval
    )

    self.cpg_mode = False

  def prepare_command(self):
    if not self.cpg_mode:
      return
    t = time.time() - self.start_time
    # calculate joint position command from stored cpg command
    # 1. update gait generator to generate phase
    self.gait_generator.update(
        t, self.residue_phase.to(self.device) + self.nominal_residual_phase)
    # 2. calculate swing_feet_pos
    #    based on "gait_generator", "default_feet_pos", "end_feet_pos"
    default_feet_pos = self.default_feet_pos.reshape(4, 3)
    foot_target_position = (self.end_feet_pos.reshape(4, 3) + default_feet_pos)
    swing_feet_pos = self._gen_swing_foot_trajectory(
        self.gait_generator._normalized_phase, default_feet_pos, foot_target_position)

    # 3. select from "swing_feet_pos" or "stance_feet_pos" based on gait generator
    stance_feet_pos = default_feet_pos

    targets_feet_pos = (
        self.gait_generator.desired_leg_state.unsqueeze(-1) != 0
    ).float() * stance_feet_pos + (
        self.gait_generator.desired_leg_state.unsqueeze(-1) == 0
    ).float() * swing_feet_pos
    # 4. use Inverse Kinematics to calculate the desired joint position
    ik_joint_pos = multiple_leg_inverse_kinematics_isaac(
        targets_feet_pos, leg_length=torch.tensor(
          [0.08505, 0.2, 0.2], device=self.device)
    )[0]
    # 5. add residue joint position command to calculated joint position
    target_joint_pos = ik_joint_pos + self.residue_joint_angle
    target_joint_pos = target_joint_pos.detach().cpu().numpy()
    target_joint_pos = convert_order_from_isaac_to_ros(target_joint_pos)
    # 6. set self.action accordingly
    self.set_action(prepare_position_cmd(
        target_joint_pos, Kp=self.Kp, Kd=self.Kd))

  def set_cpg_action(self, action):
    # scaled_action = self.action_scale * action.flatten()
    self.residue_joint_angle = action[..., :12].to(self.device)
    self.residue_phase = action[..., 12:]

  def set_cpg_mode(self, mode):
    if mode:
      self.start_time = time.time()
      self.control_step = 0
    self.cpg_mode = mode
  # @torch.jit.script

  def _gen_parabola(self, phase: Tensor, start: Tensor, mid: Tensor, end: Tensor) -> Tensor:
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

  def _gen_swing_foot_trajectory(self, input_phase: Tensor, start_pos: Tensor,
                                 end_pos: Tensor) -> Tensor:
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
    input_phase = torch.clamp(input_phase, 0, 1)
    phase = torch.zeros_like(input_phase, device=input_phase.device)
    indices = torch.nonzero(input_phase <= 0.5, as_tuple=True)
    non_indices = torch.nonzero(input_phase > 0.5, as_tuple=True)
    phase[indices] = 0.8 * torch.sin(input_phase[indices] * math.pi)
    phase[non_indices] = 0.8 + (input_phase[non_indices] - 0.5) * 0.4
    x = (1 - phase) * start_pos[..., 0] + phase * end_pos[..., 0]
    y = (1 - phase) * start_pos[..., 1] + phase * end_pos[..., 1]
    max_clearance = 0.1
    mid = torch.maximum(end_pos[..., 2], start_pos[..., 2]) + max_clearance
    z = self._gen_parabola(phase, start_pos[..., 2], mid, end_pos[..., 2])

    return torch.stack([x, y, z], dim=-1)

  def update_observation(self):
    self.observation = self.robot_interface.receive_observation()
    self.robot.update_state_with_raw_state(self.observation)
    # print("update 1")
    # self.com_vel = self.robot.base_velocity
    self.com_vel = np.zeros(3)
    self.phase_info = self.gait_generator.phase_info.reshape(-1).cpu().numpy()
    # print("update 2")

  def reset_estimator(self, t):
    self.robot.reset(t, self.control_freq)
