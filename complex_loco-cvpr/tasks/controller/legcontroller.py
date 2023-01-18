from isaacgym import gymtorch
from tasks.controller.gait_generator import GaitGenerator
from tasks.controller.com_velocity_estimator import COMVelocityEstimator
import torch
import math
from torch import Tensor
from copy import deepcopy
# We have the following mode to handle the control command:
# pd_joint: Normal pd control for legs. Input: target position for each joint. Output: Torque for each joint.
# cpg: pd control for legs with diagonal actions. Input: target position for each joint. Output: Torque for each joint.
# pd_foot: pd control for foot
MODE_DICT = ["pd_joint", "cpg", "pd_foot", "ik_foot"]
# DEFAULT_FEET_POS = [
#     0.1602, 0.0356, -0.2584,
#     0.1583, -0.0397, -0.2586,
#     -0.1809, 0.0364, -0.2563,
#     -0.1803, -0.0405, -0.2565]
DEFAULT_FEET_POS = [
    0.1583, 0.0397, -0.2584,
    0.1583, -0.0397, -0.2584,
    -0.1803, 0.0405, -0.2565,
    -0.1803, -0.0405, -0.2565]
# DEFAULT_FEET_POS = [
#     -0.0132, 0.0851, -0.2637,
#     -0.0132, -0.0851, -0.2637,
#     -0.0132, 0.0851, -0.2637,
#     -0.0132, -0.0851, -0.2637
# ]


class LegController():
  # The leg controller for quadrupedal robots.
  def __init__(self, device, num_envs, cfg=None):
    self.device = device
    self.num_envs = num_envs
    self.cfg = cfg
    self.mode = cfg["legcontroller"]
    assert self.mode in MODE_DICT, f"Unknown leg controller mode: {self.mode}"
    self.execute_control = getattr(self, '_controller_' + self.mode)
    self.init_params()

  def reset(self, reset_idx):
    if self.mode in ["cpg", "pd_foot"]:
      self.gait_generator.reset(reset_idx)
      # self.velocity_estimator.reset(reset_idx)

  def init_params(self):
    # if self.mode == "pd_foot":
    #   self.foot_p_gains = torch.as_tensor(
    #     [self.cfg["stiffness"]] * 3, device=self.device, dtype=torch.float)
    #   self.foot_d_gains = torch.as_tensor(
    #     [self.cfg["damping"]] * 3, device=self.device, dtype=torch.float)
    #   self.foot_p_gains = torch.diag(self.foot_p_gains).unsqueeze(
    #     0).unsqueeze(0).repeat(1, 4, 1, 1)
    #   self.foot_d_gains = torch.diag(self.foot_d_gains).unsqueeze(
    #     0).unsqueeze(0).repeat(1, 4, 1, 1)
    #   self.joint_d_gains = 5
    if self.mode in ["cpg", "pd_foot"]:
      self.gait_generator = GaitGenerator(
          num_envs=self.num_envs, device=self.device, stance_duration=self.cfg["stance_duration"])
      # self.velocity_estimator = COMVelocityEstimator(
      #   num_envs=self.num_envs, device=self.device
      # )
      self.forward_offset = self.cfg.get("forward_offset", 0.0)
      # self.forward_offset = 0.0
      self.end_feet_pos = torch.as_tensor([
          self.forward_offset, 0, self.cfg["foot_clearance"],
          self.forward_offset, 0, self.cfg["foot_clearance"],
          self.forward_offset, 0, self.cfg["foot_clearance"],
          self.forward_offset, 0, self.cfg["foot_clearance"]
      ], device=self.device)

      self.phase_info = self.gait_generator.phase_info
      self.time_phase_info = self.gait_generator.time_phase_info

  def _inner_set_dof(self, task, targets_pos, last_targets_pos=None):
    if not task.headless:
      for i in range(task.control_freq_inv):
        if last_targets_pos is not None:
          action = _process_action(
              targets_pos, last_targets_pos, i, task.control_freq_inv)
        else:
          action = targets_pos
        # print("Target Joint Angle:", action)
        # print("------------------------------------------------")
        # print("Raw Torque:", (action - task.dof_pos) * 60)
        # torque = (action - task.dof_pos) * 60 - 0.8 * task.dof_vel
        # clipped_toruqe = torch.sign(torque) * torch.clamp(
        #     torch.abs(torque), 0, 33.0
        # )
        # clipped_action = (clipped_toruqe + 0.8 *
        #                   task.dof_vel) / 60 + task.dof_pos
        # print("Raw Act:", action)
        # print("Clipped Torque:", clipped_toruqe)
        # print("Clipped Act:", clipped_action)
        # task.gym.set_dof_position_target_tensor(
        #     task.sim, gymtorch.unwrap_tensor(clipped_action))
        task.gym.set_dof_position_target_tensor(
            task.sim, gymtorch.unwrap_tensor(action))
        task.gym.simulate(task.sim)
        if i % 2 == 0:
          # From experience
          task.render()
    else:
      for i in range(task.control_freq_inv):
        if last_targets_pos is not None:
          action = _process_action(
              targets_pos, last_targets_pos, i, task.control_freq_inv)
        else:
          action = targets_pos
        task.gym.set_dof_position_target_tensor(
            task.sim, gymtorch.unwrap_tensor(action))
        task.gym.simulate(task.sim)
      task.render()

  def _inner_set_torque(self, task, torques, render=False):
    task.gym.set_dof_actuation_force_tensor(
        task.sim, gymtorch.unwrap_tensor(torques))
    task.gym.simulate(task.sim)
    if render:
      task.render()

  def _controller_pd_joint(self, task):
    targets_pos = task.action_scale * task.actions + task.default_dof_pos
    last_target_pos = task.action_scale * task.last_actions + task.default_dof_pos
    computed_torques = (targets_pos - task.dof_pos) * \
        task.Kp - (task.dof_vel) * task.Kd
    # print("Compute Torque:", computed_torque)
    # print(task.torques / computed_torque)
    computed_torques = torch.clamp(computed_torques, -33.5, 33.5)
    task.computed_torques = computed_torques
    self._inner_set_dof(task, targets_pos)
    # self._inner_set_dof(task, targets_pos, last_target_pos)

  def _controller_cpg(self, task):
    # CPG (central pattern generator) with residual jonit positions and phases.
    # See https://arxiv.org/pdf/2201.08117.pdf, Section S5.
    # actions has 16 dims, the first 12 dims is residual joint positions, the last 4 dims is residual phases.
    actions = task.action_scale * task.actions
    residual_joint_pos, residual_phase = actions[..., :12], (
        actions[..., 12:] + task.nominal_residual_phase)

    # print("---------------------------------")
    # # print("res joint", residual_joint_pos)
    # # print("res phase", residual_phase)

    # residual_joint_pos = torch.zeros_like(
    #     actions[..., :12], device=actions.device)
    # residual_phase = torch.zeros_like(actions[..., 12:], device=actions.device)
    # hip_offset = task._hip_offset
    # twisting_vector = torch.stack(
    #   [-hip_offset[..., 1], hip_offset[..., 0], torch.zeros_like(hip_offset[..., 0])], dim=-1)
    # twisting_vector = twisting_vector.unsqueeze(0).repeat(task.num_envs, 1, 1)
    # com_velocity = deepcopy(self.velocity_estimator.com_velocity_body_frame)
    # com_velocity[..., 2] = 0
    # yaw_dot = task.root_states[task.a1_indices, 12:13]
    # # print(com_velocity.shape, yaw_dot.shape, twisting_vector.shape)
    # hip_horizontal_velocity = com_velocity.unsqueeze(
    #   1) + yaw_dot.unsqueeze(1) * twisting_vector
    cur_time = task.progress_buf * task.dt
    # # self.velocity_estimator.update(task.root_states[task.a1_indices])
    self.gait_generator.update(cur_time, residual_phase)
    default_feet_pos = task.default_feet_pos.reshape(4, 3)
    # # end_feet_pos = self.end_feet_pos.reshape(4, 3) + default_feet_pos
    # foot_target_position = (
    #   hip_horizontal_velocity *
    #   self.gait_generator.stance_duration.unsqueeze(
    #     -1) / 2 - 0.01 * (hip_horizontal_velocity)
    # )
    # # # print("foot_target_position", foot_target_position)
    # foot_target_position = foot_target_position + \
    #   (self.end_feet_pos.reshape(4, 3) +
    #    default_feet_pos).unsqueeze(0).repeat(task.num_envs, 1, 1)

    foot_target_position = (self.end_feet_pos.reshape(4, 3) +
                            default_feet_pos).unsqueeze(0).repeat(task.num_envs, 1, 1)
    # print("Foot Target Position:", foot_target_position)

    swing_feet_pos = _gen_swing_foot_trajectory(
        self.gait_generator._normalized_phase, default_feet_pos, foot_target_position, max_clearance=self.cfg.get("max_foot_height", 0.1))
    # stance_feet_pos = _gen_swing_foot_trajectory(
    #   self.gait_generator._normalized_phase, foot_target_position, default_feet_pos)
    stance_feet_pos = default_feet_pos.unsqueeze(0).repeat(task.num_envs, 1, 1)
    # target_leg_indices = (self._gait_generator.desired_leg_state == 0).float()
    targets_feet_pos = (self.gait_generator.desired_leg_state.unsqueeze(-1) != 0).float(
    ) * stance_feet_pos + (self.gait_generator.desired_leg_state.unsqueeze(-1) == 0).float() * swing_feet_pos
    # print(targets_feet_pos)
    # print("Targets Feet Pos:", targets_feet_pos)
    targets_pos = task._footPositionsToJointAngles(targets_feet_pos)
    # print("Target joint Pos not res:", targets_pos)
    # print("Res Joint Pos:", residual_joint_pos)
    targets_pos = targets_pos.reshape(task.num_envs, -1) + residual_joint_pos
    # targets_pos = torch.zeros_like()
    # targets_pos = task.default_dof_pos
    # print("targets_pos", targets_pos)
    # print(targets_pos)
    # print("Joint Pos:", task.dof_pos)
    # print("Target Pos:", targets_pos)
    # print("Torque:", task.torques)
    computed_torques = (targets_pos - task.dof_pos) * \
        task.Kp - (task.dof_vel) * task.Kd
    # print("Compute Torque:", computed_torque)
    # print(task.torques / computed_torque)
    computed_torques = torch.clamp(computed_torques, -33.5, 33.5)
    task.computed_torques = computed_torques
    # print(computed_torque)
    self._inner_set_dof(task, targets_pos)
    # self._inner_set_torque(task, computed_torque, not task.headless)

  def _get_cpg_target_pos(self, action_scale, actions, nominal_residual_phase, task):
    # CPG (central pattern generator) with residual jonit positions and phases.
    # See https://arxiv.org/pdf/2201.08117.pdf, Section S5.
    # actions has 16 dims, the first 12 dims is residual joint positions, the last 4 dims is residual phases.
    actions = action_scale * actions
    residual_joint_pos, residual_phase = actions[..., :12], (
        actions[..., 12:] + nominal_residual_phase)

    cur_time = task.progress_buf * task.dt
    self.gait_generator.update(cur_time.to(self.device), residual_phase)
    default_feet_pos = task.default_feet_pos.reshape(4, 3).to(self.device)

    foot_target_position = (self.end_feet_pos.reshape(4, 3) +
                            default_feet_pos).unsqueeze(0).repeat(task.num_envs, 1, 1)

    swing_feet_pos = _gen_swing_foot_trajectory(
        self.gait_generator._normalized_phase, default_feet_pos, foot_target_position)
    stance_feet_pos = default_feet_pos.unsqueeze(0).repeat(task.num_envs, 1, 1)

    targets_feet_pos = (self.gait_generator.desired_leg_state.unsqueeze(-1) != 0).float(
    ) * stance_feet_pos + (self.gait_generator.desired_leg_state.unsqueeze(-1) == 0).float() * swing_feet_pos
    targets_pos = task._footPositionsToJointAngles(
        targets_feet_pos.to(task.device)).to(self.device)
    targets_pos = targets_pos.reshape(task.num_envs, -1) + residual_joint_pos
    return targets_pos
    # self._inner_set_dof(task, targets_pos)

  def _controller_ik_foot(self, task):
    # v_action = torch.as_tensor([[0, 0, 0.12, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
      #  dtype=torch.float, device=self.device).repeat(task.num_envs, 1)
    # targets_feet_pos = task.default_feet_pos + v_action
    targets_feet_pos = task.action_scale * task.actions + task.default_feet_pos
    targets_feet_pos = targets_feet_pos.reshape(task.num_envs, -1, 3)
    targets_pos = task._footPositionsToJointAngles(targets_feet_pos)
    self._inner_set_dof(task, targets_pos)

  def _controller_pd_foot(self, task):
    actions = task.action_scale * task.actions
    residual_feet_pos, residual_phase = actions[..., :12], (
        actions[..., 12:] + task.nominal_residual_phase)

    cur_time = task.progress_buf * task.dt
    self.gait_generator.update(cur_time, residual_phase)
    default_feet_pos = task.default_feet_pos.reshape(4, 3)
    foot_target_position = (self.end_feet_pos.reshape(4, 3) +
                            default_feet_pos).unsqueeze(0).repeat(task.num_envs, 1, 1)

    swing_feet_pos = _gen_swing_foot_trajectory(
        self.gait_generator._normalized_phase, default_feet_pos, foot_target_position)
    stance_feet_pos = default_feet_pos.unsqueeze(0).repeat(task.num_envs, 1, 1)
    targets_feet_pos = (self.gait_generator.desired_leg_state.unsqueeze(-1) != 0).float(
    ) * stance_feet_pos + (self.gait_generator.desired_leg_state.unsqueeze(-1) == 0).float() * swing_feet_pos
    targets_feet_pos += residual_feet_pos.reshape(task.num_envs, -1, 3)
    targets_pos = task._footPositionsToJointAngles(targets_feet_pos)
    targets_pos = targets_pos.reshape(task.num_envs, -1)
    self._inner_set_dof(task, targets_pos)



# @torch.jit.script
def _gen_parabola(phase: Tensor, start: Tensor, mid: Tensor, end: Tensor) -> Tensor:
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
def _gen_swing_foot_trajectory(input_phase: Tensor, start_pos: Tensor,
                               end_pos: Tensor, max_clearance: float = 0.2) -> Tensor:
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
  mid = torch.maximum(end_pos[..., 2], start_pos[..., 2]) + max_clearance
  z = _gen_parabola(phase, start_pos[..., 2], mid, end_pos[..., 2])

  return torch.stack([x, y, z], dim=-1)


def _process_action(action, last_action, substep_count, action_repeat):
  """If enabled, interpolates between the current and previous actions.

  Args:
    action: current action.
    substep_count: the step count should be between [0, self.__action_repeat).

  Returns:
    If interpolation is enabled, returns interpolated action depending on
    the current action repeat substep.
  """
  lerp = float(substep_count + 1) / action_repeat
  proc_action = last_action + \
      lerp * (action - last_action)
  return proc_action
