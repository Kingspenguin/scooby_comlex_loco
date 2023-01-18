import torch

# constants:
# nominal phase: +- 0.1


# reference: https://www.ijstr.org/final-print/sep2017/Inverse-Kinematic-Analysis-Of-A-Quadruped-Robot.pdf

@torch.jit.script
def leg_inverse_kinematics(target_pos, leg_length, is_left_side):
  # type: (Tensor, Tensor, Tensor) -> Tensor
  # transform position command (target_pos, N-by-3) into 3 dim target angle
  #    HEAD       Command coordinate
  # FL ---- FR         Z
  # |        |         ^
  # |        |         |
  # |        |         |
  # RL ---- RR         Y(down)--> X

  # ensure that target_pos[1] > leg_length[0]
  target_pos[:, 1] = torch.where(
      torch.abs(target_pos[:, 1]) < 1.2 * leg_length[0],
      1.2 * leg_length[0] * torch.ones_like(target_pos[:, 1]),
      target_pos[:, 1]
  )

  # normalize target pos if it is too large. After this normalization, IK is guaranteed to get a solution.
  maximum_goal_norm = torch.sqrt(torch.sum(torch.square(
      leg_length)) + 2 * leg_length[1] * leg_length[2]) - 1e-4  # numerical stability

  target_pos = torch.where(torch.norm(target_pos, dim=1).unsqueeze(1) > maximum_goal_norm,
                           target_pos * maximum_goal_norm / torch.norm(target_pos, dim=1).unsqueeze(1), target_pos)

  x = target_pos[:, 0:1]
  y = target_pos[:, 1:2]
  z = target_pos[:, 2:3]

  D = (torch.sum(torch.square(target_pos), dim=1).unsqueeze(
      1) - torch.sum(torch.square(leg_length))) / (2 * leg_length[1] * leg_length[2])

  thetas = torch.zeros_like(target_pos)
  thetas[:, 2:3] = torch.atan2(- torch.sqrt(1 - torch.square(D)), D)

  inter_result = torch.sqrt(torch.square(
      x) + torch.square(y) - torch.square(leg_length[0]))
  thetas[:, 1:2] = torch.atan2(z, inter_result) - torch.atan2(leg_length[2] * torch.sin(
      thetas[:, 2:3]), leg_length[1] + leg_length[2] * torch.cos(thetas[:, 2:3]))

  thetas[:, 0:1] = torch.where(is_left_side,
                               -torch.atan2(-y, x) -
                               torch.atan2(inter_result, -leg_length[0]),
                               -torch.atan2(-y, x) -
                               torch.atan2(inter_result, leg_length[0])
                               )

  #is_successful = ((1 - torch.square(D)) >= 0) & ((torch.square(x) + torch.square(y) - torch.square(leg_length[0])) >= 0)
  return thetas


@torch.jit.script
def multiple_leg_inverse_kinematics(target_pos, leg_length):
  # type: (Tensor, Tensor) -> Tensor
  # transform position command (target_pos, N-by-12)
  # position command order: [FL, FR, RL, RR]
  # into 12 dim target angle following the same order above.
  #
  #    HEAD       Command coordinate
  # FL ---- FR         Z
  # |        |         ^
  # |        |         |
  # |        |         |
  # RL ---- RR         Y(down)--> X

  target_angle = torch.zeros_like(target_pos)

  is_left = torch.ones_like(target_pos, dtype=torch.bool)[:, 0:1]
  is_right = torch.zeros_like(target_pos, dtype=torch.bool)[:, 0:1]

  target_angle[:, 0:3] = leg_inverse_kinematics(
      target_pos[:, 0:3], leg_length, is_left)  # FL
  target_angle[:, 3:6] = leg_inverse_kinematics(
      target_pos[:, 3:6], leg_length, is_right)  # FR
  target_angle[:, 6:9] = leg_inverse_kinematics(
      target_pos[:, 6:9], leg_length, is_left)  # RL
  target_angle[:, 9:12] = leg_inverse_kinematics(
      target_pos[:, 9:12], leg_length, is_right)  # RR

  return target_angle


def multiple_leg_inverse_kinematics_isaac(target_pos_isaac, leg_length, offset=None):
  # type: (Tensor, Tensor, Tensor) -> Tensor
  # transform position command (target_pos, 1-by-12)
  # position command order: [FL, FR, RL, RR]
  # same as above only with coordinate transformation
  #
  #    HEAD       Command coordinate
  # FL ---- FR         X
  # |        |         ^
  # |        |         |
  # |        |         |
  # RL ---- RR         Z(up)---> Y
  if not offset:
    offset = torch.tensor([
        [-0.1346, 0.0295, 0.1921],
        [-0.1346, -0.0318, 0.1921],
        [0.3027, 0.0295, 0.1921],
        [0.3027, -0.0318, 0.1921]
    ], device=target_pos_isaac.device)

  target_pos_isaac = target_pos_isaac + offset

  target_pos_rot = torch.zeros_like(
      target_pos_isaac, device=target_pos_isaac.device
  ).reshape((4, 3))

  target_pos_rot[:, 0] = target_pos_isaac[:, 1]
  target_pos_rot[:, 1] = -target_pos_isaac[:, 2]
  target_pos_rot[:, 2] = target_pos_isaac[:, 0]

  return multiple_leg_inverse_kinematics(target_pos_rot.reshape((1, 12)), leg_length)


def normalized_action_to_joint_target(normalized_action, device):

  leg_length = torch.tensor([0.08505, 0.2, 0.2]).to(device)

  lower_pos = torch.tensor([[-0.3, 0.15, -0.3]]).repeat(1, 4).to(device)
  upper_pos = torch.tensor([[0.3, 0.35, 0.3]]).repeat(1, 4).to(device)

  leg_compensation = torch.tensor([
      -leg_length[0], 0.0, 0.0,
      leg_length[0], 0.0, 0.0,
      -leg_length[0], 0.0, 0.0,
      leg_length[0], 0.0, 0.0,
  ])

  target_pos = (lower_pos + upper_pos) / 2.0 + \
      normalized_action * (upper_pos - lower_pos) / 2.0
  target_pos += leg_compensation

  return multiple_leg_inverse_kinematics(target_pos, leg_length)
