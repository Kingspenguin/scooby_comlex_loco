# Environment specific functions that determine:
# - Terrains
# - Barriers
# - Goal positions
# - Randomization of the external environments
# ...

import torch
import numpy as np
from tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper
from isaacgym import gymapi


class BoxGroundEnvWrapper(BaseEnvWrapper):

  def __init__(self, device, cfg, env_name):
    """Initializes the env wrappers."""
    self.env_name = env_name
    super(BoxGroundEnvWrapper, self).__init__(device, cfg)
    self.ground_type = self.env_cfg["groundType"]["name"]
    self.static_friction = self.env_cfg["groundType"]["staticFriction"]
    self.dynamic_friction = self.env_cfg["groundType"]["dynamicFriction"]
    self.restitution = self.env_cfg["groundType"]["restitution"]
    self.goal_position = torch.as_tensor(
      [self.env_cfg["goal_position"]], device=self.device)

  def check_termination(self, task):
    """Checks if the episode is over."""
    base_xy_pos = task.root_states[task.a1_indices, 0:2]
    base_pos = task.root_states[task.a1_indices, 0:3]
    flag = ~torch.all(torch.logical_and(base_xy_pos > -self.offset, base_xy_pos <
                                        self.offset), dim=-1)
    flag |= torch.norm(base_pos - self.goal_position, dim=-1) < 0.05
    return flag

  def create_surroundings(self, task, env_ptr, env_id):
    """Create the surroundings including terrains and obstacles for each environment."""
    handles = []
    i = 0
    assert len(self.surrounding_assets) > 0
    for surrounding_assets, surrounding_cfg in zip(self.surrounding_assets, self.env_cfg["surroundings"].values()):
      pose = gymapi.Transform()
      pose.p.x = surrounding_cfg["surrounding_origin"][0]
      pose.p.y = surrounding_cfg["surrounding_origin"][1]
      pose.p.z = surrounding_cfg["surrounding_origin"][2]

      handle = task.gym.create_actor(
        env_ptr, surrounding_assets, pose, "sr_{}".format(i), env_id, 2 + i, 0)
      if surrounding_cfg["texture"] != "none":
        th = task.gym.create_texture_from_file(
          task.sim, "../../../assets/textures/{}".format(surrounding_cfg["texture"]))
        task.gym.set_rigid_body_texture(
          env_ptr, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, th)
      handles.append(handle)
      i += 1
    return handles

  def create_ground(self, task):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = self.static_friction
    plane_params.dynamic_friction = self.dynamic_friction
    task.gym.add_ground(task.sim, plane_params)
    return self.sample_origins()

  def sample_origins(self, vertices=None):
    del vertices
    robot_origin = self.env_cfg["robot_origin"]
    num_choices = len(robot_origin)
    assert num_choices > 1

    # Sample the robot origin
    # This is a hack. We should sample the robot origin based on two cases
    # 1. The robot is facing to the stairs and on the ground
    # 2. The robot is on the stairs

    prob_ground = self.env_cfg["prob_ground"]
    prob_in_process = (1 - prob_ground) / (num_choices - 1)
    sampled_origins = np.random.choice(list(range(num_choices)), size=self.num_envs, p=[
                                       prob_ground] + [prob_in_process] * (num_choices - 1)).astype(int)
    sampled_origins = np.asarray(robot_origin)[sampled_origins]
    sampled_origins = torch.as_tensor(sampled_origins, device=self.device)
    return sampled_origins
