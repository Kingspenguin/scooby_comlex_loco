# Environment specific functions that determine:
# - Terrains
# - Barriers
# - Goal positions
# - Randomization of the external environments
# ...

import torch
import trimesh
import numpy as np
from tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper
from isaacgym import gymapi


class PlaneEnvWrapper(BaseEnvWrapper):

  def __init__(self, device, cfg, env_name):
    """Initializes the env wrappers."""
    self.env_name = env_name
    super(PlaneEnvWrapper, self).__init__(device, cfg)
    self.ground_type = self.env_cfg["groundType"]["name"]
    self.static_friction = self.env_cfg["groundType"]["staticFriction"]
    self.dynamic_friction = self.env_cfg["groundType"]["dynamicFriction"]
    self.restitution = self.env_cfg["groundType"]["restitution"]
    return

  def check_termination(self, task):
    """Checks if the episode is over."""
    base_pos = task.root_states[task.a1_indices, 0:2]
    flag = ~torch.all(torch.logical_and(base_pos > -self.offset, base_pos <
                                        self.offset), dim=-1)
    return flag

  def create_surroundings(self, task, env_ptr, env_id):
    """Create the surroundings including terrains and obstacles for each environment."""
    return []

  def create_ground(self, task):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = self.static_friction
    plane_params.dynamic_friction = self.dynamic_friction
    task.gym.add_ground(task.sim, plane_params)
    return torch.zeros((self.num_envs, 3), device=self.device)

  def sample_origins(self, vertices):
    del vertices
    raise NotImplementedError
