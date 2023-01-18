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
from tasks.terrain import Terrain
import os


class TerrainEnvWrapper(BaseEnvWrapper):

  def __init__(self, device, cfg, env_name):
    """Initializes the env wrappers."""
    self.env_name = env_name
    super(TerrainEnvWrapper, self).__init__(device, cfg)
    self.ground_type = self.env_cfg["groundType"]["name"]
    self.static_friction = self.env_cfg["groundType"]["staticFriction"]
    self.dynamic_friction = self.env_cfg["groundType"]["dynamicFriction"]
    self.restitution = self.env_cfg["groundType"]["restitution"]

  def check_termination(self, task):
    """Checks if the episode is over."""
    # (torch.norm(task.contact_forces[:, task.termination_contact_indices, :], dim=-1) > 0.1)
    flag = torch.any((torch.norm(
        task.contact_forces[:, task.termination_contact_indices, :], dim=-1) > 0.1), dim=1)

    # Check Position

    base_pos = task.root_states[task.a1_indices, 0:2]
    pos_flag = ~torch.all(
        torch.logical_and(
            base_pos > task.robot_origin[task.a1_indices, 0:2] - self.offset,
            base_pos < task.robot_origin[task.a1_indices, 0:2] + self.offset
        ), dim=-1
    )
    return flag | pos_flag

  def create_surroundings(self, task):
    """Create the surroundings including terrains and obstacles for each environment."""
    handles = []
    return handles

  def create_ground(self, task):
    pass

  def sample_origins(self, vertices=None):
    pass
