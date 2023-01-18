# Environment specific functions that determine:
# - Terrains
# - TriangleGround
# - Goal positions
# - Randomization of the external environments
# ...

import torch
import trimesh
import numpy as np
from tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper
from isaacgym import gymapi


class TriangleGroundEnvWrapper(BaseEnvWrapper):

  def __init__(self, device, cfg, env_name):
    """Initializes the env wrappers."""
    self.env_name = env_name
    super(TriangleGroundEnvWrapper, self).__init__(device, cfg)
    self.ground_type = self.env_cfg["groundType"]["name"]
    self.static_friction = self.env_cfg["groundType"]["staticFriction"]
    self.dynamic_friction = self.env_cfg["groundType"]["dynamicFriction"]
    self.restitution = self.env_cfg["groundType"]["restitution"]

  def check_termination(self, task):
    """Checks if the episode is over."""
    return False

  def create_surroundings(self, task, env_ptr, env_id):
    """Create the surroundings including terrains and obstacles for each environment."""
    return []

  def create_ground(self, task):
    # Please ensure the loaded 3d model only contains triangle faces
    task.trimesh = trimesh.load(
      "../../assets/terrains/ground/{}.obj".format(self.ground_type))
    vertices = np.asarray(task.trimesh.vertices, dtype=np.float32)
    scale = self.env_cfg["groundType"]["scale"]
    offset = self.env_cfg["groundType"]["offset"]
    vertices *= np.asarray(scale)
    faces = np.asarray(task.trimesh.faces, dtype=np.uint32)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = task.trimesh.vertices.shape[0]
    tm_params.nb_triangles = task.trimesh.faces.shape[0]
    tm_params.transform.p.x = offset[0]
    tm_params.transform.p.y = offset[1]
    tm_params.transform.p.z = offset[2]
    task.gym.add_triangle_mesh(task.sim, vertices.flatten(
      order='C'), faces.flatten(order='C'), tm_params)
    return self.sample_origins(vertices + np.asarray(offset))

  def sample_origins(self, vertices):
    self.origins = np.zeros((self.num_envs, 3))
    self.num_per_row = int(np.sqrt(self.num_envs))
    if self.num_per_row == 1:
      self.num_per_row += 1
    for k in range(self.num_envs):
      pos = (k % self.num_per_row,
             k // self.num_per_row)
      pos = np.asarray(pos) * self.env_size
      vertices_xy = vertices[..., :2]
      closest_indices = np.argsort(np.linalg.norm(
        pos - vertices_xy, axis=-1), axis=0)[:15]
      closest_index = np.argmax(vertices[closest_indices, -1], axis=-1)
      sampled_pos = vertices[closest_indices[closest_index]]
      sampled_pos[:2] -= pos
      self.origins[k] = sampled_pos
    self.origins = torch.as_tensor(self.origins, device=self.device)
    return self.origins
