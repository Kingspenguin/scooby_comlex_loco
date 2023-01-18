from tasks.env_wrappers.base_env_wrapper import BaseEnvWrapper
from tasks.env_wrappers.state.plane import PlaneEnvWrapper
from tasks.env_wrappers.state.upstairs import UpstairsEnvWrapper
from tasks.env_wrappers.state.downstairs import DownstairsEnvWrapper
from tasks.env_wrappers.state.mountain_range import MountainRangeEnvWrapper
from tasks.env_wrappers.vision.terrain import TerrainEnvWrapper
from tasks.env_wrappers.state.box_ground import BoxGroundEnvWrapper
from tasks.env_wrappers.state.triangle_ground import TriangleGroundEnvWrapper

state_env_wrappers = {
    "base": BaseEnvWrapper,
    "plane": PlaneEnvWrapper,
    "upstairs": UpstairsEnvWrapper,
    "downstairs": DownstairsEnvWrapper,
    "mountain_range": MountainRangeEnvWrapper,
    "box_ground": BoxGroundEnvWrapper,
    "triangle_ground": TriangleGroundEnvWrapper
}

vision_env_wrappers = {
    "terrain": TerrainEnvWrapper,
    "terrain_update_2": TerrainEnvWrapper,
    "terrain_update_2_multi": TerrainEnvWrapper,
    "terrain_update_2_multi_wide": TerrainEnvWrapper,
    "terrain_vision": TerrainEnvWrapper,
    "terrain_vision_1cam": TerrainEnvWrapper,
    "terrain_vision_1cam_16": TerrainEnvWrapper,
    "terrain_vision_1cam_13_4": TerrainEnvWrapper,
    "terrain_vision_1cam_13_4_stacked": TerrainEnvWrapper,
    "terrain_vision_1cam_17_4": TerrainEnvWrapper,
    "terrain_vision_1cam_17_4_update_2": TerrainEnvWrapper,
    "terrain_vision_1cam_17_4_update_2_10": TerrainEnvWrapper,
    "terrain_vis1c_17_4_up2_multi": TerrainEnvWrapper,
    "terrain_vis1c_17_4_up2_multi_wide": TerrainEnvWrapper,
    "terrain_vision_1cam_17_4_stacked": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_10": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_20": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_stacked": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_stacked_10": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_stacked_20": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_update_2": TerrainEnvWrapper,
    "terrain_vision_1cam_21_4_update_2_10": TerrainEnvWrapper,
    "terrain_vis1c_17_4_up2_multi": TerrainEnvWrapper,
    "terrain_vis1c_17_4_up2_multi_10": TerrainEnvWrapper,
    "terrain_vis1c_17_4_up2_multi_wide": TerrainEnvWrapper,
    "terrain_vis1c_21_4_up2_multi": TerrainEnvWrapper,
    "terrain_vis1c_21_4_up2_multi_10": TerrainEnvWrapper,
    "terrain_vis1c_21_4_up2_multi_wide": TerrainEnvWrapper,
    "terrain_vis1c_9_4_up2_multi_10": TerrainEnvWrapper,
    "terrain_vis1c_33_4_up2_multi_10": TerrainEnvWrapper,
    "terrain_vision_1cam_25_4": TerrainEnvWrapper,
    "terrain_vision_1cam_25_4_update_2": TerrainEnvWrapper,
    "terrain_vision_2cam_16": TerrainEnvWrapper,
    "terrain_vision_1cam_only": TerrainEnvWrapper,
    "terrain_vision_recurrent": TerrainEnvWrapper,
    "terrain_vision_recurrent_1cam": TerrainEnvWrapper
}

all_env_wrappers = state_env_wrappers.copy()
all_env_wrappers.update(vision_env_wrappers)


def build_env_wrapper(name, device, cfg):
  return all_env_wrappers[name](device, cfg, env_name=name)
