from platform import platform
import numpy as np
from tasks.terrain_utils.subterrains import *
from numpy.random import choice
from scipy import interpolate

TerrainList = [pyramid_stairs_terrain, stepping_stones_terrain, corridor_terrain,
               wave_terrain, obstacles_terrain, traversability_terrain, cliffs_terrain, hard_stepping_stones_terrain,
               jumping_obstacles_terrain, jumping_stages_terrain, stepping_barriers_terrain]


class SubTerrain:
  def __init__(self, terrain_name, width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
    self.terrain_name = terrain_name
    self.width = width
    self.length = length
    self.height_field_raw = np.zeros((self.length, self.width), dtype=np.int16)

    self.vertical_scale = vertical_scale
    self.horizontal_scale = horizontal_scale


class Terrain:
  def __init__(self, cfg) -> None:

    # self.type = cfg["mesh_type"]
    # if self.type in ["none", 'plane']:
    #     return
    self.high_res = cfg.get("high_res", False)
    if self.high_res:
      self.horizontal_scale = 0.02  # [m]
    else:
      self.horizontal_scale = 0.05  # [m]
    print(self.horizontal_scale)
    self.vertical_scale = 0.005  # [m]
    self.border_size = 2.5  # [m]
    self.env_length = cfg["map_length"]
    self.env_width = cfg["map_width"]
    self.proportions = cfg["terrain_proportions"]
    if sum(self.proportions) != 1:
      summed_proportions = sum(self.proportions)
      self.proportions = [p / summed_proportions for p in self.proportions]
    self.terrain_depth = cfg.get("depth", -3.0)
    if "perceive_depth" in cfg:
      self.terrain_perceive_depth = cfg["perceive_depth"]
    else:
      self.terrain_perceive_depth = self.terrain_depth
    print(self.terrain_perceive_depth)
    # self.keep_margin =
    self.env_rows = cfg["env_rows"]
    self.env_cols = cfg["env_cols"]
    self.hard_terrain = cfg.get("hard_terrain", False)
    self.specify_terrain = cfg.get("specify_terrain", False)
    self.num_maps = self.env_rows * self.env_cols

    self.total_length = self.env_rows * self.env_length + self.border_size * 2
    self.total_width = self.env_cols * self.env_width + self.border_size * 2

    self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

    self.env_height_low = np.zeros((self.env_rows, self.env_cols))
    self.env_height_high = np.zeros((self.env_rows, self.env_cols))

    self.env_reachable_distance = np.zeros((self.env_rows, self.env_cols, ))

    self.env_random_origin_x_low = np.zeros((self.env_rows, self.env_cols, 1))
    self.env_random_origin_x_range = np.zeros(
        (self.env_rows, self.env_cols, 1))
    self.env_random_origin_y_low = np.zeros((self.env_rows, self.env_cols, 1))
    self.env_random_origin_y_range = np.zeros(
        (self.env_rows, self.env_cols, 1))

    self.width_per_env_pixels = int(
        self.env_width / self.horizontal_scale
    )
    self.length_per_env_pixels = int(
        self.env_length / self.horizontal_scale
    )

    self.border = int(self.border_size / self.horizontal_scale)
    self.tot_cols = int(
        self.env_cols * self.width_per_env_pixels
    ) + 2 * self.border
    self.tot_rows = int(
        self.env_rows * self.length_per_env_pixels
    ) + 2 * self.border

    self.height_field_raw = np.zeros(
        (self.tot_rows, self.tot_cols), dtype=np.int16
    )

    self.percept_height = np.zeros(
        (self.tot_rows, self.tot_cols), dtype=np.int16
    )

    self.randomized_terrain()
    self.slope_treshold = cfg["slope_treshold"]
    self.heightsamples = np.reshape(self.height_field_raw, (-1), order='F')
    self.percept_height_samples = np.reshape(
        self.percept_height, (-1), order='F')

    # if self.type=="trimesh":
    #     self.slope_treshold = cfg["slope_treshold"]
    #     self.convert_to_trimesh()

  def randomized_terrain(self):
    for k in range(self.num_maps):
      # Env coordinates in the world
      (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

      # Heightfield coordinate system from now on
      start_x = self.border + i * self.length_per_env_pixels
      end_x = self.border + (i + 1) * self.length_per_env_pixels
      start_y = self.border + j * self.width_per_env_pixels
      end_y = self.border + (j + 1) * self.width_per_env_pixels

      terrain = SubTerrain(
          "terrain",
          width=self.width_per_env_pixels,
          length=self.length_per_env_pixels,
          vertical_scale=self.vertical_scale,
          horizontal_scale=self.horizontal_scale
      )
      # choice = np.random.uniform(0, 1)
      choice = np.random.choice(len(self.proportions), p=self.proportions)
      platform_size = 0.8
      random_scale = 0.1
      height_low = -4
      height_high = 4
      x_margin = 0.5
      # Only Optimized
      if choice == 0:
        if self.hard_terrain:
          step_length = np.random.choice([0.23, 0.25])
          step_height = np.random.choice([0.15, 0.18, 0.2])
        else:
          step_length = np.random.choice([0.18, 0.23])
          step_height = np.random.choice([0.1, 0.12, 0.15, 0.18])
        height_return, origin = stepping_barriers_terrain(
            terrain, step_length=step_length, step_height=step_height)
      elif choice == 1:
        if self.hard_terrain:
          step_height = np.random.choice([0.15, 0.18, 0.21, 0.23])
        else:
          step_height = np.random.choice([0.1, 0.12, 0.15, 0.18])
        height_return, origin = jumping_obstacles_terrain(
            terrain, step_length=0.5, step_height=step_height)
      elif choice == 2:
        if self.hard_terrain:
          step_height = np.random.choice([0.18, 0.21, 0.25])
          step_length = np.random.choice([0.23, 0.25, 0.3])
        else:
          step_height = np.random.choice([0.18, 0.21, 0.25])
          step_length = 0.2
        height_return, origin = jumping_stages_terrain(
            terrain, step_length=0.2, step_height=step_height)
      elif choice == 3:
        height_return, origin = corridor_terrain(terrain)
      elif choice == 4:
        # if self.hard_terrain:
        stone_distance = np.random.choice(
            # [0.08, 0.12, 0.15], p=[0.5, 0.3, 0.2])
            [0.10, 0.15, 0.2], p=[0.3, 0.35, 0.35]
        )
        # stone_distance = 0.20
        stone_size = np.random.choice([0.8, 1.0, 1.2], p=[0.2, 0.3, 0.5])
        # else:
        #   stone_distance = 0.10
        #   stone_size = 1.2
        # stone_distance = 0.20
        # stone_size = 0.8
        height_low = 0.1
        platform_size = 1.2
        height_return, origin, perceived_height = cliffs_terrain(
            terrain, stone_size=stone_size,
            stone_distance=stone_distance, max_height=0.1, downsampled_scale=0.5,
            depth=self.terrain_depth, platform_size=1.2, percept_depth=self.terrain_perceive_depth
        )
      elif choice == 5:
        stone_distance = np.random.choice([0.05, 0.07])
        stone_size = np.random.choice([0.22, 0.25, 0.3])
        max_height = np.random.choice([0.2, 0.4, 0.5])
        platform_size = 1.6
        height_return, origin = hard_stepping_stones_terrain(
            terrain, stone_size=stone_size, stone_distance=stone_distance, max_height=max_height, platform_size=1.6)
      elif choice == 6:
        if self.specify_terrain:
          stone_distance_x = 0.05
          stone_distance_y = np.random.choice([0.10, 0.15])
          stone_length = np.random.choice([0.2, 0.4])
          # stone_length = 0.2
          stone_width = np.random.choice([0.2, 0.4])
          # stone_width = 0.4
        else:
          if self.hard_terrain:
            stone_distance_x = 0.05
            stone_distance_y = np.random.choice([0.05, 0.10])
            stone_length = np.random.choice([0.2, 0.25, 0.3, 0.35])
            stone_width = np.random.choice([0.2, 0.25, 0.3, 0.35])
          else:
            # If we already have curriculum, seems not necessary to sample x distance
            stone_distance_x = 0.05
            stone_distance_y = np.random.choice([0.05, 0.10])
            stone_length = np.random.choice([0.3, 0.35])
            stone_width = np.random.choice([0.3, 0.35])
        # stone_length = 0.2
        # stone_width = 0.2
        height_low = 0.1
        platform_size = 1.6
        height_return, origin, perceived_height = stepping_stones_terrain(
            terrain, stone_length=stone_length, stone_width=stone_width,
            stone_distance_x=stone_distance_x, stone_distance_y=stone_distance_y,
            max_height=0.5, platform_size=1.6,
            depth=self.terrain_depth, percept_depth=self.terrain_perceive_depth
        )
      elif choice == 7:
        if self.hard_terrain:
          # num_waves = np.random.choice([2, 4, 8, 16])
          # if num_waves == 2:
          #   amplitude = np.random.choice([0.3, 0.35, 0.4])
          num_waves = np.random.choice([4, 8, 16])
          if num_waves == 4 or num_waves == 8:
            amplitude = np.random.choice([.1, .12, .15, .2])
          if num_waves == 16:
            amplitude = np.random.choice([.06, .07, .08])
        else:
          num_waves = np.random.choice([4, 8, 16])
          # amplitude = np.random.choice([.1, .12, .15])
          if num_waves == 4 or num_waves == 8:
            amplitude = np.random.choice([.1, .12, .15, .2])
          if num_waves == 16:
            amplitude = np.random.choice([.06, .07, .08])

        # num_waves = 16
        # amplitude = .08
        height_return, origin, perceived_height = wave_terrain(
            terrain, num_waves=num_waves, amplitude=amplitude)
      elif choice == 8:
        if self.hard_terrain:
          num_rects = np.random.choice([24, 26, 28])
        else:
          num_rects = np.random.choice([16, 20, 24])
        platform_size = 1.0
        height_return, origin = traversability_terrain(
            terrain, max_height=2.2, min_size=0.2,
            max_size=0.5, num_rects=num_rects, platform_size=1.0, random_scale=random_scale
        )
      elif choice == 9:
        # num_rects = np.random.choice([16, 20, 24])
        num_rects = 18
        platform_size = 1.0
        height_return, origin, perceived_height = obstacles_terrain(
            terrain, max_height=2.2, min_size=0.2,
            max_size=0.4, num_rects=num_rects, platform_size=1.0, random_scale=random_scale
        )
      elif choice == 10:
        if self.specify_terrain:
            # step_height = np.random.choice([-0.15, -0.1, 0.1, 0.15])
            # step_width = np.random.choice([0.3, 0.35, 0.4, 0.45])
          step_height = np.random.choice(
              [-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2]
          )
          # step_height = 0.2
          step_width = np.random.choice([0.35, 0.4, 0.5, 0.6, 1.0])
          if step_height == -0.2 or step_height == 0.2:
            step_width = 1.0
        else:
          if self.hard_terrain:
            # step_height = np.random.choice([-0.15, -0.1, 0.1, 0.15])
            # step_width = np.random.choice([0.3, 0.35, 0.4, 0.45])
            # step_height = np.random.choice(
            #     [-0.15, -0.1, -0.05, 0.05, 0.1, 0.15])
            step_height = np.random.choice(
                [-0.15, -0.1, -0.05], p=[0.5, 0.3, 0.2])
            step_width = np.random.choice([0.35, 0.4, 0.5, 0.6, 1.0])
          else:
            step_height = np.random.choice([-0.1, -0.05, 0.05, 0.1])
            step_width = np.random.choice([0.35, 0.4, 0.5, 0.6, 1.0])
            # step_height = -0.1
          # step_width = 0.35

        # step_height = -0.15
        # step_width = 0.6
        platform_size = 1.6
        height_return, origin, perceived_height = pyramid_stairs_terrain(
            terrain, step_width=step_width, step_height=step_height, platform_size=1.6)
      self.height_field_raw[
          start_x: end_x,
          start_y: end_y
      ] = terrain.height_field_raw

      self.percept_height[
          start_x: end_x,
          start_y: end_y
      ] = perceived_height

      # env_origin_x = (i + 0.5) * self.env_length
      # env_origin_x = i * self.env_length + platform_size / \
      #   2  # move to the center of the platform
      # env_origin_y = (j + 0.5) * self.env_width
      # x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
      # x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
      # y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
      # y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
      # env_origin_z = np.max(
      #   terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
      self.env_origins[i, j] = origin
      self.env_origins[i, j, 0] += i * self.env_length
      self.env_origins[i, j, 1] += j * self.env_width
      self.env_reachable_distance[i, j] = self.env_length - origin[0]
      self.env_random_origin_x_low[i, j] = i * self.env_length + origin[0]
      self.env_random_origin_x_range[i, j] = platform_size * random_scale
      self.env_random_origin_y_low[i, j] = j * self.env_width + origin[1]
      self.env_random_origin_y_range[i, j] = platform_size * random_scale

      self.env_height_low[i, j] = height_low
      self.env_height_high[i, j] = height_high

  def selected_terrain(self, terrain_kwargs):
    terrain_type = terrain_kwargs.pop('type')
    for k in range(self.num_maps):
      # Env coordinates in the world
      (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

      # Heightfield coordinate system from now on
      start_x = self.border + i * self.length_per_env_pixels
      end_x = self.border + (i + 1) * self.length_per_env_pixels
      start_y = self.border + j * self.width_per_env_pixels
      end_y = self.border + (j + 1) * self.width_per_env_pixels

      terrain = SubTerrain("terrain",
                           width=self.width_per_env_pixels,
                           length=self.width_per_env_pixels,
                           vertical_scale=self.vertical_scale,
                           horizontal_scale=self.horizontal_scale)

      eval(terrain_type)(terrain, **terrain_kwargs)
      self.height_field_raw[start_x: end_x,
                            start_y:end_y] = terrain.height_field_raw

      env_origin_x = (i + 0.5) * self.env_length
      env_origin_y = (j + 0.5) * self.env_width
      x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
      x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
      y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
      y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
      env_origin_z = np.max(
          terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
      self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

  # def curiculum(self, num_robots, num_terrains, num_levels):
  #   num_robots_per_map = int(num_robots / num_terrains)
  #   left_over = num_robots % num_terrains
  #   idx = 0
  #   for j in range(num_terrains):
  #     for i in range(num_levels):
  #       terrain = SubTerrain("terrain",
  #                            width=self.width_per_env_pixels,
  #                            length=self.width_per_env_pixels,
  #                            vertical_scale=self.vertical_scale,
  #                            horizontal_scale=self.horizontal_scale)
  #       difficulty = i / num_levels
  #       choice = j / num_terrains + 0.001

  #       slope = difficulty * 0.4
  #       step_height = 0.05 + 0.175 * difficulty
  #       discrete_obstacles_height = 0.025 + difficulty * 0.2
  #       stepping_stones_size = 1.5 * (1.1 - difficulty)
  #       if choice < self.proportions[0]:
  #         if choice < self.proportions[0] / 2:
  #           slope *= -1
  #         pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
  #       elif choice < self.proportions[1]:
  #         # if choice <= (self.proportions[0] + self.proportions[1]) / 2:
  #         #     slope *= -1
  #         pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
  #         rand_uniform_terrain(terrain, min_height=-0, max_height=15, step=5)
  #         terrain.height_field_raw -= 15
  #       elif choice < self.proportions[3]:
  #         if choice < self.proportions[2]:
  #           step_height *= -1
  #         pyramid_stairs_terrain(terrain, step_width=0.31,
  #                                step_height=step_height, platform_size=3.)
  #       elif choice < self.proportions[4]:
  #         num_rectangles = 40
  #         rectangle_min_size = int(1. / self.horizontal_scale)
  #         rectangle_max_size = int(2. / self.horizontal_scale)
  #         discrete_obstacles_terrain(terrain, discrete_obstacles_height,
  #                                    rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
  #       else:
  #         stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
  #                                 stone_distance=0.1, max_height=0., platform_size=3.)

  #       # Heightfield coordinate system
  #       start_x = self.border + i * self.length_per_env_pixels
  #       end_x = self.border + (i + 1) * self.length_per_env_pixels
  #       start_y = self.border + j * self.width_per_env_pixels
  #       end_y = self.border + (j + 1) * self.width_per_env_pixels
  #       self.height_field_raw[start_x: end_x,
  #                             start_y:end_y] = terrain.height_field_raw

  #       robots_in_map = num_robots_per_map
  #       if j < left_over:
  #         robots_in_map += 1

  #       env_origin_x = (i + 0.5) * self.env_length
  #       env_origin_y = (j + 0.5) * self.env_width
  #       x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
  #       x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
  #       y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
  #       y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
  #       env_origin_z = np.max(
  #         terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
  #       self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

  def convert_to_trimesh(self):
    hf = self.height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]
    self.slope_treshold *= self.horizontal_scale / self.vertical_scale

    y = np.linspace(0, (num_cols - 1) * self.horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * self.horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_corners = np.zeros((num_rows, num_cols))
    move_x[:num_rows - 1, :] += self.horizontal_scale * \
        (hf[1:num_rows, :] - hf[:num_rows - 1, :] > self.slope_treshold)
    move_x[1:num_rows, :] -= self.horizontal_scale * \
        (hf[:num_rows - 1, :] - hf[1:num_rows, :] > self.slope_treshold)
    move_y[:, :num_cols - 1] += self.horizontal_scale * \
        (hf[:, 1:num_cols] - hf[:, :num_cols - 1] > self.slope_treshold)
    move_y[:, 1:num_cols] -= self.horizontal_scale * \
        (hf[:, :num_cols - 1] - hf[:, 1:num_cols] > self.slope_treshold)
    move_corners[:num_rows - 1, :num_cols - 1] += self.horizontal_scale * \
        (hf[1:num_rows, 1:num_cols] -
         hf[:num_rows - 1, :num_cols - 1] > self.slope_treshold)
    move_corners[1:num_rows, 1:num_cols] -= self.horizontal_scale * \
        (hf[:num_rows - 1, :num_cols - 1] -
         hf[1:num_rows, 1:num_cols] > self.slope_treshold)

    xx += move_x + move_corners * (move_x == 0)
    yy += move_y + move_corners * (move_y == 0)
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * self.vertical_scale
    triangles = - \
        np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
      ind0 = np.arange(0, num_cols - 1) + i * num_cols
      ind1 = ind0 + 1
      ind2 = ind0 + num_cols
      ind3 = ind2 + 1
      start = 2 * i * (num_cols - 1)
      stop = start + 2 * (num_cols - 1)
      triangles[start:stop:2, 0] = ind0
      triangles[start:stop:2, 1] = ind3
      triangles[start:stop:2, 2] = ind1
      triangles[start + 1:stop:2, 0] = ind0
      triangles[start + 1:stop:2, 1] = ind2
      triangles[start + 1:stop:2, 2] = ind3

    self.vertices = vertices
    self.triangles = triangles
    # Uncomment to write to .obj file
    # file = open("terrain.obj", "w")
    # for v in range(self.vertices.shape[0]):
    #     file.write("v {:.2f} {:.2f} {:.2f}\n".format(*self.vertices[v]))
    # for t in range(self.triangles.shape[0]):
    #     file.write("f {} {} {}\n".format(*self.triangles[t] + 1))
    # file.close()

  def get_prob_distribution_for_terrain_levels(self, num_kind_of_terrain_params, num_levels=0):
    """
    Returns a probability distribution for the terrain levels.
    """
    if num_levels == 0:
      # Easy mode
      prob_dist = [i for i in range(num_kind_of_terrain_params)]
      prob_dist = np.asarray(prob_dist) / sum(prob_dist)
    else:
      prob_dist = np.asarray(
          [1 / num_kind_of_terrain_params for _ in range(num_kind_of_terrain_params)])

    return prob_dist
# def rand_uniform_terrain(terrain, min_height, max_height, step=1):
#   """
#   Generate a uniform noise terrain
#   Parameters
#       terrain (terrain): the terrain
#       min_height (int): the minimum height of the terrain
#       max_height (int): the maximum height of the terrain
#   """
#   range = np.arange(min_height, max_height + step, step)
#   downsampled_scale = 0.05
#   height_field_downsampled = np.random.choice(range, (int(terrain.length * terrain.horizontal_scale / downsampled_scale), int(
#     terrain.width * terrain.horizontal_scale / downsampled_scale)))

#   x = np.linspace(0, terrain.width * terrain.horizontal_scale,
#                   height_field_downsampled.shape[0])
#   y = np.linspace(0, terrain.length * terrain.horizontal_scale,
#                   height_field_downsampled.shape[1])

#   f = interpolate.interp2d(x, y, height_field_downsampled, kind='linear')

#   x_upsampled = np.linspace(
#     0, terrain.width * terrain.horizontal_scale, terrain.length)
#   y_upsampled = np.linspace(
#     0, terrain.length * terrain.horizontal_scale, terrain.width)
#   z_upsampled = np.rint(f(x_upsampled, y_upsampled))

#   terrain.height_field_raw += z_upsampled.astype(np.int16)


# def sloped_terrain(terrain, slope=1):
#   """
#   Generate a sloped terrain
#   Parameters:
#       terrain (terrain): the terrain
#       slope (int): positive or negative slope
#   """
#   x = np.arange(0, terrain.width)
#   y = np.arange(0, terrain.length)
#   xx, yy = np.meshgrid(x, y, sparse=True)
#   terrain.height_field_raw += (slope * yy + 0 *
#                                xx).astype(terrain.height_field_raw.dtype)


# def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
#   """
#   Generate a sloped terrain
#   Parameters:
#       terrain (terrain): the terrain
#       slope (int): positive or negative slope
#   """
#   x = np.arange(0, terrain.width)
#   y = np.arange(0, terrain.length)
#   center_x = int(terrain.width / 2)
#   center_y = int(terrain.length / 2)
#   xx, yy = np.meshgrid(x, y, sparse=True)
#   xx = (center_x - np.abs(center_x-xx)) / center_x
#   yy = (center_y - np.abs(center_y-yy)) / center_y
#   max_height = int(slope * (terrain.horizontal_scale /
#                    terrain.vertical_scale) * (terrain.length / 2))
#   terrain.height_field_raw += (max_height * yy *
#                                xx).astype(terrain.height_field_raw.dtype)

#   platform_size = int(platform_size / terrain.horizontal_scale / 2)
#   x1 = terrain.length // 2 - platform_size
#   x2 = terrain.length // 2 + platform_size
#   y1 = terrain.width // 2 - platform_size
#   y2 = terrain.width // 2 + platform_size

#   min_h = min(terrain.height_field_raw[x1, y1], 0)
#   max_h = max(terrain.height_field_raw[x1, y1], 0)
#   terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)


# def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size):
#   """
#   Generate a terrain with gaps
#   Parameters:
#       terrain (terrain): the terrain
#       min_height (int): min height
#       max_height (int): max height
#   """
#   (i, j) = terrain.height_field_raw.shape
#   d_max_height = int(max_height / terrain.vertical_scale)
#   height_range = [-d_max_height, -d_max_height //
#                   2, d_max_height // 2, d_max_height]
#   width_range = range(min_size, max_size, 4)
#   length_range = range(min_size, max_size, 4)

#   for _ in range(num_rects):
#     width = np.random.choice(width_range)
#     length = np.random.choice(length_range)
#     start_i = np.random.choice(range(0, i-width, 4))
#     start_j = np.random.choice(range(0, j-length, 4))
#     terrain.height_field_raw[start_i:start_i+width,
#                              start_j:start_j+length] = np.random.choice(height_range)

#   # Platform
#   platform_size = int(platform_size / terrain.horizontal_scale / 2)
#   # x1 = terrain.length // 2 - platform_size
#   # x2 = terrain.length // 2 + platform_size
#   x1 = 0
#   x2 = platform_size
#   y1 = terrain.width // 2 - platform_size
#   y2 = terrain.width // 2 + platform_size
#   terrain.height_field_raw[x1:x2, y1:y2] = 0


# def wave_terrain(terrain, num_waves=1):
#   """
#   Generate a wavy terrain
#   Parameters:
#       terrain (terrain): the terrain
#       num_waves (int): number of sine waves across the terrain length
#   """
#   if num_waves > 0:
#     div = terrain.length / (num_waves * np.pi * 2)
#     x = np.arange(0, terrain.width)
#     y = np.arange(0, terrain.length)
#     xx, yy = np.meshgrid(x, y, sparse=True)
#     # Discretize the sinus in 30 buckets
#     num_buckets = 30
#     terrain.height_field_raw += (num_buckets * np.sin(yy / div) + num_buckets * np.cos(xx / div)).astype(
#       terrain.height_field_raw.dtype)


# def stairs_terrain(terrain, step_width, step_height):
#   """
#   Generate a stairs
#   Parameters:
#       terrain (terrain): the terrain
#       step_width (int):  the width of the step
#       slope (int):  the slope of the terrain stairs, i.e. the step_height
#   """
#   d_step_width = int(step_width / terrain.horizontal_scale)  # +1
#   d_step_height = int(step_height / terrain.vertical_scale)
#   num_steps = terrain.length // d_step_width
#   height = d_step_height
#   for i in range(num_steps):
#     terrain.height_field_raw[i *
#                              d_step_width: (i + 1) * d_step_width, :] += height
#     height += d_step_height


# def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size):
#   """
#   Generate stairs
#   Parameters:
#       terrain (terrain): the terrain
#       step_width (float):  the width of the step [m]
#       step_height (float): the step_height [m]
#   """
#   d_step_width = int(step_width / terrain.horizontal_scale)
#   d_step_height = int(step_height / terrain.vertical_scale)
#   height = 0
#   platform_size = int(platform_size / terrain.horizontal_scale)

#   start_x = 0
#   stop_x = terrain.width
#   start_y = 0
#   stop_y = terrain.length
#   points = np.zeros((12, 3))
#   index_offset = 0
#   while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
#     start_x += d_step_width
#     stop_x -= d_step_width
#     start_y += d_step_width
#     stop_y -= d_step_width
#     height += d_step_height
#     terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height


# def stepping_stones_terrain(
#     terrain, stone_size, stone_distance,
#     height_range, obs_prob, platform_size=1.
# ):
#   """
#   Generate a stepping stones terrain
#   Parameters:
#       terrain (terrain): the terrain
#   """
#   d_stone_size = int(stone_size / terrain.horizontal_scale)
#   d_stone_distance = int(stone_distance / terrain.horizontal_scale)

#   # Platform
#   platform_size = int(platform_size / terrain.horizontal_scale / 2)

#   # d_max_height = int(max_height / terrain.vertical_scale)
#   num_steps = terrain.length // (d_stone_size + d_stone_distance)
#   start_y = 0
#   terrain.height_field_raw[:, :] = -1000
#   while start_y < terrain.length:
#     stop_y = min(terrain.length, start_y + d_stone_size)

#     start_x = np.random.randint(0, d_stone_size)
#     # fill first hole
#     stop_x = max(0, start_x - d_stone_distance)
#     terrain.height_field_raw[start_y: stop_y, 0: stop_x] = 0
#     # fill row
#     while start_x < terrain.width:
#       stop_x = min(terrain.width, start_x + d_stone_size)

#       # if start_x < terrain.width // 2 - platform_size:
#       #     start_x += d_stone_size + d_stone_distance
#       #     continue
#       # if start_x > terrain.width // 2 + platform_size:
#       #     start_x += d_stone_size + d_stone_distance
#       #     continue
#       obs_current = np.random.rand()
#       # if obs_current < obs_prob:
#       #     height_random = 100
#       # else:
#       height_random = np.random.uniform(-height_range, height_range)
#       terrain.height_field_raw[
#         start_y: stop_y, start_x: stop_x
#       ] = height_random
#       start_x += d_stone_size + d_stone_distance
#     start_y += d_stone_size + d_stone_distance

#   # x1 = terrain.length // 2 - platform_size
#   # x2 = terrain.length // 2 + platform_size
#   # y1 = terrain.width // 2 - platform_size
#   # y2 = terrain.width // 2 + platform_size
#   x1 = 0
#   x2 = 2 * platform_size
#   y1 = terrain.width // 2 - platform_size
#   y2 = terrain.width // 2 + platform_size
#   terrain.height_field_raw[x1:x2, y1:y2] = 0


# def obstacle_terrain(
#     terrain, stone_size, stone_distance,
#     height_range, obs_prob, platform_size=1.
# ):
#   """
#   Generate a obstacle terrain
#   Parameters:
#       terrain (terrain): the terrain
#   """
#   d_stone_size = int(stone_size / terrain.horizontal_scale)
#   d_stone_distance = int(stone_distance / terrain.horizontal_scale)
#   # d_max_height = int(max_height / terrain.vertical_scale)
#   num_steps = terrain.length // (d_stone_size + d_stone_distance)
#   start_y = 0
#   # terrain.height_field_raw[:, :] = -1000
#   while start_y < terrain.length:
#     stop_y = min(terrain.length, start_y + d_stone_size)

#     start_x = np.random.randint(0, d_stone_size)
#     # fill first hole
#     stop_x = max(0, start_x - d_stone_distance)
#     # terrain.height_field_raw[start_y: stop_y, 0: stop_x] = 0
#     # fill row
#     while start_x < terrain.width:
#       stop_x = min(terrain.width, start_x + d_stone_size)
#       obs_current = np.random.rand()
#       if obs_current < obs_prob:
#         height_random = 100
#       # else:
#       #     height_random = np.random.uniform(-height_range, height_range)
#         terrain.height_field_raw[
#           start_y: stop_y, start_x: stop_x
#         ] = height_random
#       start_x += d_stone_size + d_stone_distance
#     start_y += d_stone_size + d_stone_distance

#   # Platform
#   platform_size = int(platform_size / terrain.horizontal_scale / 2)
#   # x1 = terrain.length // 2 - platform_size
#   # x2 = terrain.length // 2 + platform_size
#   # y1 = terrain.width // 2 - platform_size
#   # y2 = terrain.width // 2 + platform_size
#   x1 = 0
#   x2 = platform_size * 2
#   y1 = terrain.width // 2 - platform_size
#   y2 = terrain.width // 2 + platform_size
#   terrain.height_field_raw[x1:x2, y1:y2] = 0
