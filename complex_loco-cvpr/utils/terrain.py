import numpy as np
from numpy.random import choice
from scipy import interpolate


class SubTerrain:
  def __init__(self, terrain_name, width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
    self.terrain_name = terrain_name
    self.width = width
    self.length = length
    self.height_field_raw = np.zeros(
      (self.length, self.width), dtype=np.int16)

    self.vertical_scale = vertical_scale
    self.horizontal_scale = horizontal_scale


class Terrain:
  def __init__(self, cfg, num_robots) -> None:

    self.type = "trimesh"
    self.horizontal_scale = 0.1  # [m]
    self.vertical_scale = 0.005  # [m]
    self.border_size = 20  # [m]
    self.env_length = cfg["map_length"]
    self.env_width = cfg["map_width"]
    self.proportions = [np.sum(cfg["terrain_proportions"][:i + 1])
                        for i in range(len(cfg["terrain_proportions"]))]

    self.env_rows = cfg["num_levels"]
    self.env_cols = cfg["num_terrains"]
    self.num_maps = self.env_rows * self.env_cols
    self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

    self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
    self.length_per_env_pixels = int(
      self.env_length / self.horizontal_scale)

    self.border = int(self.border_size / self.horizontal_scale)
    self.tot_cols = int(
      self.env_cols * self.width_per_env_pixels) + 2 * self.border
    self.tot_rows = int(
      self.env_rows * self.length_per_env_pixels) + 2 * self.border

    self.height_field_raw = np.zeros(
      (self.tot_rows, self.tot_cols), dtype=np.int16)
    if cfg["curriculum"]:
      self.curiculum(num_robots, num_terrains=self.env_cols,
                     num_levels=self.env_rows)
    elif cfg["selected"]:
      self.selected_terrain(cfg["terrain_kwargs"])
    else:
      self.randomized_terrain()
    self.heightsamples = np.reshape(self.height_field_raw, (-1), order='F')

    if self.type == "trimesh":
      self.slope_treshold = cfg["slope_treshold"]
      self.convert_to_trimesh()

  def curiculum(self, num_robots, num_terrains, num_levels):
    num_robots_per_map = int(num_robots / num_terrains)
    left_over = num_robots % num_terrains
    idx = 0
    for j in range(num_terrains):
      for i in range(num_levels):
        terrain = SubTerrain("terrain",
                             width=self.width_per_env_pixels,
                             length=self.width_per_env_pixels,
                             vertical_scale=self.vertical_scale,
                             horizontal_scale=self.horizontal_scale)
        difficulty = i / num_levels
        choice = j / num_terrains + 0.001

        slope = difficulty * 0.4
        step_height = 0.05 + 0.175 * difficulty
        discrete_obstacles_height = 0.025 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.1 - difficulty)
        if choice < self.proportions[0]:
          if choice < self.proportions[0] / 2:
            slope *= -1
          pyramid_sloped_terrain(
            terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
          # if choice <= (self.proportions[0] + self.proportions[1]) / 2:
          #     slope *= -1
          pyramid_sloped_terrain(
            terrain, slope=slope, platform_size=3.)
          rand_uniform_terrain(
            terrain, min_height=-0, max_height=15, step=5)
          terrain.height_field_raw -= 15
        elif choice < self.proportions[3]:
          if choice < self.proportions[2]:
            step_height *= -1
          pyramid_stairs_terrain(
            terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
          num_rectangles = 40
          rectangle_min_size = int(1. / self.horizontal_scale)
          rectangle_max_size = int(2. / self.horizontal_scale)
          discrete_obstacles_terrain(
            terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        else:
          stepping_stones_terrain(
            terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0., platform_size=3.)

        # Heightfield coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x,
                              start_y:end_y] = terrain.height_field_raw

        robots_in_map = num_robots_per_map
        if j < left_over:
          robots_in_map += 1

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
        env_origin_z = np.max(
          terrain.height_field_raw[x1:x2, y1:y2])*self.vertical_scale
        self.env_origins[i, j] = [
          env_origin_x, env_origin_y, env_origin_z]

  def randomized_terrain(self):
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
      choice = np.random.uniform(0, 1)
      if choice < 0.3:
        if np.random.choice([0, 1]):
          pyramid_sloped_terrain(
            terrain, np.random.choice([-0.3, 0.3]))
          rand_uniform_terrain(
            terrain, min_height=-0, max_height=10, step=5)
          terrain.height_field_raw -= 10
        else:
          pyramid_sloped_terrain(
            terrain, np.random.choice([-0.3, 0.3]))
      elif choice < 0.7:
        step_height = np.random.choice([-0.18, -0.15, 0.15, 0.18])
        pyramid_stairs_terrain(
          terrain, step_width=0.31, step_height=step_height, platform_size=3.)
      elif choice < 1.:
        num_rectangles = 40
        rectangle_min_size = int(1. / self.horizontal_scale)
        rectangle_max_size = int(2. / self.horizontal_scale)
        max_h = 0.15
        discrete_obstacles_terrain(
          terrain, max_h, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)

      self.height_field_raw[start_x: end_x,
                            start_y:end_y] = terrain.height_field_raw

      env_origin_x = (i + 0.5) * self.env_length
      env_origin_y = (j + 0.5) * self.env_width
      x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
      x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
      y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
      y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
      env_origin_z = np.max(
        terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
      self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

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

  def convert_to_trimesh(self):
    hf = self.height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]
    self.slope_treshold *= self.horizontal_scale / self.vertical_scale

    y = np.linspace(0, (num_cols-1)*self.horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*self.horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    move_x = np.zeros((num_rows, num_cols))
    move_y = np.zeros((num_rows, num_cols))
    move_corners = np.zeros((num_rows, num_cols))
    move_x[:num_rows-1, :] += self.horizontal_scale * \
      (hf[1:num_rows, :] - hf[:num_rows-1, :] > self.slope_treshold)
    move_x[1:num_rows, :] -= self.horizontal_scale * \
      (hf[:num_rows-1, :] - hf[1:num_rows, :] > self.slope_treshold)
    move_y[:, :num_cols-1] += self.horizontal_scale * \
      (hf[:, 1:num_cols] - hf[:, :num_cols-1] > self.slope_treshold)
    move_y[:, 1:num_cols] -= self.horizontal_scale * \
      (hf[:, :num_cols-1] - hf[:, 1:num_cols] > self.slope_treshold)
    move_corners[:num_rows-1, :num_cols-1] += self.horizontal_scale * \
      (hf[1:num_rows, 1:num_cols] -
       hf[:num_rows-1, :num_cols-1] > self.slope_treshold)
    move_corners[1:num_rows, 1:num_cols] -= self.horizontal_scale * \
      (hf[:num_rows-1, :num_cols-1] -
       hf[1:num_rows, 1:num_cols] > self.slope_treshold)

    xx += move_x + move_corners*(move_x == 0)
    yy += move_y + move_corners*(move_y == 0)
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * self.vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows-1):
      ind0 = np.arange(0, num_cols-1) + i*num_cols
      ind1 = ind0 + 1
      ind2 = ind0 + num_cols
      ind3 = ind2 + 1
      start = 2*i*(num_cols-1)
      stop = start + 2*(num_cols-1)
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


def rand_uniform_terrain(terrain, min_height, max_height, step=1):
  """
  Generate a uniform noise terrain

  Parameters
      terrain (terrain): the terrain
      min_height (int): the minimum height of the terrain
      max_height (int): the maximum height of the terrain

  """
  range = np.arange(min_height, max_height + step, step)
  downsampled_scale = 0.05
  height_field_downsampled = np.random.choice(range, (int(terrain.length * terrain.horizontal_scale / downsampled_scale), int(
    terrain.width * terrain.horizontal_scale / downsampled_scale)))

  x = np.linspace(0, terrain.width * terrain.horizontal_scale,
                  height_field_downsampled.shape[0])
  y = np.linspace(0, terrain.length * terrain.horizontal_scale,
                  height_field_downsampled.shape[1])

  f = interpolate.interp2d(x, y, height_field_downsampled, kind='linear')

  x_upsampled = np.linspace(
    0, terrain.width * terrain.horizontal_scale, terrain.length)
  y_upsampled = np.linspace(
    0, terrain.length * terrain.horizontal_scale, terrain.width)
  z_upsampled = np.rint(f(x_upsampled, y_upsampled))

  terrain.height_field_raw += z_upsampled.astype(np.int16)


def sloped_terrain(terrain, slope=1):
  """
  Generate a sloped terrain

  Parameters:
      terrain (terrain): the terrain
      slope (int): positive or negative slope
  """
  x = np.arange(0, terrain.width)
  y = np.arange(0, terrain.length)
  xx, yy = np.meshgrid(x, y, sparse=True)
  terrain.height_field_raw += (slope * yy + 0 *
                               xx).astype(terrain.height_field_raw.dtype)


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
  """
  Generate a sloped terrain

  Parameters:
      terrain (terrain): the terrain
      slope (int): positive or negative slope
  """
  x = np.arange(0, terrain.width)
  y = np.arange(0, terrain.length)
  center_x = int(terrain.width / 2)
  center_y = int(terrain.length / 2)
  xx, yy = np.meshgrid(x, y, sparse=True)
  xx = (center_x - np.abs(center_x - xx)) / center_x
  yy = (center_y - np.abs(center_y - yy)) / center_y
  max_height = int(slope * (terrain.horizontal_scale /
                            terrain.vertical_scale) * (terrain.length / 2))
  terrain.height_field_raw += (max_height * yy *
                               xx).astype(terrain.height_field_raw.dtype)

  platform_size = int(platform_size / terrain.horizontal_scale / 2)
  x1 = terrain.length // 2 - platform_size
  x2 = terrain.length // 2 + platform_size
  y1 = terrain.width // 2 - platform_size
  y2 = terrain.width // 2 + platform_size

  min_h = min(terrain.height_field_raw[x1, y1], 0)
  max_h = max(terrain.height_field_raw[x1, y1], 0)
  terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size):
  """
  Generate a terrain with gaps

  Parameters:
      terrain (terrain): the terrain
      min_height (int): min height
      max_height (int): max height
  """
  (i, j) = terrain.height_field_raw.shape
  d_max_height = int(max_height / terrain.vertical_scale)
  height_range = [-d_max_height, -d_max_height //
                  2, d_max_height // 2, d_max_height]
  width_range = range(min_size, max_size, 4)
  length_range = range(min_size, max_size, 4)

  for _ in range(num_rects):
    width = np.random.choice(width_range)
    length = np.random.choice(length_range)
    start_i = np.random.choice(range(0, i-width, 4))
    start_j = np.random.choice(range(0, j-length, 4))
    terrain.height_field_raw[start_i:start_i+width,
                             start_j:start_j+length] = np.random.choice(height_range)

  platform_size = int(platform_size / terrain.horizontal_scale / 2)
  x1 = terrain.length // 2 - platform_size
  x2 = terrain.length // 2 + platform_size
  y1 = terrain.width // 2 - platform_size
  y2 = terrain.width // 2 + platform_size
  terrain.height_field_raw[x1:x2, y1:y2] = 0


def wave_terrain(terrain, num_waves=1):
  """
  Generate a wavy terrain

  Parameters:
      terrain (terrain): the terrain
      num_waves (int): number of sine waves across the terrain length
  """
  if num_waves > 0:
    div = terrain.length / (num_waves * np.pi * 2)
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # Discretize the sinus in 30 buckets
    num_buckets = 30
    terrain.height_field_raw += (num_buckets * np.sin(yy / div) + num_buckets * np.cos(xx / div)).astype(
      terrain.height_field_raw.dtype)


def stairs_terrain(terrain, step_width, step_height):
  """
  Generate a stairs

  Parameters:
      terrain (terrain): the terrain
      step_width (int):  the width of the step
      slope (int):  the slope of the terrain stairs, i.e. the step_height
  """
  d_step_width = int(step_width / terrain.horizontal_scale)  # +1
  d_step_height = int(step_height / terrain.vertical_scale)
  num_steps = terrain.length // d_step_width
  height = d_step_height
  for i in range(num_steps):
    terrain.height_field_raw[i *
                             d_step_width: (i + 1) * d_step_width, :] += height
    height += d_step_height


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size):
  """
  Generate stairs

  Parameters:
      terrain (terrain): the terrain
      step_width (float):  the width of the step [m]
      step_height (float): the step_height [m]
  """
  d_step_width = int(step_width / terrain.horizontal_scale)
  d_step_height = int(step_height / terrain.vertical_scale)
  height = 0
  platform_size = int(platform_size / terrain.horizontal_scale)

  start_x = 0
  stop_x = terrain.width
  start_y = 0
  stop_y = terrain.length
  points = np.zeros((12, 3))
  index_offset = 0
  while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
    start_x += d_step_width
    stop_x -= d_step_width
    start_y += d_step_width
    stop_y -= d_step_width
    height += d_step_height
    terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1.):
  """
  Generate a stepping stones terrain

  Parameters:
      terrain (terrain): the terrain
  """
  d_stone_size = int(stone_size / terrain.horizontal_scale)
  d_stone_distance = int(stone_distance / terrain.horizontal_scale)
  d_max_height = int(max_height / terrain.vertical_scale)
  num_steps = terrain.length // (d_stone_size + d_stone_distance)
  start_y = 0
  terrain.height_field_raw[:, :] = -1000
  while start_y < terrain.length:
    stop_y = min(terrain.length, start_y + d_stone_size)

    start_x = np.random.randint(0, d_stone_size)
    # fill first hole
    stop_x = max(0, start_x - d_stone_distance)
    terrain.height_field_raw[start_y: stop_y, 0: stop_x] = max_height
    # fill row
    while start_x < terrain.width:
      stop_x = min(terrain.width, start_x + d_stone_size)
      terrain.height_field_raw[start_y: stop_y,
                               start_x: stop_x] = max_height
      start_x += d_stone_size + d_stone_distance
    start_y += d_stone_size + d_stone_distance

  platform_size = int(platform_size / terrain.horizontal_scale / 2)
  x1 = terrain.length // 2 - platform_size
  x2 = terrain.length // 2 + platform_size
  y1 = terrain.width // 2 - platform_size
  y2 = terrain.width // 2 + platform_size
  terrain.height_field_raw[x1:x2, y1:y2] = 0


if __name__ == "__main__":
  terrain = Terrain()
