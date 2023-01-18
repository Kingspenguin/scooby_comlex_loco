# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from tkinter import HORIZONTAL
import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import gymutil, gymapi
from math import sqrt
import torch


def traversability_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=0.8, random_scale=0.1):

  # switch parameters to discrete units
  max_height = int(max_height / terrain.vertical_scale)
  min_size = int(min_size / terrain.horizontal_scale)
  max_size = int(max_size / terrain.horizontal_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  init_height = 0

  (i, j) = terrain.height_field_raw.shape
  width_range = range(min_size, max_size)
  length_range = range(min_size, max_size)

  for _ in range(num_rects):
    width = np.random.choice(width_range)
    length = np.random.choice(length_range)
    start_i = np.random.choice(range(0, i - width, 4))
    start_j = np.random.choice(range(0, j - length, 4))
    terrain.height_field_raw[start_i:start_i + width,
                             start_j:start_j + length] = np.random.uniform(-max_height, max_height)

  terrain.height_field_raw[:, :2] = max_height
  terrain.height_field_raw[:, -2:] = max_height

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.8 / terrain.horizontal_scale))
  x1 = a1_size
  x2 = 2 * a1_size
  y1 = int((terrain.length - platform_size -
           2 * random_scale * platform_size) / 2)
  y2 = int((terrain.length + platform_size +
           2 * random_scale * platform_size) / 2)
  terrain.height_field_raw[x1:x2, y1:y2] = init_height
  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, init_height)


def obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=0.8, random_scale=0.1):

  # switch parameters to discrete units
  max_height = int(max_height / terrain.vertical_scale)
  min_size = int(min_size / terrain.horizontal_scale)
  max_size = int(max_size / terrain.horizontal_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  init_height = 0

  (i, j) = terrain.height_field_raw.shape

  percept_height_field = np.zeros_like(terrain.height_field_raw)

  # print(i, j)
  width_range = range(min_size, max_size)
  # print(width_range)
  length_range = range(min_size, max_size)

  for o_i in range(5):
    square_size = i // 5
    for o_j in range(num_rects // 5 + 1):
      w = j // (num_rects // 5)
      # print(w, j)
      anchor_x = int((o_i + 0.1) * square_size)
      anchor_y = o_j * w + (o_i % 2 == 0) * w // 2
      anchor_x += int(
          square_size * (np.random.rand() - 0.5) * 0.4
      )
      anchor_y += int(
          w * (np.random.rand() - 0.5) * 0.8
      )
      width = np.random.choice(width_range)
      # width = 5
      # print(o_i, o_j, width, width_range)
      # print(o_i, o_j, anchor_x, anchor_x + width)
      # print(o_i, o_j, anchor_y, anchor_y + width)
  #   length = np.random.choice(length_range)
  #   start_i = np.random.choice(range(0, i - width, 4))
  #   start_j = np.random.choice(range(0, j - length, 4))
      terrain.height_field_raw[
          anchor_x: anchor_x + width,
          anchor_y: anchor_y + width
      ] = np.random.uniform(max_height // 2, max_height)

  terrain.height_field_raw[:, :2] = max_height
  terrain.height_field_raw[:, -2:] = max_height

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.4 / terrain.horizontal_scale))
  x1 = int(0.5 * a1_size)
  x2 = int(1.5 * a1_size)
  pos_random = (np.random.rand() * 0.8 + 0.1) * j
  y1 = int((pos_random - platform_size / 2 - random_scale * platform_size / 2))
  y2 = int((pos_random + platform_size / 2 + random_scale * platform_size / 2))
  # print(y1, y2, pos_random)
  terrain.height_field_raw[0: i // 6, :] = init_height
  terrain.height_field_raw[:, :2] = max_height
  terrain.height_field_raw[:, -2:] = max_height

  percept_height_field[:, :] = terrain.height_field_raw

  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, init_height), percept_height_field


def wave_terrain(terrain, num_waves=1, amplitude=1.):
  percept_height_field = np.zeros_like(terrain.height_field_raw)
  wall_height = int(2.2 / terrain.vertical_scale)
  amplitude = int(0.5 * amplitude / terrain.vertical_scale)
  if num_waves > 0:
    div = terrain.length / (num_waves * np.pi * 2)
    x = np.arange(0, terrain.length)
    y = np.arange(0, terrain.width)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.length, 1)
    yy = yy.reshape(1, terrain.width)
    terrain.height_field_raw += (amplitude * np.cos(yy / div) + amplitude * np.sin(xx / div) - amplitude * np.sqrt(2)).astype(
        terrain.height_field_raw.dtype)
  pos_xy = np.random.uniform(low=(0.7 / terrain.horizontal_scale, terrain.width / 2 - 0.5 / terrain.horizontal_scale), high=(
      1.5 / terrain.horizontal_scale, terrain.width / 2 + 0.5 / terrain.horizontal_scale), size=2)
  x, y = pos_xy[0], pos_xy[1]
  init_height = terrain.height_field_raw[int(x), int(y)]

  terrain.height_field_raw[:, :2] = wall_height
  terrain.height_field_raw[:, -2:] = wall_height

  percept_height_field[:, :] = terrain.height_field_raw
  return terrain, (x * terrain.horizontal_scale, y * terrain.horizontal_scale, init_height * terrain.vertical_scale), percept_height_field


def corridor_terrain(terrain, platform_size=0.8):
  platform_size = int(platform_size / terrain.horizontal_scale)
  terrain.height_field_raw[:, :] = -5 / terrain.vertical_scale
  a1_size = int((0.8 / terrain.horizontal_scale))
  x1 = a1_size
  x2 = 2 * a1_size
  y1 = int((terrain.length - 3 * platform_size) / 2)
  y2 = int((terrain.length + 3 * platform_size) / 2)
  terrain.height_field_raw[:, y1:y2] = 0
  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]
  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, 0)


def jumping_stages_terrain(terrain, step_length, step_height, platform_size=0.8):
  wall_height = int(2.2 / terrain.vertical_scale)
  # switch parameters to discrete units
  step_length = int(step_length / terrain.horizontal_scale)
  step_height = int(step_height / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)

  num_steps = terrain.length // step_length
  height = step_height
  for i in range(num_steps):
    if i % 3 == 1:
      terrain.height_field_raw[
          i * step_length: (i + 1) * step_length, :
      ] -= height
    # height += step_height

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.8 / terrain.horizontal_scale))
  x1 = int(0.5 * a1_size)
  x2 = int(1.5 * a1_size)
  y1 = int((terrain.width - platform_size) / 2)
  y2 = int((terrain.width + platform_size) / 2)
  # terrain.height_field_raw[x1:x2, y1:y2] = init_height
  terrain.height_field_raw[x1:x2, :] = 0

  terrain.height_field_raw[:, :2] = wall_height
  terrain.height_field_raw[:, -2:] = wall_height

  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, 0)


def jumping_obstacles_terrain(terrain, step_length, step_height, platform_size=0.8):
  # switch parameters to discrete units
  wall_height = int(2.2 / terrain.vertical_scale)
  step_length = int(step_length / terrain.horizontal_scale)
  step_height = int(step_height / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)

  num_steps = terrain.length // step_length
  height = step_height
  for i in range(num_steps):
    if i % 4 == 3:
      height = int(step_height * np.random.uniform(low=0.5, high=1.5))
      terrain.height_field_raw[i *
                               step_length: (i + 1) * step_length, :] += height
    # height += step_height

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.8 / terrain.horizontal_scale))
  x1 = int(0.5 * a1_size)
  x2 = int(1.5 * a1_size)
  y1 = int((terrain.width - platform_size) / 2)
  y2 = int((terrain.width + platform_size) / 2)
  # terrain.height_field_raw[x1:x2, y1:y2] = init_height
  terrain.height_field_raw[x1:x2, :] = 0

  terrain.height_field_raw[:, :2] = wall_height
  terrain.height_field_raw[:, -2:] = wall_height

  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, 0)


def stepping_barriers_terrain(terrain, step_length, step_height, platform_size=1.2):
  wall_height = int(2.2 / terrain.vertical_scale)
  # switch parameters to discrete units
  step_length = int(step_length / terrain.horizontal_scale)
  step_height = int(step_height / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)

  num_steps = terrain.length // step_length
  height = step_height
  for i in range(num_steps):
    if i % 2 == 1:
      terrain.height_field_raw[i *
                               step_length: (i + 1) * step_length, :] -= height
    # height += step_height

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.8 / terrain.horizontal_scale))
  x1 = int(0.5 * a1_size)
  x2 = int(1.5 * a1_size)
  y1 = int((terrain.width - platform_size) / 2)
  y2 = int((terrain.width + platform_size) / 2)
  # terrain.height_field_raw[x1:x2, y1:y2] = init_height
  terrain.height_field_raw[x1:x2, :] = 0

  terrain.height_field_raw[:, :2] = wall_height
  terrain.height_field_raw[:, -2:] = wall_height

  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, 0)


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=0.8):
  percept_height_field = np.zeros_like(terrain.height_field_raw)
  max_height = int(3.5 / terrain.vertical_scale)

  (i, j) = terrain.height_field_raw.shape
  step_width = int(step_width / terrain.horizontal_scale)
  step_height = int(step_height / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  height = 0
  start_x = 0
  stop_x = terrain.length
  # start_y = 0
  # stop_y = terrain.length
  # and (stop_y - start_y) > platform_size:
  while (stop_x - start_x) > platform_size:
    start_x += step_width
    stop_x -= step_width
    # start_y += step_width
    # stop_y -= step_width
    height += step_height
    terrain.height_field_raw[start_x: stop_x, :] = height

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)

  x1 = int((terrain.length - platform_size) / 2)
  x2 = int((terrain.length + platform_size) / 2)
  y1 = int((terrain.width - platform_size) / 2)
  y2 = int((terrain.width + platform_size) / 2)
  z = np.max(terrain.height_field_raw[x1:x2, :]) * terrain.vertical_scale
  # print(z)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  terrain.height_field_raw[x1:x2, :] = height - step_height
  z = np.max(terrain.height_field_raw[x1:x2, :]) * terrain.vertical_scale
  # print(z)
  # exit()

  terrain.height_field_raw[:, :2] = max_height
  terrain.height_field_raw[:, -2:] = max_height

  percept_height_field[:, :] = terrain.height_field_raw
  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, z), percept_height_field


def stepping_stones_terrain(
    terrain, stone_length, stone_width,
    stone_distance_x, stone_distance_y, max_height, platform_size=0.8, depth=-3,
    percept_depth=-3,
):
  percept_height_field = np.zeros_like(terrain.height_field_raw)
  # switch parameters to discrete units
  wall_height = int(2.2 / terrain.vertical_scale)
  max_height = int(max_height / terrain.vertical_scale)
  depth = int(depth / terrain.vertical_scale)
  percept_depth = int(percept_depth / terrain.vertical_scale)
  stone_length = int(stone_length / terrain.horizontal_scale)
  stone_width = int(stone_width / terrain.horizontal_scale)
  stone_distance_x = int(stone_distance_x / terrain.horizontal_scale)
  stone_distance_y = int(stone_distance_y / terrain.horizontal_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  max_height_gap = 0.
  # max_height_gap = max_height / 10.

  start_x = 0
  start_y = 0
  init_height = curr_height = 0
  terrain.height_field_raw[:, :] = depth
  percept_height_field[:, :] = percept_depth
  # print(terrain.length >= terrain.width)
  # exit()
  assert terrain.length >= terrain.width
  if terrain.length >= terrain.width:

    row_count = 0
    while start_x < terrain.length:
      # print(stone_distance)
      row_count += 1
      stop_x = min(terrain.length, start_x + stone_length)
      # start_y = terrain.width // 2 - int(1.2 *
      #                                    platform_size) + np.random.randint(0, stone_width)

      start_y = np.random.randint(0, stone_width)
      # fill first hole
      stop_y = max(0, start_y - stone_distance_y)
      # terrain.height_field_raw[0: stop_x,
      #                          start_y: stop_y] = curr_height
      # fill row
      # enlarge_stone_distance = stone_distance

      enlarge_stone_distance_x = int(stone_distance_x * (1.11 ** row_count))
      enlarge_stone_distance_y = int(stone_distance_y * (1.03 ** row_count))
      # while start_y < terrain.width // 2 + 0.8 * platform_size:
      while start_y < terrain.width:
        # print(enlarge_stone_distance)
        curr_height += np.random.choice([-max_height_gap, max_height_gap])
        curr_height = max(min(curr_height, max_height), -max_height)
        stop_y = min(terrain.width, start_y + stone_width)
        terrain.height_field_raw[start_x: stop_x,
                                 start_y: stop_y] = curr_height
        percept_height_field[start_x: stop_x,
                             start_y: stop_y] = curr_height
        start_y += stone_width + enlarge_stone_distance_y
      start_x += stone_length + enlarge_stone_distance_x
  # elif terrain.width > terrain.length:
  #   while start_x < terrain.width:
  #     stop_x = min(terrain.width, start_x + stone_size)
  #     start_y = np.random.randint(0, stone_size)
  #     # fill first hole
  #     stop_y = max(0, start_y - stone_distance)
  #     terrain.height_field_raw[start_x: stop_x,
  #                              0: stop_y] = curr_height
  #     # fill column
  #     while start_y < terrain.length:
  #       curr_height += np.random.choice([-max_height_gap, max_height_gap])
  #       curr_height = max(min(curr_height, max_height), -max_height)
  #       stop_y = min(terrain.length, start_y + stone_size)
  #       terrain.height_field_raw[start_x: stop_x,
  #                                start_y: stop_y] = curr_height
  #       start_y += stone_size + stone_distance
  #     start_x += stone_size + stone_distance
  # exit()
  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  # terrain.height_field_raw += (
  #     (random_height >= 0.7) * (random_height - 0.5) *
  #     0.1 / terrain.vertical_scale
  # ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.8 / terrain.horizontal_scale))
  # x1 = int(0.5 * a1_size)
  # x2 = int(1.5 * a1_size)
  x1 = int(0.1 * platform_size)
  x2 = int(1.1 * platform_size)
  y1 = int((terrain.width - platform_size) / 2)
  y2 = int((terrain.width + platform_size) / 2)
  terrain.height_field_raw[x1:x2, y1:y2] = init_height

  terrain.height_field_raw[:, :2] = wall_height
  terrain.height_field_raw[:, -2:] = wall_height

  percept_height_field[x1:x2, y1:y2] = init_height
  percept_height_field[:, :2] = wall_height
  percept_height_field[:, -2:] = wall_height

  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, init_height), percept_height_field


def hard_stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=0.8, depth=-3):
  # switch parameters to discrete units
  max_height = int(max_height / terrain.vertical_scale)
  depth = int(depth / terrain.vertical_scale)
  stone_size = int(stone_size / terrain.horizontal_scale)
  stone_distance = int(stone_distance / terrain.horizontal_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  # max_height_gap = 0.
  max_height_gap = max_height / 10.

  start_x = 0
  start_y = 0
  init_height = curr_height = 0
  terrain.height_field_raw[:, :] = depth
  if terrain.length >= terrain.width:
    while start_y < terrain.length:
      stop_y = min(terrain.length, start_y + stone_size)
      start_x = np.random.randint(0, stone_size)
      # fill first hole
      stop_x = max(0, start_x - stone_distance)
      terrain.height_field_raw[0: stop_x,
                               start_y: stop_y] = curr_height
      # fill row
      while start_x < terrain.width:
        curr_height = init_height + \
            np.random.choice([-max_height_gap, max_height_gap])
        stop_x = min(terrain.width, start_x + stone_size)
        terrain.height_field_raw[start_x: stop_x,
                                 start_y: stop_y] = curr_height
        start_x += stone_size + stone_distance
      start_y += stone_size + stone_distance
  elif terrain.width > terrain.length:
    while start_x < terrain.width:
      stop_x = min(terrain.width, start_x + stone_size)
      start_y = np.random.randint(0, stone_size)
      # fill first hole
      stop_y = max(0, start_y - stone_distance)
      terrain.height_field_raw[start_x: stop_x,
                               0: stop_y] = curr_height
      # fill column
      while start_y < terrain.length:
        curr_height = init_height + \
            np.random.choice([-max_height_gap, max_height_gap])
        stop_y = min(terrain.length, start_y + stone_size)
        terrain.height_field_raw[start_x: stop_x,
                                 start_y: stop_y] = curr_height
        start_y += stone_size + stone_distance
      start_x += stone_size + stone_distance

  random_height = np.random.rand(
      terrain.length // 8, terrain.width // 8)
  terrain.height_field_raw += (
      (random_height >= 0.7) * (random_height - 0.5) *
      0.1 / terrain.vertical_scale
  ).astype(np.int16).repeat(9, axis=0).repeat(9, axis=1)[:terrain.length, :terrain.width]

  a1_size = int((0.8 / terrain.horizontal_scale))
  x1 = a1_size
  x2 = 2 * a1_size
  y1 = int((terrain.length - platform_size) / 2)
  y2 = int((terrain.length + platform_size) / 2)
  terrain.height_field_raw[x1:x2, y1:y2] = init_height
  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, init_height)


def cliffs_terrain(
    terrain, stone_size, stone_distance, max_height, downsampled_scale, platform_size=0.8, depth=-3,
    percept_depth=-3
):
  percept_height_field = np.zeros_like(terrain.height_field_raw)
  # switch parameters to discrete units
  depth = int(depth / terrain.vertical_scale)
  percept_depth = int(percept_depth / terrain.vertical_scale)
  stone_size = int(stone_size / terrain.horizontal_scale)
  stone_distance = int(stone_distance / terrain.horizontal_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  start_x = 0
  start_y = 0
  init_height = 0.05
  terrain.height_field_raw[:, :] = depth
  percept_height_field[:, :] = percept_depth
  print(depth, percept_depth)

  assert terrain.length >= terrain.width
  # if terrain.length >= terrain.width:
  row_count = 0
  while start_x < terrain.length:
    row_count += 1
    enlarge_stone_distance = int(stone_distance * (1.10 ** row_count))
    # print(enlarge_stone_distance)
    stop_x = min(terrain.length, start_x + stone_size)
    start_y = np.random.randint(0, stone_size)
    # fill first hole
    stop_y = max(0, start_y - stone_distance)
    terrain.height_field_raw[start_x: stop_x,
                             0: stop_y] = 0
    percept_height_field[start_x: stop_x,
                         0: stop_y] = 0
    # fill row
    while start_y < terrain.width:
      stop_y = min(terrain.width, start_y + stone_size)
      # print(start_y, stop_y)
      terrain.height_field_raw[start_x: stop_x,
                               start_y: stop_y] = 0
      percept_height_field[start_x: stop_x,
                           start_y: stop_y] = 0
      start_y += stone_size + stone_distance
    start_x += stone_size + enlarge_stone_distance
  # elif terrain.width > terrain.length:
  #   while start_x < terrain.width:
  #     stop_x = min(terrain.width, start_x + stone_size)
  #     start_y = np.random.randint(0, stone_size)
  #     # fill first hole
  #     stop_y = max(0, start_y - stone_distance)
  #     terrain.height_field_raw[start_x: stop_x,
  #                              0: stop_y] = 0
  #     # fill column
  #     while start_y < terrain.length:
  #       stop_y = min(terrain.length, start_y + stone_size)
  #       terrain.height_field_raw[start_x: stop_x,
  #                                start_y: stop_y] = 0
  #       start_y += stone_size + stone_distance
  #     start_x += stone_size + stone_distance

  heights_range = np.arange(-max_height, max_height + 0.1 /
                            terrain.vertical_scale, 0.1 / terrain.vertical_scale)
  height_field_downsampled = np.random.choice(heights_range, (int(terrain.length * terrain.horizontal_scale / downsampled_scale), int(
      terrain.width * terrain.horizontal_scale / downsampled_scale)))
  x = np.linspace(0, terrain.length * terrain.horizontal_scale,
                  height_field_downsampled.shape[0])
  y = np.linspace(0, terrain.width * terrain.horizontal_scale,
                  height_field_downsampled.shape[1])

  f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

  x_upsampled = np.linspace(
      0, terrain.length * terrain.horizontal_scale, terrain.length)
  y_upsampled = np.linspace(
      0, terrain.width * terrain.horizontal_scale, terrain.width)
  z_upsampled = np.rint(f(y_upsampled, x_upsampled))
  terrain.height_field_raw += z_upsampled.astype(np.int16)
  percept_height_field += z_upsampled.astype(np.int16)

  a1_size = int((0.8 / terrain.horizontal_scale))
  # x1 = int(0.5 * a1_size)
  # x2 = int(1.5 * a1_size)

  x1 = int(0.1 * platform_size)
  x2 = int(1.8 * platform_size)

  y1 = int((terrain.width - platform_size) / 2)
  y2 = int((terrain.width + platform_size) / 2)
  terrain.height_field_raw[x1:x2, y1:y2] = int(
      init_height / terrain.vertical_scale
  )
  percept_height_field[x1:x2, y1:y2] = int(
      init_height / terrain.vertical_scale
  )

  wall_height = int(2.2 / terrain.vertical_scale)
  terrain.height_field_raw[:, :2] = wall_height
  terrain.height_field_raw[:, -2:] = wall_height

  percept_height_field[:, :2] = wall_height
  percept_height_field[:, -2:] = wall_height

  return terrain, ((x1 + x2) * terrain.horizontal_scale / 2, (y1 + y2) * terrain.horizontal_scale / 2, init_height), percept_height_field


# TerrainList = [pyramid_stairs_terrain, stepping_stones_terrain, corridor_terrain,
#                wave_terrain, obstacles_terrain, traversability_terrain, cliffs_terrain, hard_stepping_stones_terrain,
#                jumping_obstacles_terrain, jumping_stages_terrain, stepping_barriers_terrain]
