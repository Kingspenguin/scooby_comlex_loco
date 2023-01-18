import numpy as np


def depth_process(depth):
  depth[depth < 1e-3] = +np.inf
  # depth[depth > 5] = 0.0
  depth = np.clip(depth, a_min=0.1, a_max=3)
  # print("depth", depth)
  # print(np.min(depth), np.max(depth))
  # depth = depth * -1.0
  depth = np.sqrt(np.log(depth + 1))
  # depth = np.sqrt(1 / (3 * depth))
  # print("after log", depth)
  return depth


a = np.ones(10).astype(np.float32) * 1
a = depth_process(a) / (1 + 1e-4)
print(a)
