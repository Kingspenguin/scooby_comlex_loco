import json
import sys
import os
import numpy as np


logdir = "stats_v149_hard_1108"

methods = [
    "teachers", "natcnn_bc", "natvoxel_bc",
    "natvoxts_bc", "natvoxel_bcaux_001", "natvoxts_bcaux_001"
]
method_name_mapping = {
    "teachers": "Teachers",
    "natcnn_bc": "NaiveCNN",
    "natvoxel_bc": "Voxel",
    "natvoxts_bc": "VoxelTS",
    "natvoxel_bcaux_001": "Voxel_SL_001",
    "natvoxts_bcaux_001": "VoxelTS_SL_001",
    # "cpg": "CPG"
}

envs = {
    "cliffs": "Cliffs",
    "pyramid_stairs": "Stairs",
    "stepping_stones": "Stepping Stones",
    "obstacles": "Obstacles",
    "wave": "Waves"
}

latex_header = '''
'''

latex_header += "Scenarios"
for e in envs:
  latex_header += "& {}".format(envs[e])
latex_header += "\\\\\\midrule\n"

for m in methods:
  path = os.path.join(logdir, m)
  latex_header += "{}".format(
      method_name_mapping[m])
  for e in envs:
    file_name = os.path.join(path, "{}.json".format(e))
    with open(file_name) as f:
      data = json.load(f)

    reachable_distance_ratio_mean, reachable_distance_ratio_std = data[
        "reachable_distance_ratio"]
    if m[0] == "multi_task" and m[1] == "cpg":
      latex_header += "& $\\mathbf{{{} \scriptstyle{{\pm {} }}}}$ ".format(np.round(
          reachable_distance_ratio_mean * 100, 1), np.round(reachable_distance_ratio_std * 100, 1))
    else:
      latex_header += "& ${} \scriptstyle{{\pm {} }}$ ".format(np.round(
          reachable_distance_ratio_mean * 100, 1), np.round(reachable_distance_ratio_std * 100, 1))

  latex_header += "\\\\\n"

print(latex_header)
