import json
import sys
import os
import numpy as np


logdir = "teacher_stats_v149"

methods = ["teachers"]
method_name_mapping = {
    "teachers": "Teachers",
    # "multi_task": "Multi-Task",
    # "pd_joint": "AMP",
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
