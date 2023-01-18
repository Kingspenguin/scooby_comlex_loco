import json
import sys
import os
import numpy as np


logdir = "final_stats"

methods = [
    "teachers",
    "natcnn_bc",
    "natmst_bc",
    "natcnn_rnnbc",
    "nattrans_bc",
    "natvoxel_bc_17_12",
    # "natvoxts_bc",
    # "natvoxel_bcaux_001",
    # "natvoxts_bcaux_001"
]
method_name_mapping = {
    "teachers": "Teacher",
    "natcnn_bc": "NaiveCNN",
    "natmst_bc": "NaiveCNN-MS",
    "natcnn_rnnbc": "NaiveCNN-RNN",
    "nattrans_bc": "LocoTransformer",
    "natvoxel_bc_17_12": "NVM",
    # "natvoxts_bc": "NVMTS",
    # "natvoxel_bcaux_001": "NVMSL001",
    # "natvoxts_bcaux_001": "NVMTSSL_001",
    # "cpg": "CPG"
}

envs = {
    "cliffs": "Stages",
    "pyramid_stairs": "Stairs",
    "stepping_stones": "Stepping Stones",
    "obstacles": "Obstacles",
    # "wave": "Waves"
}

latex_header = '''
'''

latex_header += '''
& \\multicolumn{4}{c}{Traversing Rate (\%) $\\uparrow$ }  & \\multicolumn{4}{c}{Success Rate (\%) $\\uparrow$ }\\\\
\\midrule
'''

latex_header += "Scenarios"
for e in envs:
  latex_header += "& {}".format(envs[e])

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
    print(data)
    reachable_distance_ratio_mean, reachable_distance_ratio_std = data[
        "reachable_distance_ratio"]
    print(m, reachable_distance_ratio_mean, reachable_distance_ratio_std,
          np.round(reachable_distance_ratio_mean * 100, 1))
    if method_name_mapping[m] == "NVM" or method_name_mapping[m] == "Teacher":
      latex_header += "& $\\mathbf{{{} \scriptstyle{{\pm {} }}}}$ ".format(
          np.round(reachable_distance_ratio_mean * 100, 1),
          np.round(reachable_distance_ratio_std * 100, 1)
      )
    else:
      latex_header += "& ${} \scriptstyle{{\pm {} }}$ ".format(
          np.round(reachable_distance_ratio_mean * 100, 1),
          np.round(reachable_distance_ratio_std * 100, 1)
      )

  for e in envs:
    file_name = os.path.join(path, "{}.json".format(e))
    with open(file_name) as f:
      data = json.load(f)
    episode_return_mean, episode_return_std = data[
        "traverse_success_ratio"]
    # episode_return_mean = float()

    if method_name_mapping[m] == "NVM" or method_name_mapping[m] == "Teacher":
      latex_header += "& $\\mathbf{{{} \scriptstyle{{\pm {} }}}}$ ".format(
          np.round(episode_return_mean * 100, 1),
          np.round(episode_return_std * 100, 1)
      )
    else:
      latex_header += "& ${} \scriptstyle{{\pm {} }}$ ".format(
          np.round(episode_return_mean * 100, 1),
          np.round(episode_return_std * 100, 1)
      )
    # latex_header += "& ${} \scriptstyle{{\pm {} }}$ ".format(
    #     np.round(episode_return_mean * 100, 1),
    #     np.round(episode_return_std * 100, 1)
    # )

  latex_header += "\\\\\n"

print(latex_header)


latex_header += "\n"

# latex_header += '''
# \\multicolumn{6}{c}{Success Rate (\%) $\\uparrow$ }\\\\
# \\midrule
# '''


# for m in methods:
#   path = os.path.join(logdir, m)
#   latex_header += "{}".format(
#       method_name_mapping[m])
#   for e in envs:
#     file_name = os.path.join(path, "{}.json".format(e))
#     with open(file_name) as f:
#       data = json.load(f)

#     episode_return_mean, episode_return_std = data[
#         "traverse_success_ratio"]
#     # episode_return_mean = float()
#     latex_header += "& ${} \scriptstyle{{\pm {} }}$ ".format(
#         np.round(float(episode_return_mean) * 100, 1),
#         np.round(float(episode_return_std) * 100, 1)
#     )

#   latex_header += "\\\\\n"


# print(latex_header)
