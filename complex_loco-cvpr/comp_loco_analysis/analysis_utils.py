from cmx import doc
from pathlib import Path

import pandas as pd
import wandb

import matplotlib.pyplot as plt
import numpy as np


def get_wandb_data(group_name, entity, project):
  api = wandb.Api()
  runs = api.runs(
      entity + "/" + project,
      filters={"group": group_name}
  )

  summary_list, config_list, name_list, history_list = [], [], [], []

  for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {
            k: v for k, v in run.config.items()
            if not k.startswith('_')
        }
    )
    # .name is the human-readable name of the run.
    name_list.append(run.name)
    history_list.append(run.history())

  return summary_list, config_list, name_list, history_list


def smooth_data(array, smoth_para=5):
  new_array = []
  for i in range(len(array)):
    if i < len(array) - smoth_para:
      new_array.append(np.mean(array[i:i + smoth_para]))
    else:
      new_array.append(np.mean(array[i:None]))
  return new_array


def smooth_curve():
  min_step_number = 1000000000000
  step_number = []
  all_scores = {}

  for seed in args.seed:
    file_path = os.path.join(
        args.log_dir, exp_name, env_name, str(seed), 'log.csv')

    all_scores[seed] = []
    temp_step_number = []
    with open(file_path, 'r') as f:
      csv_reader = csv.DictReader(f)
      for row in csv_reader:
        all_scores[seed].append(float(row[args.entry]))
        temp_step_number.append(int(row["Total Frames"]))

    if temp_step_number[-1] < min_step_number:
      min_step_number = temp_step_number[-1]
      step_number = temp_step_number

    all_mean = []
    all_upper = []
    all_lower = []

    step_number = np.array(step_number) / 1e6
    final_step = []
    for i in range(len(step_number)):
      if args.max_m is not None and step_number[i] >= args.max_m:
        continue
      final_step.append(step_number[i])
      temp_list = []
      for key, valueList in all_scores.items():
        try:
          temp_list.append(valueList[i])
        except Exception:
          print(i)
          # exit()
      all_mean.append(np.mean(temp_list))
      all_upper.append(np.mean(temp_list) + np.std(temp_list))
      all_lower.append(np.mean(temp_list) - np.std(temp_list))
    # print(exp_tag, np.mean(all_mean[-10:]))
    all_mean = post_process(all_mean)
    all_lower = post_process(all_lower)
    all_upper = post_process(all_upper)
