# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import wandb
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_ppo_rnn import process_ppo_rnn
import random
import numpy as np
import os
import sys
import os.path as osp
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# print(os.path.join(os.path.dirname(__file__), ".."))


def train():
  seed = cfg_train.get("seed", -1)
  d = {}
  d.update(cfg)
  d.update(cfg_train)
  if not args.test and not args.local:
    if not args.wandb_resume:
      run_id = wandb.util.generate_id()
      with open("run_id.txt", "w") as f:
        f.write(run_id)
    else:
      with open("run_id.txt", "r") as f:
        # f.write(run_id)
        run_id = f.readline()
    print(run_id)
    wandb.init(
        project="MultiViewLocomotion",
        entity="isaac-locomotion",
        name="{}_{}_{}{}_{}".format(
            cfg["env"]["name"],
            cfg["task"]["name"],
            cfg_train["learn"]["agent_name"],
            args.add_label,
            str(seed)
        ),
        group="{}_{}_{}{}".format(
            cfg["env"]["name"],
            cfg["task"]["name"],
            cfg_train["learn"]["agent_name"],
            args.add_label
        ),
        # job_type=env_name,
        id=run_id,
        # resume=args.wandb_resume,
        resume="must" if args.wandb_resume else False,
        config=d
    )

  task, env = parse_task(args, cfg, cfg_train, sim_params)
  ppo = process_ppo_rnn(args, env, cfg_train, logdir)

  ppo_iterations = cfg_train["learn"]["max_iterations"]
  if args.max_iterations > 0:
    ppo_iterations = args.max_iterations

  ppo.run(num_learning_iterations=ppo_iterations,
          log_interval=cfg_train["learn"]["save_interval"])


if __name__ == '__main__':
  set_np_formatting()
  args = get_args()
  cfg, cfg_train, logdir = load_cfg(args)
  sim_params = parse_sim_params(args, cfg, cfg_train)
  set_seed(cfg_train.get("seed", -1),
           cfg_train.get("torch_deterministic", False))
  train()
