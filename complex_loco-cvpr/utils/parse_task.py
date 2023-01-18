# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# from tasks.a1 import A1
from tasks.robot import Robot
from tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython

from utils.config import warn_task_name

# from isaacgym import rlgpu
from utils.config import warn_task_name

import json


def parse_task(args, cfg, cfg_train, sim_params):

  # create native task and pass custom config
  device_id = args.device_id
  rl_device = args.rl_device
  sync_frame_time = args.sync_frame_time

  cfg["seed"] = cfg_train.get("seed", -1)
  cfg_task = cfg["env"]
  cfg_task["seed"] = cfg["seed"]
  if args.get_statistics:
    cfg["testing"] = True
    cfg["test_episodes"] = args.test_episodes
    cfg["statistics_logdir"] = args.statistics_logdir
    cfg["test_id"] = args.test_id
  if args.local:
    cfg_task["numEnvs"] = 1
    if "terrain" in cfg:
      cfg["terrain"]["env_rows"] = 1
      cfg["terrain"]["env_cols"] = 1
  if args.task_type == "C++":
    if args.device == "cpu":
      print("C++ CPU")
      task = create_task_cpu(args.task, json.dumps(cfg_task))
      if not task:
        warn_task_name()
      if args.headless:
        task.init(device_id, -1, args.physics_engine, sim_params)
      else:
        task.init(device_id, device_id,
                  args.physics_engine, sim_params)
      env = VecTaskCPU(task, rl_device, sync_frame_time, cfg_train.get(
          "clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
    else:
      print("C++ GPU")

      task = create_task_gpu(args.task, json.dumps(cfg_task))
      if not task:
        warn_task_name()
      if args.headless:
        task.init(device_id, -1, args.physics_engine, sim_params)
      else:
        task.init(device_id, device_id,
                  args.physics_engine, sim_params)
      env = VecTaskGPU(task, rl_device, cfg_train.get(
          "clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))

  elif args.task_type == "Python":
    print("Python")

    try:
      terrain_params = None
      if "terrain" in cfg:
        terrain_params = cfg["terrain"]
      task = eval(args.task)(
          cfg=cfg,
          sim_params=sim_params,
          physics_engine=args.physics_engine,
          device_type=args.device,
          device_id=device_id,
          headless=args.headless,
          test_mode=args.test,
          log_video=args.log_video,
          terrain_params=terrain_params,
          tracking_cam_viz=args.tracking_cam_viz,
          get_statistics=args.get_statistics,
      )
    except NameError as e:
      print(e)
      warn_task_name()
    env = VecTaskPython(task, rl_device)

  return task, env
