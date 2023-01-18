import torch
import random
import numpy as np
import yaml
from rl_pytorch.ppo import ActorCritic
from rl_pytorch.encoder import get_encoder
from rl_pytorch.asymmetric_distill_recurrent import RecurrentActorCritic
from rl_pytorch.asymmetric_distill_rnn_lm_locotrans import RecurrentActorCriticLM as RecurrentActorCriticLocoTrans
from rl_pytorch.asymmetric_distill_lm import ActorCriticLM
from rl_pytorch.distill_reuse import ActorCriticLM as ReuseActorCriticLM
import argparse
import os
import time
import sys
import os.path as osp
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

def warn_task_name():
  raise Exception(
      "Unrecognized task!\nTask should be one of: [Robot]")


def set_seed(seed, torch_deterministic=False):
  if seed == -1 and torch_deterministic:
    seed = 42
  elif seed == -1:
    seed = np.random.randint(0, 10000)
  print("Setting seed: {}".format(seed))

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  if torch_deterministic:
    # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

  return seed


def retrieve_cfg(args, use_rlg_config=False):
  if use_rlg_config:
    if args.task == "A1":
      return os.path.join(args.logdir, "a1"), "cfg/train/rlg/rlg_a1.yaml", "cfg/a1.yaml"
    elif args.task == "Robot":
      return os.path.join(args.logdir, "robot"), "cfg/train/rlg/rlg_a1.yaml", "cfg/a1.yaml"
    else:
      warn_task_name()
  else:
    if args.task == "A1":
      return os.path.join(args.logdir, "a1"), "cfg/train/rlpt/pytorch_ppo_a1.yaml", "cfg/a1.yaml"
    elif args.task == "Robot":
      return os.path.join(args.logdir, "robot"), "cfg/train/rlpt/pytorch_ppo_a1.yaml", "cfg/robot.yaml"
    else:
      warn_task_name()


def load_cfg(args, use_rlg_config=False):
  with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
    cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

  with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

  logdir = args.logdir

  return cfg, cfg_train, logdir


def get_args(benchmark=False, use_rlg_config=False):
  custom_parameters = [
      {"name": "--resume", "type": str, "default": "0",
       "help": "Resume training or start testing from a checkpoint"},
      {"name": "--save_name", "type": str, "default": "trt_policy",
       "help": "Resume training or start testing from a checkpoint"},
      {"name": "--logdir", "type": str, "default": "logs/"},
      {"name": "--cfg_train", "type": str,
          "default": "Base"},
      {"name": "--cfg_env", "type": str, "default": "Base"},
      {"name": "--seed", "type": int, "help": "Random seed"},
  ]
  # parse arguments
  parser = argparse.ArgumentParser(
      description="RL Policy")
  parser.add_argument(
      "--logdir", type=str
  )
  parser.add_argument(
      "--save_name", type=str
  )
  parser.add_argument(
      "--cfg_env", type=str
  )
  parser.add_argument(
      "--cfg_train", type=str
  )
  parser.add_argument(
      "--resume", type=str
  )
  args = parser.parse_args()
  # allignment with examples
  # logdir, cfg_train, cfg_env = retrieve_cfg(args, use_rlg_config)

  # use custom parameters if provided by user
  # if args.logdir == "logs/":
  #   args.logdir = logdir

  # if args.cfg_train == "Base":
  #   args.cfg_train = cfg_train

  # if args.cfg_env == "Base":
  #   args.cfg_env = cfg_env

  return args


def get_policy(
    encoder_type,
    observation_shape,
    state_shape,
    action_shape,
    asymmetric,
    model_cfg,
    init_noise_std=0.1
):
  if "student_pi_hid_sizes" in model_cfg:
    model_cfg["pi_hid_sizes"] = model_cfg["student_pi_hid_sizes"]
  policy = ActorCritic(
      get_encoder(encoder_type),
      observation_shape,
      state_shape,
      action_shape,
      init_noise_std,
      model_cfg,
      asymmetric=asymmetric
  )
  return policy


def get_recurrent_policy(
    encoder_type,
    observation_shape,
    state_shape,
    action_shape,
    model_cfg,
    init_noise_std=0.1
):
  if "student_pi_hid_sizes" in model_cfg:
    model_cfg["pi_hid_sizes"] = model_cfg["student_pi_hid_sizes"]
  policy = RecurrentActorCritic(
      get_encoder(encoder_type),
      observation_shape,
      state_shape,
      action_shape,
      init_noise_std,
      model_cfg, asymmetric=True
  )
  policy.hidden_state_size = model_cfg['encoder_params']['recurrent']["hidden_size"]
  policy.hidden_state_num = model_cfg['encoder_params']['recurrent']["num_layers"]
  return policy


def get_recurrent_policy_locotrans(
    encoder_type,
    observation_shape,
    state_shape,
    action_shape,
    model_cfg,
    init_noise_std=0.1
):
  if "student_pi_hid_sizes" in model_cfg:
    model_cfg["pi_hid_sizes"] = model_cfg["student_pi_hid_sizes"]
  policy = RecurrentActorCriticLocoTrans(
      get_encoder(encoder_type),
      observation_shape,
      state_shape,
      action_shape,
      init_noise_std,
      model_cfg, asymmetric=True
  )
  policy.hidden_state_size = model_cfg['encoder_params']['recurrent']["hidden_size"]
  policy.hidden_state_num = model_cfg['encoder_params']['recurrent']["num_layers"]
  return policy


def get_policy_locotrans(
    encoder_type,
    observation_shape,
    state_shape,
    action_shape,
    model_cfg,
    init_noise_std=0.1
):
  if "student_pi_hid_sizes" in model_cfg:
    model_cfg["pi_hid_sizes"] = model_cfg["student_pi_hid_sizes"]
  policy = ActorCriticLM(
      get_encoder(encoder_type),
      observation_shape,
      state_shape,
      action_shape,
      init_noise_std,
      model_cfg, asymmetric=True
  )
  return policy


def get_policy_reuse_distill(
    encoder_type,
    observation_shape,
    state_shape,
    action_shape,
    model_cfg,
    init_noise_std=0.1
):
  if "student_pi_hid_sizes" in model_cfg:
    model_cfg["pi_hid_sizes"] = model_cfg["student_pi_hid_sizes"]
  policy = ReuseActorCriticLM(
      get_encoder(encoder_type),
      observation_shape,
      state_shape,
      action_shape,
      init_noise_std,
      model_cfg, asymmetric=False
  )
  return policy
