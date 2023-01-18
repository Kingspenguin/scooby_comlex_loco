import numpy as np
from utils import get_recurrent_policy, get_args, load_cfg
import glob
import time
import torch
import pickle
from a1_utilities.a1_sensor_process import *
from a1_utilities.realsense import A1RealSense
from a1_utilities.cpg.cpg_controller import CPGController
from control_loop_execution.cpg_lstm_executor import CPGLSTMExecutor
from control_loop_execution.lstm_cpg_rl_policy_wrapper import LSTMCPGPolicyWrapper
import os
import sys
import os.path as osp
sys.path.append(os.path.join(os.path.dirname(__file__), "."))


if __name__ == "__main__":
  args = get_args()
  cfg, cfg_train, logdir = load_cfg(args)

  SEED_ = 0
  POLICYNAME = "recurrent_1cam"
  EXECUTION_TIME = 8

  data_save_dir = "test_records/"

  if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)

  idx = len(glob.glob(data_save_dir + "*"))

  print("Idx:", str(idx))

  comment = ""

  save_dir_name = data_save_dir + "%03d_%s_snap%s_%dseconds_%s/" % (
      idx, POLICYNAME, args.resume, int(EXECUTION_TIME), comment)

  if not os.path.exists(save_dir_name):
    os.makedirs(save_dir_name)

  robot_controller = CPGController(
      state_save_path=save_dir_name,
      stance_duration=0.2,
      foot_clearance=0.025,
      Kp=60, Kd=0.8
  )
  realsense = A1RealSense(
      save_frames=True,
      save_dir_name=save_dir_name
  )

  PARAM_PATH = os.path.join(args.logdir, "model_{}.pt".format(args.resume))

  get_image_interval = 4  # params['env']['env_build']['get_image_interval']
  # num_action_repeat = params['env']['env_build']['num_action_repeat']

  control_freq = 1 / (
      cfg["env"]["control"]["controlFrequencyInv"] * cfg["sim"]["dt"]
  )
  print(control_freq)
  # PPO components
  cfg_train["policy"]["encoder_params"] = cfg_train["policy"]["student_encoder_params"]
  actor_critic = get_recurrent_policy(
      cfg_train["policy"]["encoder_params"]["encoder_type"],
      (73,),
      (350,),
      (16,),
      model_cfg=cfg_train["policy"],
      init_noise_std=0.1
  )
  actor_critic.load_state_dict(torch.load(PARAM_PATH))
  actor_critic.eval()
  actor_critic.to("cuda:0")
  obs_scale = {
      "base_ang": 1,
      "proj_g": 1,
      "base_ang_vel": cfg["env"]["learn"]["angularVelocityScale"],
      "joint_angle": cfg["env"]["learn"]["dofPositionScale"],
      "joint_vel": cfg["env"]["learn"]["dofVelocityScale"],
      "last_action": 1,
      "foot_contact": 1,
      "com_vel": cfg["env"]["learn"]["linearVelocityScale"],
      "command": np.array([
          cfg["env"]["learn"]["linearVelocityScale"],
          cfg["env"]["learn"]["linearVelocityScale"],
          cfg["env"]["learn"]["angularVelocityScale"]
      ]),
      "phase_info": 1
  }

  # exit()
  policyComputer = LSTMCPGPolicyWrapper(
      actor_critic,
      obs_scale,
      get_image_interval,
      save_dir_name=save_dir_name,
      no_tensor=False,
      state_only=False,
      action_scale=[0.1, 0.25, 0.25],
      phase_scale=0.1,
      use_foot_contact=False,
      use_com_vel=True,
      use_command=True,
      num_hist=2,
  )
  executor = CPGLSTMExecutor(
      realsense,
      robot_controller,
      policyComputer,
      control_freq=control_freq,
      frame_interval=get_image_interval,
      Kp=60, Kd=0.8
  )
  executor.execute(EXECUTION_TIME)
