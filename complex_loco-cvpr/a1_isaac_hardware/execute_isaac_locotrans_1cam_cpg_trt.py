import numpy as np
from control_loop_execution.locotrans_trt_policy_wrapper import LocotransTRTPolicyWrapper
from control_loop_execution.cpg_rl_policy_wrapper_trt import CPGTRTPolicyWrapper
from utils import get_recurrent_policy, get_args, load_cfg
import glob
import time
import torch
import pickle
from a1_utilities.a1_sensor_process import *
from a1_utilities.realsense import A1RealSense
from a1_utilities.robot_controller import RobotController
from control_loop_execution.cpg_executor_trt import CPGTRTExecutor
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
  EXECUTION_TIME = 20

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

  robot_controller = RobotController(
      state_save_path=save_dir_name,
      control_freq=400
  )
  realsense = A1RealSense(
      save_frames=True,
      save_dir_name=save_dir_name,
      update_rate=1,
  )

  PARAM_PATH = os.path.join(args.logdir, "locotransformer_1cam_ff_dr_1_fp16.trt")

  get_image_interval = 3  # params['env']['env_build']['get_image_interval']
  # num_action_repeat = params['env']['env_build']['num_action_repeat']

  control_freq = 1 / (
      cfg["env"]["control"]["controlFrequencyInv"] * cfg["sim"]["dt"]
  )
  print(control_freq)
  # PPO components
  cfg_train["policy"]["encoder_params"] = cfg_train["policy"]["student_encoder_params"]
  actor_critic = LocotransTRTPolicyWrapper(
      PARAM_PATH,
      55,
      (6, 64, 64),
      16,
  )
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
  policyComputer = CPGTRTPolicyWrapper(
      actor_critic,
      obs_scale,
      get_image_interval,
      save_dir_name=save_dir_name,
      no_tensor=True,
      state_only=False,
      action_scale=[0.1, 0.25, 0.25],
      phase_scale=0.2,
      use_foot_contact=False,
      save_log=False,
      num_hist=2,
      phase_dt=1 / control_freq,
      Kp=60, Kd=1.0,
      sliding_frames=True
  )
  executor = CPGTRTExecutor(
      realsense,
      robot_controller,
      policyComputer,
      control_freq=control_freq,
      frame_interval=get_image_interval,
      Kp=60, Kd=1.0
  )
  executor.execute(EXECUTION_TIME)
