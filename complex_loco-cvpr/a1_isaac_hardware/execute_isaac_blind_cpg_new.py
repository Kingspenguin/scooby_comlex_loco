import numpy as np
from utils import get_policy, get_args, load_cfg
import glob
import time
import torch
import pickle
from a1_utilities.a1_sensor_process import *
from a1_utilities.realsense import A1RealSense
from a1_utilities.robot_controller import RobotController
from control_loop_execution.main_cpg_executor import CPGExecutor
from control_loop_execution.cpg_rl_policy_wrapper import CPGPolicyWrapper
import os
import sys
import os.path as osp
sys.path.append(os.path.join(os.path.dirname(__file__), "."))


if __name__ == "__main__":
  args = get_args()
  cfg, cfg_train, logdir = load_cfg(args)

  SEED_ = 0
  POLICYNAME = "blind_cpg_new"
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
      update_rate=3,
  )

  PARAM_PATH = os.path.join(args.logdir, "model_{}.pt".format(args.resume))

  get_image_interval = 4  # params['env']['env_build']['get_image_interval']
  # num_action_repeat = params['env']['env_build']['num_action_repeat']

  control_freq = 1 / (
      cfg["env"]["control"]["controlFrequencyInv"] * cfg["sim"]["dt"]
  )
  print(control_freq)
  # PPO components
#   cfg_train["policy"]["encoder_params"] = cfg_train["policy"]["student_encoder_params"]
  actor_critic = get_policy(
      cfg_train["policy"]["encoder_type"],
      (79,),
      (),
      (16,),
      False,
      model_cfg=cfg_train["policy"],
      init_noise_std=0.1
  )
  actor_critic.load_state_dict(torch.load(PARAM_PATH)["ac"])
  # actor_critic.load_state_dict(torch.load(PARAM_PATH))
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
  policyComputer = CPGPolicyWrapper(
      actor_critic,
      obs_scale,
      get_image_interval,
      save_dir_name=save_dir_name,
      no_tensor=False,
      state_only=True,
      action_scale=[0.1, 0.25, 0.25],
      phase_scale=0.1,
      use_foot_contact=False,
      use_com_vel=True,
      use_command=True,
      save_log=False,
      num_hist=2,
      phase_dt = 1 / control_freq,
      Kp=40, Kd=1.0
  )
  executor = CPGExecutor(
      realsense,
      robot_controller,
      policyComputer,
      control_freq=control_freq,
      frame_interval=get_image_interval,
      Kp=40, Kd=1.0
  )
  executor.execute(EXECUTION_TIME)
