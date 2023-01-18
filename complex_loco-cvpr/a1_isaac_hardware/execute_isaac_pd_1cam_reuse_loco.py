import numpy as np
# from utils import get_policy, get_args, load_cfg
from utils.config import get_args, load_cfg
from utils_ac import get_policy_reuse_distill
import glob
import time
import torch
import pickle
from a1_utilities.a1_sensor_process import *
from a1_utilities.realsense import A1RealSense
from a1_utilities.robot_controller import RobotController
from control_loop_execution.main_executor import Executor
from control_loop_execution.rl_policy_wrapper import PolicyWrapper
import os
import sys
import os.path as osp
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


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

  PARAM_PATH = os.path.join(args.logdir, "model_{}.pt".format(args.resume))

  get_image_interval = 4

  control_freq = 1 / (
      cfg["env"]["control"]["controlFrequencyInv"] * cfg["sim"]["dt"]
  )
  print(control_freq)

  cfg_train["policy"]["encoder_params"] = cfg_train["policy"]["student_encoder_params"]
  actor_critic = get_policy_reuse_distill(
      cfg_train["policy"]["encoder_params"]["encoder_type"],
      (39,),
      (347,),
      (12,),
      model_cfg=cfg_train["policy"],
      init_noise_std=0.1
  )
  actor_critic.load_state_dict(torch.load(PARAM_PATH)["student_ac"])
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
  }

  policyComputer = PolicyWrapper(
      actor_critic,
      obs_scale,
      get_image_interval,
      save_dir_name=save_dir_name,
      no_tensor=False,
      state_only=False,
      default_joint_angle=[0, 0.9, -1.7],
      action_range=cfg["env"]["control"]["actionScale"],
      #   use_foot_contact=False,
      #   use_com_vel=True,
      #   use_command=True,
      save_log=False,
      sliding_frames=True,
      num_hist=cfg["env"]["sensor"]["historical_step"],
      use_inverse_depth=cfg["env"]["use_inverse_depth"],
      use_original_depth=cfg["env"]["use_original_depth"]
  )
  executor = Executor(
      realsense,
      robot_controller,
      policyComputer,
      control_freq=control_freq,
      frame_interval=get_image_interval,
      Kp=cfg["env"]["control"]["stiffness"],
      Kd=cfg["env"]["control"]["damping"]
  )
  executor.execute(EXECUTION_TIME)
