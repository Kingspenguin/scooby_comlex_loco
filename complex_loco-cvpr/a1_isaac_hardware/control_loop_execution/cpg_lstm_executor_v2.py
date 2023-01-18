"""Main Executor"""
import time
import numpy as np
import torch
from a1_utilities.realsense import A1RealSense
from a1_utilities.a1_sensor_process import *
from a1_utilities.InformationSaver import InformationSaver
from a1_utilities.a1_sensor_process import \
    convert_order_from_ros_to_isaac, prepare_high_level_cmd, prepare_position_cmd
from a1_utilities.predefined_pose import move_to_sit, move_to_stand
from control_loop_execution.main_executor import Executor


class CPGLSTMExecutorV2(Executor):
  def __init__(
      self,
      realsense_device,
      robot_controller,
      policy,
      use_high_command=False,
      control_freq=50, frame_interval=4, Kp=60, Kd=0.8
  ):
    super().__init__(
        realsense_device,
        robot_controller,
        policy,
        use_high_command,
        control_freq, frame_interval, Kp, Kd
    )

  def main_execution(self, execution_time):
    count = 0
    if hasattr(self.policy.pf, "cuda_cxt"):
      self.policy.pf.cuda_cxt.push()

    self.control_step = 0
    main_start_time = time.time()

    while True:
      start_time = time.time()
      # Get observation
      robot_observation = self.robot_controller.get_observation()
      com_vel = self.robot_controller.com_vel
      # Get frame every time
      curr_frame = self.realsense_device.get_depth_frame()
      # print(curr_frame)
      # compute next action
      action, self.hidden_states, phase_time, current_joint_angle = self.policy.get_action(
          robot_observation, com_vel,
          self.hidden_states, self.mask,
          curr_frame, self.depth_scale,
          self.last_action
      )
      # print(self.hidden_states)
      # print(action)
      self.last_action = action
      # prepare command
      if self.use_high_command:
        command = prepare_high_level_cmd(action)
      else:
        # command = prepare_position_cmd(target_joint_pos, self.Kp, self.Kd)
        # print(action)
        # print(target_joint_pos)
        command = action
      self.robot_controller.set_cpg_action(
        command, phase_time, current_joint_angle
      )

      end_time = time.time()
      # control loop frequency
      count += 1

      delay = end_time - start_time
      delay_time = 1 / self.control_freq - delay
      print("Delay:", delay)
      # time.sleep(max(0, delay_time))

      current_time = time.time()
      print("Sleep:", (self.control_step + 1) / self.control_freq -
            (current_time - main_start_time)
            )
      time.sleep(
          max(0, (self.control_step + 1) / self.control_freq -
              (current_time - main_start_time)
              )
      )
      self.control_step += 1

      if end_time - main_start_time > execution_time:
        break

    if hasattr(self.policy.pf, "cuda_cxt"):
      self.policy.pf.cuda_cxt.pop()

  def warmup_observations(self):
    # FIXME:
    self.last_action = np.zeros(16)
    # Fill sensor history buffer with observation
    for i in range(3):  # sensor history buffer have length 3 at most
      observation = self.robot_controller.get_observation()
      com_vel = self.robot_controller.com_vel
      if not self.policy.vis_only:
        # joint angle
        joint_angle_hist_normalized = self.policy.joint_angle_historical_data.record_and_normalize(
            # observation_to_joint_position(observation)
            convert_order_from_ros_to_isaac(np.array(observation.q))
        )
      time.sleep(0.05)

    # read one frame
    self.curr_frame = self.realsense_device.get_depth_frame()
    self.hidden_states = [(
        torch.zeros(1, self.policy.pf.hidden_state_size,
                    dtype=torch.float, device="cuda:0"),
        torch.zeros(1, self.policy.pf.hidden_state_size,
                    dtype=torch.float, device="cuda:0")
    )] * self.policy.pf.hidden_state_num
    self.mask = torch.as_tensor([0.0], device="cuda:0", dtype=torch.float)
    for i in range(self.frame_extract * 3 + 1):
        # fill action
      action, self.hidden_states, phase_time, current_joint_angle = self.policy.get_action(
          self.robot_controller.get_observation(), com_vel,  # phase_info,
          self.hidden_states, self.mask, self.curr_frame, self.depth_scale,
          self.last_action
      )  # force frame update

      # last_action_normalized = self.policy.last_action_historical_data.record_and_normalize(
      #     self.last_action
      # )
      self.last_action = action
      time.sleep(0.05)
    print("Policy thread initialization done!")

  def execute(self, execution_time):
    self.realsense_device.start_thread()
    self.robot_controller.start_thread()
    time.sleep(10)
    self.depth_scale = self.realsense_device.get_depth_scale()
    move_to_stand(self.robot_controller)
    self.warmup_observations()
    # self.robot_controller.reset_estimator(1)
    time.sleep(1)
    self.policy.set_cpg_mode(True)
    # self.policy.phase_time = 0
    self.main_execution(execution_time)
    self.policy.set_cpg_mode(False)
    # Get robot to sitting position
    move_to_sit(self.robot_controller)
    # Terminate all processes
    self.realsense_device.stop_thread()
    self.robot_controller.stop_thread()
    self.policy.write()
