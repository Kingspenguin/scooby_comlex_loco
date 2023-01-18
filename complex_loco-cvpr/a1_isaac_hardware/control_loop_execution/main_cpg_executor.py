import threading
import time
import numpy as np
from a1_utilities.realsense import A1RealSense
from a1_utilities.a1_sensor_process import *
from a1_utilities.InformationSaver import InformationSaver

# from control_loop_execution.rl_policy_wrapper import PolicyWrapper
# from a1_utilities.robot_controller import CommandRelay

from a1_utilities.a1_sensor_process import *
from a1_utilities.predefined_pose import move_to_sit, move_to_stand


class CPGExecutor:
  def __init__(
      self,
      realsense_device,
      robot_controller,
      policy,
      # observationGetter,
      # actionSetter,
      # depthFrameGetter,
      use_high_command=False,
      control_freq=50, frame_interval=4, Kp=40, Kd=0.4
  ):
    self.realsense_device = realsense_device
    self.robot_controller = robot_controller
    self.policy = policy
    self.continue_thread = False
    self.control_freq = control_freq
    self.frame_extract = frame_interval

    self.Kp = Kp
    self.Kd = Kd

    self.execution_thread = None

    self.use_high_command = use_high_command

    self.default_cmd = np.zeros((12, 5), dtype=np.float32)
    self.default_cmd[:, 1] = Kp
    self.default_cmd[:, 3] = Kd

  def warmup_observations(self):
    # FIXME:
    self.last_action = np.zeros(16)
    # Fill sensor history buffer with observation
    for i in range(3):  # sensor history buffer have length 3 at most
      observation = self.robot_controller.get_observation()
      com_vel = self.robot_controller.com_vel
      print(observation)
      if not self.policy.vis_only:
        # fill IMU and joint angle sensor buffer
        # joint angle
        joint_angle_hist_normalized = self.policy.joint_angle_historical_data.record_and_normalize(
            # observation_to_joint_position_isaac(observation)
            convert_order_from_ros_to_isaac(np.array(observation.q))
        )
      time.sleep(0.05)

    # read one frame
    self.curr_frame = self.realsense_device.get_depth_frame()
    for i in range(self.frame_extract * 3 + 1):
        # fill action
      action, target_joint_pos = self.policy.get_action(
          self.robot_controller.get_observation(), com_vel,
          self.curr_frame, self.depth_scale,
          self.last_action
      )  # force frame update

      # last_action_normalized = self.policy.last_action_historical_data.record_and_normalize(
      #   self.last_action
      # )
      self.last_action = action.detach().cpu().numpy()
      time.sleep(0.05)
    print("Policy thread initialization done!")

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
      get_start = time.time()
      curr_frame = self.realsense_device.get_depth_frame()
      print("get depth time:", time.time() - get_start)
      # compute next action
      action, target_joint_pos = self.policy.get_action(
          robot_observation, com_vel,
          curr_frame, self.depth_scale,
          self.last_action
      )
      # print(action)
      self.last_action = action.detach().cpu().numpy()

      prepare_start = time.time()
      # prepare command
      if self.use_high_command:
        command = prepare_high_level_cmd(action)
      else:
        # print("Raw Action:", action)
        # command = prepare_position_cmd_with_default(target_joint_pos, self.default_cmd)
        command = (target_joint_pos, self.Kp, self.Kd)
        # print("Raw Target Joint:", target_joint_pos)
        # print("Raw Action:", action)
        # command = prepare_torque_cmd(target_torque)
      self.robot_controller.set_action(command)
      print("prepare time", time.time() - prepare_start)

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

  def start_thread(self):
    print("Start policy thread called")
    self.continue_thread = True
    self.execution_thread = threading.Thread(target=self.main_execution)
    self.execution_thread.start()

  def stop_thread(self):
    print("Stop policy thread called")
    self.continue_thread = False
    # self.policy.write()
    self.execution_thread.join()

  def execute(self, execution_time):
    self.realsense_device.start_thread()
    self.robot_controller.start_thread()
    time.sleep(10)
    self.depth_scale = self.realsense_device.get_depth_scale()
    move_to_stand(self.robot_controller)
    self.warmup_observations()
    time.sleep(2)

    self.policy.set_cpg_mode(True)
    # self.start_thread()
    self.main_execution(execution_time)
    self.policy.set_cpg_mode(False)
    # time.sleep(execution_time) # RUN POLICY FOR TEN SECONDS?
    # self.stop_thread()
    # Get robot to sitting position
    move_to_sit(self.robot_controller)
    # Terminate all processes
    self.realsense_device.stop_thread()
    self.robot_controller.stop_thread()
    self.policy.write()
