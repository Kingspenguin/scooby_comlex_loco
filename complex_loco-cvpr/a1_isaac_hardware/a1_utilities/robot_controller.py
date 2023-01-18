import threading
import time
import numpy as np
from a1_utilities.a1_sensor_process import observation_to_joint_state

from robot_interface import RobotInterface
from a1_utilities.logger import StateLogger


class RobotController(object):
  def __init__(
      self,
      control_freq=400,
      default_action=None,
      use_high_level_command=False,
      save_log=False,
      log_interval=10,
      state_save_path=None,
  ):
    if default_action is None:
      # no output command for robot
      # default_action = np.zeros(60)
      default_action = (np.zeros(12), 0.0, 0.0)
      default_high_action = np.zeros(3)

    self.control_freq = control_freq
    self.control_interval = 1 / self.control_freq
    self.action = default_action
    self.observation = None              # observation buffer

    self.control_thread_enabled = False  # when False, thread_main ends
    self.control_thread = None           # control thread

    # self.high_mode = 0
    self.high_action = default_high_action

    self.use_high_level_command = use_high_level_command

    # robot interface
    if self.use_high_level_command:
      self.robot_interface = RobotInterface(0)
    else:
      self.robot_interface = RobotInterface()

    self.save_log = save_log
    self.log_interval = log_interval
    self.state_save_path = state_save_path
    if save_log:
      self.joint_state_logger = StateLogger(
          np.zeros((36), dtype=np.float64),
          duration=60,
          frequency=400,
          data_save_name=self.state_save_path + "joint_state.npz"
      )

      self.imu_logger = StateLogger(
          np.zeros((10), dtype=np.float64),
          duration=60,
          frequency=400,
          data_save_name=self.state_save_path + "IMU.npz"
      )

      self.command_logger = StateLogger(
          np.zeros((60), dtype=np.float64),
          duration=60,
          frequency=400,
          data_save_name=self.state_save_path + "actual_command.npz"
      )

      self.foot_force_saver = StateLogger(
          np.zeros((4), dtype=np.float64),
          duration=60,
          frequency=400,
          data_save_name=self.state_save_path + "foot_force.npz"
      )

  def start_thread(self):
    self.control_thread_enabled = True
    self.control_thread = threading.Thread(target=self.control_function)
    self.control_thread.start()
    print("Control thread started")

  def stop_thread(self):
    self.control_thread_enabled = False
    if self.save_log:
      self.joint_state_saver.write()
      self.IMU_saver.write()
      self.actual_command_saver.write()
      self.foot_force_saver.write()
    self.control_thread.join()
    print("Control thread stopped")

  def record_state(self):
    joint_state = observation_to_joint_state(self.observation)
    imu_obs = np.hstack((
        np.array(self.observation.imu.accelerometer),
        np.array(self.observation.imu.gyroscope),
        np.array(self.observation.imu.quaternion),
    ))
    self.joint_state_saver.record(joint_state)
    self.IMU_saver.record(imu_obs)
    self.actual_command_saver.record(self.action)
    self.foot_force_saver.record(self.observation.footForce)

  def prepare_command(self):
    pass

  def control_function(self):
    self.control_step = 0
    self.start_time = time.time()
    print("start time", self.start_time)

    while self.control_thread_enabled:
      # single transcation

      self.prepare_command()
      # print("action", self.action)
      if self.use_high_level_command:
        self.robot_interface.send_high_command(self.high_action)
      else:
        # print(self.action)
        # self.robot_interface.send_command(self.action)
        self.robot_interface.send_kp_command(self.action[0], self.action[1], self.action[2])

      # self.update_observation()

      # process observation and save
      if self.save_log & self.control_step & self.log_interval:
        self.record_state()

      current_time = time.time()
      time.sleep(
          max(0, (self.control_step + 1) * self.control_interval -
              (current_time - self.start_time))
      )
      self.control_step += 1

  def update_observation(self):
    self.observation = self.robot_interface.receive_processed_observation()
    self.com_vel = np.zeros(3)

  def get_observation(self):
    # print(self.observation)
    self.update_observation()
    return self.observation

  def set_action(self, action):
    # print(self.action)
    self.action = action

  def set_high_action(self, high_action):
    self.high_action = high_action


if (__name__ == "__main__"):

  command_relay = RobotController(control_freq=400)
  command_relay.start_thread()
  time.sleep(10.0)
  command_relay.stop_thread()
