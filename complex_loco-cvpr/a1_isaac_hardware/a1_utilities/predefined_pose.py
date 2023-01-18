import numpy as np
from a1_utilities.a1_sensor_process import *
import time


def move_to_pose(
  robot_controller, 
  target_joint_position, 
  interpolate_duration=0.1,
  after_duration=0.02):
  '''
  Blocking command that takes the robot to standing position by sending command to command relay object.
  '''
  # hard coded parameters for simplicity of use
  # target_joint_position = np.array(([
  #     0.0, 0.9, -1.7,
  #     0.0, 0.9, -1.7,
  #     0.0, 0.9, -1.7,
  #     0.0, 0.9, -1.7
  # ]))
  # interpolate_duration = 2.0
  # stand_duration = 1.0
  Kp = 60
  Kd = 5.0
  freq = robot_controller.control_freq

  # receive initial position
  init_joint_angles = np.zeros(12, dtype=np.float32)

  # if not robot_controller.use_high_level_command:
  #   while True:
  #     # send zero torque command in order to receive observation
  observation = robot_controller.get_observation()
      # print(observation)

      # init_joint_angles = convert_order_from_isaac_to_ros(
      #     observation_to_joint_position_isaac(observation))
  init_joint_angles = np.array(observation.q)
      # time.sleep(0.01)

      # if (check_joint_angle_sanity(init_joint_angles)):
      #   break

  # interpolate to target position and stand
  for i in range(int((interpolate_duration + after_duration) * freq)):
    t1 = time.time()

    # cmd_joint_pos = interpolate_joint_position(
    #     init_joint_angles,
    #     target_joint_position,
    #     i / int(interpolate_duration * freq)
    # )

    # cmd = prepare_position_cmd(
    #     cmd_joint_pos,
    #     Kp,
    #     Kd
    # )
    # cmd = (cmd_joint_pos, Kp, Kd)
    cmd = (target_joint_position, Kp, Kd)

    robot_controller.set_action(cmd)
    if robot_controller.use_high_level_command:
      robot_controller.set_high_action(np.array([0, 0, 0]))

    t2 = time.time()

    time.sleep(max(0, 1 / freq - (t2 - t1)))


def move_to_stand(robot_controller):
  '''
  Blocking command that takes the robot to standing position by sending command to command relay object.
  '''
  # hard coded parameters for simplicity of use
  target_joint_position = np.array(([
      0.0, 0.9, -1.7,
      0.0, 0.9, -1.7,
      0.0, 0.9, -1.7,
      0.0, 0.9, -1.7
  ]))
  interpolate_duration = 2.0
  stand_duration = 1.0
  Kp = 80
  Kd = 0.4
  freq = robot_controller.control_freq

  # receive initial position
  init_joint_angles = np.zeros(12, dtype=np.float32)

  if not robot_controller.use_high_level_command:
    while True:
      # send zero torque command in order to receive observation
      observation = robot_controller.get_observation()
      print(observation)

      # init_joint_angles = convert_order_from_isaac_to_ros(
      #     observation_to_joint_position_isaac(observation))
      init_joint_angles = np.array(observation.q)
      time.sleep(0.01)

      if (check_joint_angle_sanity(init_joint_angles)):
        break

  # interpolate to target position and stand
  for i in range(int((interpolate_duration + stand_duration) * freq)):
    t1 = time.time()

    cmd_joint_pos = interpolate_joint_position(
        init_joint_angles,
        target_joint_position,
        i / int(interpolate_duration * freq)
    )

    # cmd = prepare_position_cmd(
    #     cmd_joint_pos,
    #     Kp,
    #     Kd
    # )
    cmd = (cmd_joint_pos, Kp, Kd)

    robot_controller.set_action(cmd)
    if robot_controller.use_high_level_command:
      robot_controller.set_high_action(np.array([0, 0, 0]))

    t2 = time.time()

    time.sleep(max(0, 1 / freq - (t2 - t1)))


def move_to_sit(robot_controller):
  '''
  Blocking command that takes the robot to standing position by sending command to command relay object.
  '''
  # hard coded parameters for simplicity of use
  target_joint_position = np.array([
      -0.27805507, 1.1002517, -2.7185173,
      0.307049, 1.0857971, -2.7133338,
      -0.263221, 1.138222, -2.7211301,
      0.2618303, 1.1157601, -2.7110581
  ])

  interpolate_duration = 3.0
  stand_duration = 1.0
  Kp = 80
  Kd = 0.4
  freq = robot_controller.control_freq

  # receive initial position
  init_joint_angles = np.zeros(12, dtype=np.float32)

  while True:
    # send zero torque command in order to receive observation
    observation = robot_controller.get_observation()
    # init_joint_angles = convert_order_from_isaac_to_ros(
    #     observation_to_joint_position_isaac(observation))
    init_joint_angles = np.array(observation.q)
    time.sleep(0.01)

    if (check_joint_angle_sanity(init_joint_angles)):
      break

  # interpolate to target position and stand
  for i in range(int((interpolate_duration + stand_duration) * freq)):
    t1 = time.time()

    cmd_joint_pos = interpolate_joint_position(
        init_joint_angles,
        target_joint_position,
        i / int(interpolate_duration * freq)
    )

    # cmd = prepare_position_cmd(
    #     cmd_joint_pos,
    #     Kp,
    #     Kd
    # )
    cmd = (cmd_joint_pos, Kp, Kd)

    robot_controller.set_action(cmd)
    if robot_controller.use_high_level_command:
      robot_controller.set_high_action(np.array([0, 0, 0]))

    t2 = time.time()

    time.sleep(max(0, 1 / freq - (t2 - t1)))
