from re import L
import pybullet as p
import time
import pybullet_data as pd
import numpy as np
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
dt = 1. / 1000.

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.loadURDF("plane.urdf")
robot = p.loadURDF("a1/a1.urdf", [0, 0, 0.5])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -9.8)

A1_DEFAULT_ABDUCTION_ANGLE = 0
A1_DEFAULT_HIP_ANGLE = 0.9
A1_DEFAULT_KNEE_ANGLE = -1.8
NUM_LEGS = 4
INIT_MOTOR_ANGLES = np.array([
    A1_DEFAULT_ABDUCTION_ANGLE,
    A1_DEFAULT_HIP_ANGLE,
    A1_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
motor_ids = []

for j in range(p.getNumJoints(robot)):
  joint_info = p.getJointInfo(robot, j)
  name = joint_info[1].decode('utf-8')
  print("joint_info[1]=", name)
  if name in MOTOR_NAMES:
    motor_ids.append(j)


control_step = 0


def reset():
  for index in range(12):
    joint_id = motor_ids[index]
    p.setJointMotorControl2(
        robot, joint_id, p.POSITION_CONTROL, INIT_MOTOR_ANGLES[index])
    p.resetJointState(robot, joint_id, INIT_MOTOR_ANGLES[index])
    control_step = 0


traj = np.load("traj_2_a1.npz")
print(traj.files)

names = ['fr_hip', 'fr_upper', 'fr_lower', 'fl_hip', 'fl_upper', 'fl_lower',
         'rr_hip', 'rr_upper', 'rr_lower', 'rl_hip', 'rl_upper', 'rl_lower']

print("motor_ids=", motor_ids)

reset()

last_reset_time = time.time()
control_interval = 1 / 20.
# last_control_time =
last_control_time = time.time()
reset_interval = 5

total_seq = 89


def set_command(c_id):
  c_id = c_id % total_seq
  for index in range(12):
    joint_id = motor_ids[index]
    command = traj[names[index]][c_id]
    # p.setJointMotorControl2(
    #     robot, joint_id, p.POSITION_CONTROL, command, positionGain=40, velocityGain=0.4)
    p.resetJointState(robot, joint_id, command)
    # control_step = 0


while p.isConnected():
  if time.time() - last_reset_time > reset_interval:
    reset()
  if time.time() - last_control_time > control_interval:
    set_command(control_step)
    control_step += 1

  p.stepSimulation()
  time.sleep(dt)
