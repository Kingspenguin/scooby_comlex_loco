from robot_interface import RobotInterface  # pytype: disable=import-error
from a1_utilities.predefined_pose import *
from a1_utilities.robot_controller import *
from a1_utilities.cpg.A1_inverse_kinematics import *
from a1_utilities.a1_sensor_process import convert_order_from_isaac_to_ros
from a1_utilities.cpg.cpg_controller_old import CPGController

import numpy as np
import time

command_relay = RobotController(control_freq=400)
command_relay.start_thread()
time.sleep(1.0)

print("Move To Stand")
move_to_stand(command_relay)
print("Stand")


traj = np.load("traj_2_a1_may.npz")
print(traj.files)

names = ['fr_hip', 'fr_upper', 'fr_lower', 'fl_hip', 'fl_upper', 'fl_lower',
         'rr_hip', 'rr_upper', 'rr_lower', 'rl_hip', 'rl_upper', 'rl_lower']

actions = []
for n in names:
  actions.append(traj[n].reshape((-1, 1)))


actions = np.hstack(actions)

# print(actions[0])
# print(actions[:10])
# exit()

for pose in actions[5:]:
  move_to_pose(command_relay, pose, 0.045, 0.05)

time.sleep(0.5)
move_to_sit(command_relay)

command_relay.stop_thread()
