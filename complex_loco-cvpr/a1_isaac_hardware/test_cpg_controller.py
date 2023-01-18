from robot_interface import RobotInterface  # pytype: disable=import-error
from a1_utilities.predefined_pose import *
from a1_utilities.robot_controller import *
from a1_utilities.cpg.A1_inverse_kinematics import *
from a1_utilities.a1_sensor_process import convert_order_from_isaac_to_ros
from a1_utilities.cpg.cpg_controller import CPGController

import numpy as np
import time

command_relay = RobotController(control_freq=400)
command_relay.start_thread()
time.sleep(1.0)

print("Move To Stand")
move_to_stand(command_relay)

print("Stand")
# time.sleep(20)
command_relay.reset_estimator(1)

command_relay.set_cpg_mode(True)
time.sleep(10)
command_relay.set_cpg_mode(False)

time.sleep(0.5)
move_to_sit(command_relay)

command_relay.stop_thread()
