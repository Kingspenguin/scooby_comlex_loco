import robot_interface
import numpy as np

r = robot_interface.RobotInterface(0)

import time

start_time = time.time()

while True:
    current_time = time.time()
    if current_time - start_time > 3:
        break
    # high_state = r.receive_high_state()
    # print(high_state.imu.temperature)
    # print(high_state.forwardSpeed)
    r.send_high_command(np.array([2.0, 0.3, 0.0]))

r.send_high_command(np.array([0.0, 0.0, 0.0]))