#!/usr/bin/python

import sys
import time
import math
import numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk
def interpolate_joint_position(pos_1, pos_2, p):
    '''
    Interpolate between joint position 1 and joint position 2

    Input:
        pos_1 - numpy array of shape (12), contains initial joint position in radians.
        pos_2 - numpy array of shape (12), contains end joint position in radians.
        p - interpolate coefficient, a number in [0.0,1.0]. 
            0.0 represents initial position, 1.0 represents end position.
            number in between will output linear combination of the positions.

    Output:
        numpy array of shape (12), interpolated joint angles. 
    '''

    # check size
    assert pos_1.shape == (12,)
    assert pos_2.shape == (12,)

    # constrain p between 0.0 and 1.0
    p = min(1.0,p)
    p = max(0.0,p)

    ## TODO cautious!!! Do we need to worry about angle crossover? 
    return (1.0 - p) * pos_1 + p * pos_2


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(LOWLEVEL, 8084, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    
    
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
    init_joint_angles = np.zeros(12, dtype=np.float32)
    
    
        # print(motiontime)
        # print(state.imu.rpy[0])
        
    
    freq = 400
    while True:
        udp.Recv()
        udp.GetRecv(state)
        for i in range(len(init_joint_angles)):
    	     init_joint_angles[i]=state.motorState[i].q
    	     print (init_joint_angles)
        time.sleep(0.01)
        motiontime = motiontime + 1

       
             
