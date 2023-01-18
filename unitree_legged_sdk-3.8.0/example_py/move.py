#!/usr/bin/python

import sys
import time
import math
import numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(LOWLEVEL, 8081, "192.168.123.10", 8007)
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
    Kp = 40
    Kd = 0.4
    
    
    
        # print(motiontime)
        # print(state.imu.rpy[0])
        
    
    freq = 400
    while True:
        time.sleep(0.002)
        udp.Recv()
        udp.GetRecv(state)
        init_joint_angles = np.zeros(12, dtype=np.float32)
        for i in range(len(init_joint_angles)):
    	    init_joint_angles[i]=state.motorState[i].q
        print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        time.sleep(0.01)
        #if (check_joint_angle_sanity(init_joint_angles)):
            #break

   
             
