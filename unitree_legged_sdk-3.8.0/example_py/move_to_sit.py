#!/usr/bin/python

import sys
import time
import math
import numpy as np
sys.path.append('../lib/python/amd64')
import robot_interface as sdk

def check_joint_angle_sanity(joint_position):
    '''
    Check if the joint angle reported by the robot is correct. 

    This will check:
        1. is the returned joint position within the limit?
        2. Are they not all zeros? (all zeros probably indicates no return data)

    Input:
        joint_position - numpy array of shape (12,). 

    Output:
        True if passed check. 
    '''

    A1_joint_angle_limits = np.array([
        [-1.047,    -0.663,      -2.821],  # Hip, Thigh, Calf Min
        [1.047,     2.966,       -0.837] # Hip, Thigh, Calf Max
    ])

    # check input shape
    assert joint_position.shape == (12,)

    # check is angles within limit
    is_within_limit = np.logical_and(
        joint_position.reshape(4,3) >= A1_joint_angle_limits[0,:],
        joint_position.reshape(4,3) <= A1_joint_angle_limits[1,:]
    ).all()

    # check is angles not all zero
    # already checked by above
    return is_within_limit

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

    udp = sdk.UDP(LOWLEVEL, 8081, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    
    
    target_joint_position = np.array([
      -0.27805507, 1.1002517, -2.7185173,
      0.307049, 1.0857971, -2.7133338,
      -0.263221, 1.138222, -2.7211301,
      0.2618303, 1.1157601, -2.7110581
  ])
    
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
        print(init_joint_angles)
        udp.SetSend(cmd)
        udp.Send()
        time.sleep(0.01)
        if (check_joint_angle_sanity(init_joint_angles)):
            break
    while True:
        motiontime += 1
        udp.Recv()
        udp.GetRecv(state)
        if motiontime in range(int((interpolate_duration + stand_duration) * freq)):
             t1 = time.time()
             print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
             cmd_joint_pos = interpolate_joint_position(
             init_joint_angles,
             target_joint_position,
             motiontime / int(interpolate_duration * freq))
             #print(cmd_joint_pos)
             for j in range (len(target_joint_position)):
                cmd.motorCmd[j].q = cmd_joint_pos[j]
                cmd.motorCmd[j].dq = 0
                cmd.motorCmd[j].Kp = 40
                cmd.motorCmd[j].Kd = 1
             t2 = time.time()
             time.sleep(max(0, 1 / freq - (t2 - t1)))
        if motiontime >= int((interpolate_duration + stand_duration) * freq):
             for j in range (len(target_joint_position)):
                print(1)
                cmd.motorCmd[j].q = target_joint_position[j]
                cmd.motorCmd[j].dq = 0
                cmd.motorCmd[j].Kp = 40
                cmd.motorCmd[j].Kd = 1
        if(motiontime > 10):
             safe.PowerProtect(cmd, state, 1)  
        udp.SetSend(cmd)
        udp.Send()
        
             
