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
    while True:
        time.sleep(0.002)
        motiontime += 1

        # print(motiontime)
        # print(state.imu.rpy[0])
        
        
        udp.Recv()
        udp.GetRecv(state)
        print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        if( motiontime >= 500 and motiontime <=2000):
            speed = 2 * math.sin(3*math.pi*Tpi/2000.0)+2
            torque = (0 - state.motorState[d['FL_2']].q)*10.0 + (0 - state.motorState[d['FL_2']].dq)*1.0
            torque = np.fmin(np.fmax(torque, -5.0), 5.0)
            # cmd.motorCmd[d['FL_2']].q = PosStopF
            cmd.motorCmd[d['FL_2']].q = -1.7
            cmd.motorCmd[d['FL_2']].dq = 1
            cmd.motorCmd[d['FL_2']].Kp = 0
            cmd.motorCmd[d['FL_2']].Kd = 4
            cmd.motorCmd[d['FL_2']].tau =0.0

            Tpi += 1
        if( motiontime >= 2000):
            speed = 2 * math.sin(3*math.pi*Tpi/2000.0)+2
            torque = (0 - state.motorState[d['FL_2']].q)*10.0 + (0 - state.motorState[d['FL_2']].dq)*1.0
            torque = np.fmin(np.fmax(torque, -5.0), 5.0)
            # cmd.motorCmd[d['FL_2']].q = PosStopF
            cmd.motorCmd[d['FL_2']].q = -1.7
            cmd.motorCmd[d['FL_2']].dq = 0
            cmd.motorCmd[d['FL_2']].Kp = 0
            cmd.motorCmd[d['FL_2']].Kd = 4
            cmd.motorCmd[d['FL_2']].tau =0.0
        
        if(motiontime > 10):
            safe.PowerProtect(cmd, state, 1)
        
        udp.SetSend(cmd)
        udp.Send()
