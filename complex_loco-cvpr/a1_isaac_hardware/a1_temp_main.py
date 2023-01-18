from a1_utilities.cpg.gait_generator import GaitGenerator
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    g = GaitGenerator("cuda:0")

    g.reset()
    g.update(0.0,torch.tensor([0,0,0,0], device="cuda:0"))

    N = 1000
    dt = 0.001
    phase_log = torch.zeros((4, N), device="cuda:0")
    for i in range(N):
        g.update(i * dt, torch.tensor([0,0,0,0], device="cuda:0"))
        phase_log[:,i] = g.normalized_phase 
    
    plt.plot(phase_log[:,:].cpu().numpy().transpose())
    plt.show()

    fig, axs = plt.subplots(4, 1)
    for i in range(4):
        axs[i].plot(phase_log[i,:].cpu().numpy().transpose())
    
    plt.show()