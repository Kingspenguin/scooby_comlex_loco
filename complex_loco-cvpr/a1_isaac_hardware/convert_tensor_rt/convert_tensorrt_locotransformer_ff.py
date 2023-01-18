import os
import sys
import os.path as osp
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch.onnx
import numpy as np
from utils import get_policy_locotrans, get_args, load_cfg
import glob
import time
import torch
import torch.nn as nn
import pickle
from a1_utilities.a1_sensor_process import *


args = get_args()
cfg, cfg_train, logdir = load_cfg(args)

PARAM_PATH = os.path.join(args.logdir, "model_{}.pt".format(args.resume))

# PPO components
cfg_train["policy"]["encoder_params"] = cfg_train["policy"]["student_encoder_params"]
actor_critic = get_policy_locotrans(
    cfg_train["policy"]["encoder_params"]["encoder_type"],
    (39,),
    (307,),
    (12,),
    model_cfg=cfg_train["policy"],
    init_noise_std=0.1
)
actor_critic.load_state_dict(torch.load(PARAM_PATH)["student_ac"])
actor_critic.eval()
actor_critic.to("cuda:0")


class ForwardWrapper(nn.Module):
  def __init__(self, ac) -> None:
    super().__init__()
    self.ac = ac

  def forward(self, input_t):
    rnn_out = self.ac.encoder(
        input_t)
    actions_mean = self.ac.actor(rnn_out)
    return actions_mean


pf = ForwardWrapper(
    actor_critic,
)
pf = pf.half()
print(
    pf(torch.rand(
        39 + 64 * 64 * 6
    ).to("cuda:0").half())
)

BATCH_SIZE = 1
state_dim = 39
visual_dim = (6, 64, 64)
dummy_input = torch.randn(
    np.prod(visual_dim) + state_dim
).to("cuda:0").half()

save_name = "onnx/{}.onnx".format(args.save_name)
torch.onnx.export(
    pf,
    args=(dummy_input),
    f=save_name,
    input_names=["input_t"],
    output_names=["output_t"],
    verbose=False
)
# Multiple output version
# torch.onnx.export(
#     pf,
#     args=(dummy_input, hidden_states),
#     f=save_name,
#     input_names=["x", "h"],
#     output_names=["action", "hidden_states"],
#     verbose=False
# )
