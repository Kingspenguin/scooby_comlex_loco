import pickle
import torch.nn as nn
import torch
import time
import glob
from utils import get_recurrent_policy_locotrans, get_args, load_cfg
import numpy as np
import torch.onnx
import os
import sys

args = get_args()
cfg, cfg_train, logdir = load_cfg(args)

PARAM_PATH = os.path.join(args.logdir, "model_{}.pt".format(args.resume))

# PPO components
cfg_train["policy"]["encoder_params"] = cfg_train["policy"]["student_encoder_params"]
cfg_train["policy"]["encoder_params"]["transformer_params"] = [
    [1, 2048], [1, 2048], [1, 2048], [1, 2048], [1, 2048], [1, 2048]
]
actor_critic = get_recurrent_policy_locotrans(
    cfg_train["policy"]["encoder_params"]["encoder_type"],
    (47,),
    (315,),
    (16,),
    model_cfg=cfg_train["policy"],
    init_noise_std=0.1
)
# actor_critic.load_state_dict(torch.load(PARAM_PATH)["student_ac"])
actor_critic.eval()
actor_critic.to("cuda:0")


class ForwardWrapper(nn.Module):
  def __init__(self, ac, obs_dim, h_dim, h_num_layers) -> None:
    super().__init__()
    self.ac = ac
    self.obs_dim = obs_dim
    self.h_dim = h_dim
    self.h_num_layers = h_num_layers

  def forward(self, input_t):
    x = input_t[:self.obs_dim]
    h = input_t[self.obs_dim:]
    x = x.reshape(1, -1)
    h = h.reshape(self.h_num_layers, 1, self.h_dim)
    action, hidden_state = self.ac.act_inference(x, h)
    output_t = torch.cat([
        action.reshape(-1), hidden_state.reshape(-1)
    ], dim=0)
    return output_t


pf = ForwardWrapper(
    actor_critic,
    47 + 64 * 64,
    actor_critic.encoder.rnn_hidden_size,
    actor_critic.encoder.rnn_num_layers,
)
pf = pf.half()
hidden_states = torch.zeros(
    actor_critic.encoder.rnn_num_layers,
    actor_critic.encoder.rnn_hidden_size,
    dtype=torch.float, device="cuda:0"
).half()
# hidden_states = [(
#     torch.zeros(1, actor_critic.hidden_state_size,
#                 dtype=torch.float32, device="cuda:0"),
#     torch.zeros(1, actor_critic.hidden_state_size,
#                 dtype=torch.float32, device="cuda:0")
# )] * actor_critic.hidden_state_num

# masks = torch.zeros((1, 1)).to("cuda:0")  # .half()

# print(
#     pf(torch.rand(
#         47 + 64 * 64 * 1 +
#         actor_critic.encoder.rnn_num_layers * actor_critic.encoder.rnn_hidden_size
#     ).to("cuda:0").half())
# )
c_t = time.time()
for _ in range(1000):
  pf(torch.rand(
      47 + 64 * 64 * 1 +
      actor_critic.encoder.rnn_num_layers * actor_critic.encoder.rnn_hidden_size
  ).to("cuda:0").half())
print("GPU infer:", time.time() - c_t)

actor_critic = get_recurrent_policy_locotrans(
    cfg_train["policy"]["encoder_params"]["encoder_type"],
    (47,),
    (315,),
    (16,),
    model_cfg=cfg_train["policy"],
    init_noise_std=0.1
)
# actor_critic.load_state_dict(torch.load(PARAM_PATH)["student_ac"])
actor_critic.eval()
pf = ForwardWrapper(
    actor_critic,
    47 + 64 * 64,
    actor_critic.encoder.rnn_hidden_size,
    actor_critic.encoder.rnn_num_layers,
).to("cpu")
c_t = time.time()
for _ in range(1000):
  pf(torch.rand(
      47 + 64 * 64 * 1 +
      actor_critic.encoder.rnn_num_layers * actor_critic.encoder.rnn_hidden_size
  ).to("cpu"))
print("CPU infer:", time.time() - c_t)

# BATCH_SIZE = 1
# state_dim = 47
# visual_dim = (1, 64, 64)
# dummy_input = torch.randn(
#     np.prod(visual_dim) + state_dim +
#     actor_critic.encoder.rnn_num_layers * actor_critic.encoder.rnn_hidden_size
# ).to("cuda:0").half()

# save_name = "onnx/locotransformer_1cam_1.onnx"
# torch.onnx.export(
#     pf,
#     args=(dummy_input),
#     f=save_name,
#     input_names=["input_t"],
#     output_names=["output_t"],
#     verbose=False
# )
# # Multiple output version
# # torch.onnx.export(
# #     pf,
# #     args=(dummy_input, hidden_states),
# #     f=save_name,
# #     input_names=["x", "h"],
# #     output_names=["action", "hidden_states"],
# #     verbose=False
# # )
