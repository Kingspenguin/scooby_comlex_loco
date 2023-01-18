import torch
import torch.nn as nn
from rl_pytorch.encoder import init
import numpy as np
from .init import weight_init


class MLPBase(nn.Module):
  def __init__(
          self,
          input_shape,
          hidden_shapes,
          activation_func=nn.ReLU,
          init_func=init.basic_init,
          add_ln=False,
          last_activation_func=None):
    super().__init__()

    self.activation_func = activation_func
    self.fcs = []
    self.add_ln = add_ln
    if last_activation_func is not None:
      self.last_activation_func = last_activation_func
    else:
      self.last_activation_func = activation_func
    input_shape = np.prod(input_shape)

    self.output_shape = input_shape
    for next_shape in hidden_shapes:
      fc = nn.Linear(input_shape, next_shape)
      init_func(fc)
      self.fcs.append(fc)
      self.fcs.append(activation_func())
      if self.add_ln:
        self.fcs.append(nn.LayerNorm(next_shape))
      input_shape = next_shape
      self.output_shape = next_shape

    self.fcs.pop(-1)
    self.fcs.append(self.last_activation_func())
    self.seq_fcs = nn.Sequential(*self.fcs)

  def forward(self, x):
    return self.seq_fcs(x)


class RLProjection(nn.Module):
  def __init__(self, in_dim, out_dim, proj=True, proj_function=nn.ReLU):
    super().__init__()
    self.out_dim = out_dim
    module_list = [
        nn.Linear(in_dim, out_dim)
    ]
    if proj:
      module_list += [
          proj_function()
      ]

    self.projection = nn.Sequential(
        *module_list
    )
    self.output_dim = out_dim
    self.apply(weight_init)

  def forward(self, x):
    return self.projection(x)


class MLPHeightEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()

    self.height_dim = cfg["height_dim"]
    self.state_dim = cfg["state_dim"]

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    self.proprio_base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        ** kwargs
    )

    self.height_base = MLPBase(
        input_shape=cfg["height_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        **kwargs
    )

    self.overall_proj = RLProjection(
        in_dim=2 * cfg["hidden_dims"][-1],
        out_dim=2 * cfg["hidden_dims"][-1],
        proj_function=self.non_linear_func,
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x, height_x = x[..., :self.state_dim], x[..., self.state_dim:]
    # heig_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    height_out = self.height_base(height_x)
    state_out = self.proprio_base(state_x)
    out = self.overall_proj(torch.cat([state_out, height_out], dim=-1))
    return out


class MLPMultiHeightEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()

    self.dense_height_dim = cfg["dense_height_dim"]
    self.sparse_height_dim = cfg["sparse_height_dim"]
    self.state_dim = cfg["state_dim"]

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    self.proprio_base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        ** kwargs
    )

    self.dense_height_base = MLPBase(
        input_shape=cfg["dense_height_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        **kwargs
    )

    self.sparse_height_base = MLPBase(
        input_shape=cfg["sparse_height_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        **kwargs
    )

    self.overall_proj = RLProjection(
        in_dim=3 * cfg["hidden_dims"][-1],
        out_dim=2 * cfg["hidden_dims"][-1],
        proj_function=self.non_linear_func,
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x = x[..., :self.state_dim]
    dense_height_x = x[
        ..., self.state_dim:self.state_dim + self.dense_height_dim
    ]
    sparse_height_x = x[..., self.state_dim + self.dense_height_dim:]
    # heig_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    dense_height_out = self.dense_height_base(dense_height_x)
    sparse_height_out = self.sparse_height_base(sparse_height_x)
    state_out = self.proprio_base(state_x)
    out = self.overall_proj(torch.cat([
        state_out, dense_height_out, sparse_height_out
    ], dim=-1))
    return out


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)
