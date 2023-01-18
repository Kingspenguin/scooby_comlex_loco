import math
from tokenize import group
import torch
import torch.nn as nn
import numpy as np
from rl_pytorch.ppo import init
from rl_pytorch.networks import get_network

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from .impala import ImpalaEncoder
# from .voxel_encoder import ImpalaVoxelEncoder


def get_encoder(encoder_type):
  if encoder_type == "mlp":
    return None
  if encoder_type == "height_mlp":
    return None
  elif encoder_type == "pure_nature":
    return NaturePureEncoder
  elif encoder_type == "nature":
    return NatureFuseEncoder
  elif encoder_type == "nature_add":
    return NatureFuseAddEncoder
  elif encoder_type == "nature_multiply":
    return NatureFuseMultiplyEncoder
  elif encoder_type == "nature_lstm":
    return NatureFuseLSTMEncoder
  elif encoder_type == "nature_lstm_v2":
    return NatureFuseLSTMEncoderV2
  elif encoder_type == "nature_lstm_1cam_multiply":
    return NatureFuseLSTM1CamMultiplyEncoder
  elif encoder_type == "nature_separate_lstm":
    return NatureFuseLSTMSeparateEncoder
  elif encoder_type == "nature_sep":
    return NatureSepFuseEncoder
  elif encoder_type == "nature_att":
    return NatureAttEncoder
  elif encoder_type == "nature_lstm_add":
    return NatureFuseLSTMAddEncoder
  elif encoder_type == "nature_lstm_multiply":
    return NatureFuseLSTMMultiplyEncoder
  elif encoder_type == "nature_lstm_multiply_v2":
    return NatureFuseLSTMMultiplyEncoderV2
  elif encoder_type == "nature_lstm_concat_v2":
    return NatureFuseLSTMConcatEncoderV2
  elif encoder_type == "nature_lstm_all_multiply":
    return NatureFuseLSTMAllMultiplyEncoder
  elif encoder_type == "nature_lstm_multiply_attention_v2":
    return NatureFuseLSTMMultiplyAttentionEncoderV2
  elif encoder_type == "nature_lstm_multiply_all_attention_v2":
    return NatureFuseLSTMMultiplyAllAttentionEncoderV2
  elif encoder_type == "locotransformer_lstm":
    return LocoTransformerLSTMEncoder
  elif encoder_type == "locotransformer":
    return LocoTransformerEncoder
  elif encoder_type == "locotransformer_v2":
    return LocoTransformerV2Encoder
  elif encoder_type == "impala_voxel":
    return ImpalaVoxelEncoder
  else:
    raise NotImplementedError


def weight_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
  elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.orthogonal_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module


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
  def __init__(self, in_dim, out_dim, proj=True):
    super().__init__()
    self.out_dim = out_dim
    module_list = [
        nn.Linear(in_dim, out_dim)
    ]
    if proj:
      module_list += [
          nn.ReLU()
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
    self.height_base = NatureEncoder(
        cfg["in_channels"]
    )

    self.height_dim = cfg["height_dim"]
    self.state_dim = cfg["state_dim"]

    self.proprio_base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.height_base = MLPBase(
        input_shape=cfg["height_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)

    return torch.cat([visual_out, state_out], dim=-1)


class NatureEncoder(nn.Module):
  def __init__(self,
               in_channels,
               groups=1,
               flatten=True,
               **kwargs):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super(NatureEncoder, self).__init__()
    self.groups = groups
    layer_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=32 * self.groups,
                  kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32 * self.groups, out_channels=64 * self.groups,
                  kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64 * self.groups, out_channels=64 * self.groups,
                  kernel_size=3, stride=1), nn.ReLU(),
    ]
    if flatten:
      layer_list.append(
          nn.Flatten()
      )
    self.layers = nn.Sequential(*layer_list)

    self.output_dim = 1024 * self.groups
    self.apply(orthogonal_init)

  def forward(self, x):
    x = x.view(torch.Size(
        [np.prod(x.size()[:-3])]) + x.size()[-3:])
    x = self.layers(x)
    return x


class NaturePureEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NaturePureEncoder, self).__init__()
    self.visual_base = NatureEncoder(
        cfg["in_channels"]
    )

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.hidden_states_shape = cfg["hidden_dims"][-1]

  def forward(self, visual_x):
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = self.visual_projector(visual_out)
    return visual_out


class NatureFuseEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseEncoder, self).__init__()
    self.visual_base = NatureEncoder(
        cfg["in_channels"]
    )

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)

    return torch.cat([visual_out, state_out], dim=-1)


class NatureFuseAddEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseAddEncoder, self).__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_out = visual_head_out + visual_neck_out
    # visual_out = self.visual_base(visual_x)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)

    return torch.cat([visual_out, state_out], dim=-1)


class NatureFuseMultiplyEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseMultiplyEncoder, self).__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_out = visual_head_out * visual_neck_out
    # visual_out = self.visual_base(visual_x)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)

    return torch.cat([visual_out, state_out], dim=-1)


class NatureSepFuseEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureSepFuseEncoder, self).__init__()
    self.visual_base_first = NatureEncoder(
        cfg["in_channels"]
    )
    self.visual_base_second = NatureEncoder(
        cfg["in_channels"]
    )

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]

    self.visual_projector_first = RLProjection(
        in_dim=self.visual_base_first.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.visual_projector_second = RLProjection(
        in_dim=self.visual_base_second.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 3 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x = x[..., :self.state_dim]
    visual_x = x[..., self.state_dim:]
    # visual_x_second = x[..., self.state_dim + self.visual_dim: ]

    visual_x = visual_x.view(-1, self.in_channels * 2, self.w, self.h)
    visual_x_first, visual_x_second = visual_x.chunk(2, dim=-3)
    # visual_x_first = visual_x_first.view(-1, self.in_channels, self.w, self.h)
    # visual_x_second = visual_x_second.view(-1, self.in_channels, self.w, self.h)
    visual_out_first = self.visual_projector_first(
        self.visual_base_first(visual_x_first))
    visual_out_second = self.visual_projector_second(
        self.visual_base_second(visual_x_second))
    state_out = self.base(state_x)

    return torch.cat([visual_out_first, visual_out_second, state_out], dim=-1)


class NatureAttEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs
               ):
    super(NatureAttEncoder, self).__init__()
    self.visual_base_first = NatureEncoder(
        cfg["in_channels"]
    )
    self.visual_base_second = NatureEncoder(
        cfg["in_channels"]
    )

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]

    self.visual_projector_first = RLProjection(
        in_dim=self.visual_base_first.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.visual_projector_second = RLProjection(
        in_dim=self.visual_base_second.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    encoder_list = []
    for _ in range(2):
      encoder_list.append(
          nn.TransformerEncoderLayer(
              cfg["hidden_dims"][-1], 1, cfg["hidden_dims"][-1], dropout=0
          )
      )
    self.trans_encoder = nn.Sequential(*encoder_list)
    self.hidden_states_shape = 3 * cfg["hidden_dims"][-1]

  def forward(self, x):
    state_x = x[..., :self.state_dim]
    visual_x = x[..., self.state_dim:]
    # visual_x_second = x[..., self.state_dim + self.visual_dim: ]

    visual_x = visual_x.view(-1, self.in_channels * 2, self.w, self.h)
    visual_x_first, visual_x_second = visual_x.chunk(2, dim=-3)
    # visual_x_first = visual_x_first.view(-1, self.in_channels, self.w, self.h)
    # visual_x_second = visual_x_second.view(-1, self.in_channels, self.w, self.h)
    visual_out_first = self.visual_projector_first(
        self.visual_base_first(visual_x_first)).unsqueeze(0)
    visual_out_second = self.visual_projector_second(
        self.visual_base_second(visual_x_second)).unsqueeze(0)
    state_out = self.base(state_x).unsqueeze(0)

    _, B, F = state_out.shape

    x = torch.cat([visual_out_first, visual_out_second, state_out], dim=0)
    x = self.trans_encoder(x)
    x = x.permute(1, 0, 2).reshape(B, 3 * F)

    return x


class NatureFuseLSTMEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseLSTMEncoder, self).__init__()
    self.visual_base = NatureEncoder(
        in_channels=cfg["in_channels"],
        groups=cfg["in_channels"],
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1] * 3
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 4 * cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMSeparateEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseLSTMSeparateEncoder, self).__init__()
    self.visual_base = NatureEncoder(
        in_channels=cfg["in_channels"],
        groups=cfg["in_channels"],
    )

    self.state_dim = cfg["state_dim"]
    self.proprioceptive_recurrent_dim = cfg["proprioceptive_recurrent_dim"]
    self.hidden_states_shape = cfg["recurrent"]["hidden_size"] + \
        self.state_dim - self.proprioceptive_recurrent_dim
    cfg["recurrent"]["input_size"] = 2 * cfg["hidden_dims"][-1]
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1] * 3
    )

    self.base = MLPBase(
        input_shape=cfg["proprioceptive_recurrent_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

  def forward(self, x, states, mask):
    proprioceptive_recurrent_x, proprioceptive_non_recurrent_x, visual_x = x[...,
                                                                             :self.proprioceptive_recurrent_dim], x[..., self.proprioceptive_recurrent_dim:self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = visual_out.view(*proprioceptive_non_recurrent_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(proprioceptive_recurrent_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    rnn_output = torch.cat(
        [rnn_output, proprioceptive_non_recurrent_x], dim=-1)
    return rnn_output, new_states


class NatureFuseLSTMAddEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseLSTMAddEncoder, self).__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1] * 3
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 4 * cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_out = visual_head_out + visual_neck_out
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMMultiplyEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super(NatureFuseLSTMMultiplyEncoder, self).__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1] * 3
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 4 * cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_out = visual_head_out * visual_neck_out
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTM1CamMultiplyEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_base = NatureEncoder(
        in_channels=cfg["in_channels"],
        groups=cfg["in_channels"],
    )

    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )
    self.hidden_states_shape = cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    # rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_input = visual_out * state_out
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMEncoderV2(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_base = NatureEncoder(
        in_channels=cfg["in_channels"],
        groups=cfg["in_channels"],
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )
    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    print(rnn_input.shape)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMMultiplyEncoderV2(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_out = visual_head_out * visual_neck_out
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMConcatEncoderV2(nn.Module):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim * 2,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_head_out = visual_head_out.view(*state_x.shape[:2], -1)
    visual_neck_out = visual_neck_out.view(*state_x.shape[:2], -1)
    visual_out = torch.cat([visual_head_out, visual_neck_out], dim=-1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMAllMultiplyEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)
    visual_out = visual_head_out * visual_neck_out
    visual_out = visual_out.view(*state_x.shape[:2], -1)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)
    # rnn_input = torch.cat([visual_out, state_out], dim=-1)
    rnn_input = visual_out * state_out
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states

# class SelfAttention(nn.Module):
#   def __init__(self, in_channels):
#     super().__init__()
#     self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#     self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#     self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
#     self.in_channels = in_channels

#   def forward(self, query, key, value):
#     N, C, H, W = query.shape
#     assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
#     q = self.conv_query(query).reshape(N, C, H*W)#.permute(0, 2, 1)
#     k = self.conv_key(key).reshape(N, C, H*W)#.permute(0, 2, 1)
#     v = self.conv_value(value).reshape(N, C, H*W)#.permute(0, 2, 1)
#     attention = k.transpose(1, 2)@q / C**0.5
#     attention = attention.softmax(dim=1)
#     output = v@attention
#     output = output.reshape(N, C, H, W)
#     return query + output # Add with query and output


class NatureFuseLSTMMultiplyAttentionEncoderV2(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]

    self.head_visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.neck_visual_projector = RLProjection(
        in_dim=self.visual_neck_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.head_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.head_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]
                     ), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1])
    )
    self.neck_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.neck_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]
                     ), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1])
    )
    # self.state_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.hidden_states_shape = cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]
    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)

    visual_head_out = self.head_visual_projector(visual_head_out)
    visual_head_out = visual_head_out.view(*state_x.shape[:2], -1)
    visual_neck_out = self.neck_visual_projector(visual_neck_out)
    visual_neck_out = visual_neck_out.view(*state_x.shape[:2], -1)

    # print(visual_head_out.shape)
    # print(visual_neck_out.shape)
    visual_head_out = self.head_attention(
        visual_head_out, visual_neck_out, visual_neck_out
    )[0] + visual_head_out
    visual_head_out = self.head_mlp(visual_head_out)

    visual_neck_out = self.neck_attention(
        visual_neck_out, visual_head_out, visual_head_out
    )[0] + visual_neck_out
    visual_neck_out = self.neck_mlp(visual_neck_out)

    visual_out = visual_head_out * visual_neck_out
    state_out = self.base(state_x)
    rnn_input = visual_out * state_out
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class NatureFuseLSTMMultiplyAllAttentionEncoderV2(nn.Module):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__()
    self.visual_head_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.visual_neck_cam = NatureEncoder(
        in_channels=cfg["in_channels"] // 2,
    )
    self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]

    self.head_visual_projector = RLProjection(
        in_dim=self.visual_head_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.neck_visual_projector = RLProjection(
        in_dim=self.visual_neck_cam.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )
    self.head_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.head_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]),
        nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1])
    )
    self.neck_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.neck_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]),
        nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1])
    )
    # self.state_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )
    self.state_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.state_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]),
        nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1])
    )
    self.hidden_states_shape = cfg["hidden_dims"][-1]

  def forward(self, x, states, mask):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_head_x, visual_neck_x = visual_x[:, 0:(
        self.in_channels // 2)], visual_x[:, (self.in_channels // 2):]

    state_out = self.base(state_x)

    visual_head_out = self.visual_head_cam(visual_head_x)
    visual_neck_out = self.visual_neck_cam(visual_neck_x)

    visual_head_out = self.head_visual_projector(visual_head_out)
    visual_head_out = visual_head_out.view(*state_x.shape[:2], -1)
    visual_neck_out = self.neck_visual_projector(visual_neck_out)
    visual_neck_out = visual_neck_out.view(*state_x.shape[:2], -1)

    head_att_input = torch.cat([visual_neck_out, state_out], dim=0)
    neck_att_input = torch.cat([visual_head_out, state_out], dim=0)
    state_att_input = torch.cat([visual_head_out, visual_neck_out], dim=0)

    visual_head_out = self.head_attention(
        visual_head_out, head_att_input, head_att_input
    )[0] + visual_head_out
    visual_head_out = self.head_mlp(visual_head_out)

    visual_neck_out = self.neck_attention(
        visual_neck_out, neck_att_input, neck_att_input
    )[0] + visual_neck_out
    visual_neck_out = self.neck_mlp(visual_neck_out)

    state_out = self.state_attention(
        state_out, state_att_input, state_att_input
    )[0] + state_out
    state_out = self.state_mlp(state_out)

    visual_out = visual_head_out * visual_neck_out

    rnn_input = visual_out * state_out
    rnn_output, new_states = self.rnn(rnn_input, states, mask)
    return rnn_output, new_states


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


class LocoTransformerBaseEncoder(nn.Module):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__()
    # assert in_channels == 16
    self.in_channels = cfg["in_channels"]
    self.camera_num = cfg["camera_num"]
    self.token_dim = cfg["token_dim"]
    # if self.in_channels == 1:

    self.front_base = NatureEncoder(
        self.in_channels, flatten=False
    )
    self.front_up_conv = nn.Conv2d(64, self.token_dim, 1)

    if self.camera_num == 2:
      self.neck_base = NatureEncoder(
          self.in_channels, flatten=False
      )
      self.neck_up_conv = nn.Conv2d(64, self.token_dim, 1)

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )
    self.state_projector = RLProjection(
        in_dim=self.base.output_shape,
        out_dim=self.token_dim
    )

    self.per_modal_tokens = 16
    self.add_pos_emb = cfg["add_pos_emb"]
    if self.add_pos_emb:
      self.pos_emb = nn.Embedding(
          (1 + (self.camera_num == 2)) * self.per_modal_tokens + 1,
          self.token_dim
      )

    self.visual_dim = self.token_dim * \
        (1 + self.add_pos_emb)  # RGBD(DEPTH) + POSITIONAL
    self.flatten_layer = Flatten()

  def forward(self, visual_x, state_x, return_raw_visual_vecs=False):
    if len(visual_x.shape) <= 3:
      visual_x = visual_x.unsqueeze(0)
      state_x = state_x.unsqueeze(0)

    visual_x = visual_x.view(torch.Size(
        [np.prod(visual_x.size()[:-3])]) + visual_x.size()[-3:]
    )
    # (Batch Shape, 16, 64, 64)

    if self.camera_num == 2:
      print(visual_x.shape)
      front_visual_x = visual_x[..., 0:-1:2, :, :]
      neck_visual_x = visual_x[..., 1::2, :, :]
      print(front_visual_x.shape)
      print(neck_visual_x.shape)
    else:
      front_visual_x = visual_x[..., :, :, :]

    raw_visual_vecs = []

    front_visual_out_raw = self.front_base(
        front_visual_x
    )
    front_visual_out = self.front_up_conv(front_visual_out_raw)
    if self.camera_num == 2:
      neck_visual_out_raw = self.neck_base(
          neck_visual_x
      )
      neck_visual_out = self.neck_up_conv(neck_visual_out_raw)

    if return_raw_visual_vecs:
      raw_visual_vecs.append(
          self.flatten_layer(front_visual_out_raw)
      )
      if self.camera_num == 2:
        raw_visual_vecs.append(
            self.flatten_layer(neck_visual_out_raw)
        )

    # (Batch Shape, Channel, # Patches, # Patches)
    visual_shape = front_visual_out.shape
    # (Batch Shape, Channel, # Patches, # Patches)
    num_patches = visual_shape[-1]

    # (# Patches ** 2， Batch Shape, Channel)
    front_visual_out = front_visual_out.reshape(
        visual_shape[0], visual_shape[1], num_patches * num_patches
    )
    # (Batch Shape, Channel, # Patches ** 2)
    front_visual_out = front_visual_out.permute(
        2, 0, 1
    )
    # (# Patches ** 2， Batch Shape, Channel)
    if self.camera_num == 2:
      neck_visual_out = neck_visual_out.reshape(
          visual_shape[0], visual_shape[1], num_patches * num_patches
      )
      # (Batch Shape, Channel, # Patches ** 2)
      neck_visual_out = neck_visual_out.permute(
          2, 0, 1
      )

    state_out = self.base(state_x)
    state_out_proj = self.state_projector(state_out)
    # (Batch Shape, Channel)
    state_out_proj = state_out_proj.unsqueeze(0)

    out_list = [state_out_proj]
    out_list.append(front_visual_out)
    if self.camera_num == 2:
      out_list.append(neck_visual_out)
    out = torch.cat(out_list, dim=0)
    if self.add_pos_emb:
      idx = torch.arange(out.shape[0]).to(out.device)
      pos_emb = self.pos_emb(idx)
      pos_emb = pos_emb.unsqueeze(1).repeat_interleave(out.shape[1], 1)
      out = torch.cat([out, pos_emb], dim=-1)

    if return_raw_visual_vecs:
      return out, raw_visual_vecs
    return out


class LocoTransformerLSTMEncoder(nn.Module):
  def __init__(
      self,
      cfg
  ):
    super().__init__()

    self.encoder = LocoTransformerBaseEncoder(cfg)

    # self.rnn = get_network("lnlstm", cfg["recurrent"])
    self.in_channels = cfg["in_channels"]
    self.camera_num = cfg["camera_num"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(
        math.sqrt((self.visual_dim // self.camera_num) // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

    self.add_ln = cfg["add_ln"]
    # self.activation_func = activation_func
    self.max_pool = cfg["max_pool"]
    visual_append_input_shape = self.encoder.visual_dim

    self.token_norm = cfg["token_norm"]
    if self.token_norm:
      self.token_ln = nn.LayerNorm(self.encoder.visual_dim)

    transformer_params = cfg["transformer_params"]
    self.visual_append_layers = nn.ModuleList()
    for n_head, dim_feedforward in transformer_params:
      visual_att_layer = nn.TransformerEncoderLayer(
          visual_append_input_shape, n_head, dim_feedforward,
          dropout=0
      )
      self.visual_append_layers.append(visual_att_layer)

    self.per_modal_tokens = self.encoder.per_modal_tokens
    # self.per_modal_tokens = 16
    self.second = (self.camera_num == 2)
    self.rnn_hidden_size = cfg["recurrent"]["hidden_size"]
    self.rnn_num_layers = cfg["recurrent"]["num_layers"]
    self.rnn = nn.GRU(
        input_size=visual_append_input_shape * (2 + self.second),
        hidden_size=cfg["recurrent"]["hidden_size"],
        num_layers=cfg["recurrent"]["num_layers"],
        # batch_first=True
    )

  def forward(self, x, states):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(-1, self.in_channels *
                             self.camera_num, self.w, self.h)

    if len(original_state_x_shape) >= 3:
      # Num Transition, Num Layers, Batch, Hidden
      states = states.permute(1, 0, 2, 3)
      states = states.reshape(self.rnn_num_layers, -1,
                              self.rnn_hidden_size).contiguous()
    else:
      states = states.reshape(self.rnn_num_layers, -1,
                              self.rnn_hidden_size).contiguous()
    out = self.encoder(
        visual_x, state_x
    )
    if self.token_norm:
      out = self.token_ln(out)
    for att_layer in self.visual_append_layers:
      out = att_layer(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    out_state = out[0, ...]
    if self.max_pool:
      out_first = out[1: 1 + self.per_modal_tokens, ...].max(dim=0)[0]
    else:
      out_first = out[1: 1 + self.per_modal_tokens, ...].mean(dim=0)
    out_list = [out_state, out_first]
    if self.second:
      out_second = out[1 + self.per_modal_tokens: 1 +
                       2 * self.per_modal_tokens, ...]
      if self.max_pool:
        out_second = out_second.max(dim=0)[0]
      else:
        out_second = out_second.mean(dim=0)
      out_list.append(out_second)

    out = torch.cat(out_list, dim=-1)
    # out = out.view(*original_state_x_shape[:2], -1)
    out = out.unsqueeze(0)
    out, new_states = self.rnn(out, states)
    out = out.squeeze(0)
    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    return out, new_states


class LocoTransformerEncoder(nn.Module):
  def __init__(
      self,
      cfg
  ):
    super().__init__()

    self.encoder = LocoTransformerBaseEncoder(cfg)

    self.in_channels = cfg["in_channels"]
    self.camera_num = cfg["camera_num"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(
        math.sqrt((self.visual_dim // self.camera_num) // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

    self.add_ln = cfg["add_ln"]
    # self.activation_func = activation_func
    self.max_pool = cfg["max_pool"]
    visual_append_input_shape = self.encoder.visual_dim

    self.token_norm = cfg["token_norm"]
    if self.token_norm:
      self.token_ln = nn.LayerNorm(self.encoder.visual_dim)

    transformer_params = cfg["transformer_params"]
    self.visual_append_layers = nn.ModuleList()
    for n_head, dim_feedforward in transformer_params:
      visual_att_layer = nn.TransformerEncoderLayer(
          visual_append_input_shape, n_head, dim_feedforward,
          dropout=0
      )
      self.visual_append_layers.append(visual_att_layer)

    self.per_modal_tokens = self.encoder.per_modal_tokens
    # self.per_modal_tokens = 16
    self.second = (self.camera_num == 2)
    self.out_lienar = nn.Linear(
        (self.camera_num + 1) * visual_append_input_shape,
        cfg["encoder_out"]
    )

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(-1, self.in_channels *
                             self.camera_num, self.w, self.h)

    out = self.encoder(
        visual_x, state_x
    )
    if self.token_norm:
      out = self.token_ln(out)
    for att_layer in self.visual_append_layers:
      out = att_layer(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    out_state = out[0, ...]
    if self.max_pool:
      out_first = out[1: 1 + self.per_modal_tokens, ...].max(dim=0)[0]
    else:
      out_first = out[1: 1 + self.per_modal_tokens, ...].mean(dim=0)
    out_list = [out_state, out_first]
    if self.second:
      out_second = out[1 + self.per_modal_tokens: 1 +
                       2 * self.per_modal_tokens, ...]
      if self.max_pool:
        out_second = out_second.max(dim=0)[0]
      else:
        out_second = out_second.mean(dim=0)
      out_list.append(out_second)

    out = torch.cat(out_list, dim=-1)
    # out = out.view(*original_state_x_shape[:2], -1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    return out


class LocoTransformerBaseV2Encoder(nn.Module):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__()
    # assert in_channels == 16
    self.in_channels = cfg["in_channels"]
    self.camera_num = cfg["camera_num"]
    self.token_dim = cfg["token_dim"]
    # if self.in_channels == 1:

    self.front_base = NatureEncoder(
        self.in_channels, groups=self.in_channels, flatten=False
    )
    self.front_up_conv = nn.Conv2d(
        64 * self.in_channels, self.token_dim * self.in_channels, 1, groups=self.in_channels)

    assert self.camera_num == 1
    if self.camera_num == 2:
      self.neck_base = NatureEncoder(
          self.in_channels, flatten=False
      )
      self.neck_up_conv = nn.Conv2d(64, self.token_dim, 1)

    self.base = MLPBase(
        input_shape=cfg["state_dim"] // self.in_channels,
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.proprio_base = MLPBase(
        input_shape=cfg["state_dim"] // self.in_channels,
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )

    self.proprio_projector = RLProjection(
        in_dim=self.proprio_base.output_shape,
        out_dim=self.token_dim
    )

    self.state_projector = RLProjection(
        in_dim=self.base.output_shape,
        out_dim=self.token_dim
    )

    self.per_modal_tokens = 16
    self.add_pos_emb = cfg["add_pos_emb"]
    if self.add_pos_emb:
      self.pos_emb = nn.Embedding(
          (1 + (self.camera_num == 2)) *
          self.per_modal_tokens + 1 + self.in_channels,
          self.token_dim
      )

    self.visual_dim = self.token_dim * \
        (1 + self.add_pos_emb)  # RGBD(DEPTH) + POSITIONAL
    self.flatten_layer = Flatten()

  def forward(self, visual_x, state_x, return_raw_visual_vecs=False):
    if len(visual_x.shape) <= 3:
      visual_x = visual_x.unsqueeze(0)
      state_x = state_x.unsqueeze(0)

    state_x = state_x.view(
        torch.Size([np.prod(visual_x.size()[:-3])]) +
        (self.in_channels, state_x.shape[-1] // self.in_channels,)
    )

    proprio_x = state_x[..., 0, :]

    visual_x = visual_x.view(torch.Size(
        [np.prod(visual_x.size()[:-3])]) + visual_x.size()[-3:]
    )
    # (Batch Shape, 16, 64, 64)

    assert self.camera_num == 1
    if self.camera_num == 2:
      print(visual_x.shape)
      front_visual_x = visual_x[..., 0:-1:2, :, :]
      neck_visual_x = visual_x[..., 1::2, :, :]
      print(front_visual_x.shape)
      print(neck_visual_x.shape)
    else:
      front_visual_x = visual_x[..., :, :, :]

    raw_visual_vecs = []

    front_visual_out_raw = self.front_base(
        front_visual_x
    )
    front_visual_out = self.front_up_conv(front_visual_out_raw)
    if self.camera_num == 2:
      neck_visual_out_raw = self.neck_base(
          neck_visual_x
      )
      neck_visual_out = self.neck_up_conv(neck_visual_out_raw)

    if return_raw_visual_vecs:
      raw_visual_vecs.append(
          self.flatten_layer(front_visual_out_raw)
      )
      if self.camera_num == 2:
        raw_visual_vecs.append(
            self.flatten_layer(neck_visual_out_raw)
        )

    # (Batch Shape, Channel, # Patches, # Patches)
    visual_shape = front_visual_out.shape
    # (Batch Shape, Channel, # Patches, # Patches)
    num_patches = visual_shape[-1]

    # (# Patches ** 2， Batch Shape, Channel)
    front_visual_out = front_visual_out.reshape(
        visual_shape[0], self.in_channels, visual_shape[1] // self.in_channels, num_patches * num_patches
    )
    # (Batch Shape, Channel, # Patches ** 2)
    front_visual_out = front_visual_out.permute(
        1, 3, 0, 2
    ).reshape(num_patches * num_patches * self.in_channels, visual_shape[0], -1)
    # (# Patches ** 2， Batch Shape, Channel)
    if self.camera_num == 2:
      neck_visual_out = neck_visual_out.reshape(
          visual_shape[0], visual_shape[1], num_patches * num_patches
      )
      # (Batch Shape, Channel, # Patches ** 2)
      neck_visual_out = neck_visual_out.permute(
          2, 0, 1
      )

    state_out = self.base(state_x)
    state_out_proj = self.state_projector(state_out)
    # (Batch Shape, Channel)
    # state_out_proj = state_out_proj.unsqueeze(0)
    state_out_proj = state_out_proj.permute(1, 0, 2)

    proprio_out = self.proprio_base(proprio_x)
    proprio_out_proj = self.proprio_projector(proprio_out)
    proprio_out_proj = proprio_out_proj.unsqueeze(0)

    out_list = [proprio_out_proj, state_out_proj]
    out_list.append(front_visual_out)
    if self.camera_num == 2:
      out_list.append(neck_visual_out)
    out = torch.cat(out_list, dim=0)
    if self.add_pos_emb:
      idx = torch.arange(out.shape[0]).to(out.device)
      pos_emb = self.pos_emb(idx)
      pos_emb = pos_emb.unsqueeze(1).repeat_interleave(out.shape[1], 1)
      out = torch.cat([out, pos_emb], dim=-1)

    if return_raw_visual_vecs:
      return out, raw_visual_vecs
    return out


class LocoTransformerV2Encoder(nn.Module):
  def __init__(
      self,
      cfg
  ):
    super().__init__()

    self.encoder = LocoTransformerBaseV2Encoder(cfg)

    self.in_channels = cfg["in_channels"]
    self.camera_num = cfg["camera_num"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(
        math.sqrt((self.visual_dim // self.camera_num) // self.in_channels))
    self.state_dim = cfg["state_dim"]
    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

    self.add_ln = cfg["add_ln"]
    # self.activation_func = activation_func
    self.max_pool = cfg["max_pool"]
    visual_append_input_shape = self.encoder.visual_dim

    self.token_norm = cfg["token_norm"]
    if self.token_norm:
      self.token_ln = nn.LayerNorm(self.encoder.visual_dim)

    transformer_params = cfg["transformer_params"]
    self.visual_append_layers = nn.ModuleList()
    for n_head, dim_feedforward in transformer_params:
      visual_att_layer = nn.TransformerEncoderLayer(
          visual_append_input_shape, n_head, dim_feedforward,
          dropout=0
      )
      self.visual_append_layers.append(visual_att_layer)

    self.per_modal_tokens = self.encoder.per_modal_tokens
    # self.per_modal_tokens = 16
    self.second = (self.camera_num == 2)
    self.out_lienar = nn.Linear(
        (self.camera_num + 1) * visual_append_input_shape,
        cfg["encoder_out"]
    )

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(-1, self.in_channels *
                             self.camera_num, self.w, self.h)

    out = self.encoder(
        visual_x, state_x
    )
    if self.token_norm:
      out = self.token_ln(out)
    for att_layer in self.visual_append_layers:
      out = att_layer(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    out_state = out[0, ...]
    if self.max_pool:
      out_vis = out[1:, ...].max(dim=0)[0]
    else:
      out_vis = out[1:, ...].mean(dim=0)
    out_list = [out_state, out_vis]
    # if self.second:
    #   out_second = out[1 + self.per_modal_tokens: 1 +
    #                    2 * self.per_modal_tokens, ...]
    #   if self.max_pool:
    #     out_second = out_second.max(dim=0)[0]
    #   else:
    #     out_second = out_second.mean(dim=0)
    #   out_list.append(out_second)

    out = torch.cat(out_list, dim=-1)
    # out = out.view(*original_state_x_shape[:2], -1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    return out


def euler2mat(angle):
  """Convert euler angles to rotation matrix.
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      angle: rotation angle along 3 axis (a, b, y) in radians -- size = [B, 3]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
  """
  B = angle.size(0)
  # import pdb; pdb.set_trace()

  x, y, z = angle[:, 1], angle[:, 0], angle[:, 2]

  cosz = torch.cos(z)
  sinz = torch.sin(z)

  zeros = z.detach() * 0
  ones = zeros.detach() + 1
  zmat = torch.stack([cosz, -sinz, zeros,
                      sinz, cosz, zeros,
                      zeros, zeros, ones], dim=1).reshape(B, 3, 3)

  cosy = torch.cos(y)
  siny = torch.sin(y)

  ymat = torch.stack([cosy, zeros, siny,
                      zeros, ones, zeros,
                      -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

  cosx = torch.cos(x)
  sinx = torch.sin(x)

  xmat = torch.stack([ones, zeros, zeros,
                      zeros, cosx, -sinx,
                      zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

  rotMat = xmat @ ymat @ zmat

  v_trans = angle[:, 3:]

  # F.affine_grid takes 3x4
  rotMat = torch.cat([rotMat, v_trans.view([B, 3, 1]).to(rotMat.device)], 2)

  return rotMat


class EncoderTraj(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.convs = NatureEncoder(
        2, flatten=False
    )  # the last frame and current frame
    self.relu = nn.ReLU()
    self.pose_pred = nn.Conv2d(
        64, 6, 1, padding=0
    )

    self.scale_rotate = cfg["scale_rotate"]
    self.scale_translate = cfg["scale_translate"]

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
          zeros_(m.bias)

  def forward(self, input_frames):

    out = self.convs(input_frames)
    pose = self.pose_pred(out)
    pose = pose.mean(3).mean(2)
    pose = pose.view(pose.size(0), 6)

    pose_r = pose[:, :3] * self.scale_rotate
    pose_t = pose[:, 3:] * self.scale_translate

    pose_final = torch.cat([pose_r, pose_t], 1)

    return pose_final


class Encoder3D(nn.Module):
  def __init__(self):
    super(Encoder3D, self).__init__()
    self.feature_extraction = ImpalaEncoder(
        1, channels=[16, 32, 128], flatten=False
    )  # the last frame and current frame
    self.conv3d_1 = nn.ConvTranspose3d(16, 16, 4, stride=2, padding=1)
    # self.conv3d_2 = nn.ConvTranspose3d(16, 16, 4, stride=2, padding=1)

  def forward(self, img):
    z_2d = self.feature_extraction(img)
    B, C, H, W = z_2d.shape
    z_3d = z_2d.reshape([-1, 16, 8, H, W])
    z_3d = F.leaky_relu(self.conv3d_1(z_3d))
    # z_3d = F.leaky_relu(self.conv3d_2(z_3d))
    return z_3d


class Decoder(nn.Module):
  def __init__(self, args):
    super(Decoder, self).__init__()
    self.depth_3d = 32
    self.conv3 = nn.Conv2d(2048, 512, 1)
    self.upconv3 = nn.ConvTranspose2d(
        512, 256, kernel_size=4, stride=2, padding=1)
    self.upconv4 = nn.ConvTranspose2d(
        256, 64, kernel_size=4, stride=2, padding=1)
    self.upconv_final = nn.ConvTranspose2d(
        64, 3, kernel_size=3, stride=1, padding=1)

  def forward(self, code):
    code = code.view(-1, code.size(1) * code.size(2),
                     code.size(3), code.size(4))
    code = F.leaky_relu(self.conv3(code))
    code = F.leaky_relu(self.upconv3(code))
    code = F.leaky_relu(self.upconv4(code))
    output = self.upconv_final(code)
    return output


class Rotate(nn.Module):
  def __init__(self, cfg):
    super(Rotate, self).__init__()
    self.padding_mode = cfg["padding_mode"]
    self.conv3d_1 = nn.Conv3d(16, 32, 3, padding=1)
    self.conv3d_2 = nn.Conv3d(32, 32, 3, padding=1)

  def forward(self, code, theta):
    rot_code = stn(code, theta, self.padding_mode)
    rot_code = F.leaky_relu(self.conv3d_1(rot_code))
    rot_code = F.leaky_relu(self.conv3d_2(rot_code))
    return rot_code

# class ConvHead(nn.Module):
#   def __init__(self, cfg):
#     super().__init__()
#     self.depth_3d = 32
#     self.conv3 = nn.Conv2d(64 * 8, 256, 1)
#     # self.upconv3 = nn.ConvTranspose2d(
#     #     512, 256, kernel_size=4, stride=2, padding=1)
#     # self.upconv4 = nn.ConvTranspose2d(
#     #     256, 64, kernel_size=4, stride=2, padding=1)
#     # self.upconv_final = nn.ConvTranspose2d(
#     #     64, 3, kernel_size=3, stride=1, padding=1)

#   def forward(self, code):
#     code = code.view(-1, code.size(1) * code.size(2),
#                      code.size(3), code.size(4))
#     code = F.leaky_relu(self.conv3(code))
#     code = F.leaky_relu(self.upconv3(code))
#     code = F.leaky_relu(self.upconv4(code))
#     output = self.upconv_final(code)
#     return output


def stn(x, theta, padding_mode='zeros'):
  grid = F.affine_grid(theta, x.size())
  img = F.grid_sample(x, grid, padding_mode=padding_mode)
  return img


class ImpalaVoxelEncoder(nn.Module):
  def __init__(
      self,
      cfg
  ):
    super().__init__()

    self.in_channels = cfg["in_channels"]
    self.camera_num = cfg["camera_num"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(
        math.sqrt((self.visual_dim // self.camera_num) // self.in_channels))
    self.state_dim = cfg["state_dim"]
    # self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

    assert self.camera_num == 1, "suport 1 cam for now"

    self.encoder_3d = Encoder3D()
    self.encoder_traj = EncoderTraj(cfg)
    self.rotate = Rotate(cfg)

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 16, out_channels=256,
            kernel_size=8, stride=4, padding=1
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)

    # if flatten:
    #   layer_list.append(
    #       nn.Flatten()
    #   )
    self.visual_projector = RLProjection(
        in_dim=128,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
    )

    self.hidden_states_shape = cfg["hidden_dims"][-1] * 2

  def forward(self, x):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(
        -1, self.in_channels * self.camera_num, self.w, self.h
    )

    b = visual_x.shape[0]

    # b, t, c, h, w = clips.size()
    codes = self.encoder_3d(
        visual_x.reshape(-1, 1, self.w, self.h)
    )
    _, C, D, H, W = codes.size()
    print(codes.size())
    code_t = codes.view(
        b * self.in_channels, C, D, H, W
    )
    # code_t = codes.unsqueeze(1).repeat(
    #     1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

    frame_ref = visual_x[:, 0:1].unsqueeze(
        2).repeat(1, self.in_channels, 1, 1, 1)
    frame_pair = torch.cat([
        frame_ref,
        visual_x.view(- 1, self.in_channels, 1, self.w, self.h)
    ], dim=2).reshape(
        b * self.in_channels, 2, self.w, self.h
    )
    # pair_tensor = clips_pair.view(b * t, 2, self.h, self.w)

    poses = self.encoder_traj(frame_pair)
    theta = euler2mat(poses)

    rot_codes = self.rotate(code_t, theta)
    print(rot_codes.shape)
    voxels = rot_codes.view(
        b, self.in_channels, 32, D, H, W
    ).mean(1)
    # voxel shape: B, C, D, H, W
    flatten_voxel = voxels.view(b, 32 * D, H, W)

    flatten_voxel = self.convs(flatten_voxel).flatten(1)
    print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)

    state_out = self.base(state_x)

    out = torch.cat([flatten_voxel, state_out], dim=-1)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    return out
