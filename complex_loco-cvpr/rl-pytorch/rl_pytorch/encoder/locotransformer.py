import torch
import torch.nn as nn
from .nature import NatureEncoder, WideNatureEncoder, CoarseNatureEncoder, SmallNatureEncoder
from .mlp import MLPBase, RLProjection
import math
import numpy as np
from .mlp import Flatten
from .decoder import MLPHeightDecoder
from .resnet18 import get_small_resnet18


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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

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
        activation_func=self.non_linear_func,
        **kwargs
    )
    self.state_projector = RLProjection(
        in_dim=self.base.output_shape,
        out_dim=self.token_dim,
        proj_function=self.non_linear_func,
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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

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

    self.height_decoder_class = MLPHeightDecoder

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
    self.out_lienar = RLProjection(
        (self.camera_num + 1) * visual_append_input_shape,
        cfg["encoder_out"],
        proj_function=self.non_linear_func,
    )

  def forward(self, x, return_3d_code=False):
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
    if return_3d_code:
      return out, out_first
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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

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
        out_dim=self.token_dim,
        proj_function=self.non_linear_func,
    )

    self.state_projector = RLProjection(
        in_dim=self.base.output_shape,
        out_dim=self.token_dim,
        proj_function=self.non_linear_func,
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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

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


class LocoTransformerBaseV3Encoder(nn.Module):
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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    self.front_base = NatureEncoder(
        2, flatten=False
    )
    self.front_up_conv = nn.Conv2d(64, self.token_dim, 1)

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        **kwargs
    )
    self.state_projector = RLProjection(
        in_dim=self.base.output_shape,
        out_dim=self.token_dim,
        proj_function=self.non_linear_func,
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


class LocoTransformerV3Encoder(nn.Module):
  def __init__(
      self,
      cfg
  ):
    super().__init__()

    self.encoder = LocoTransformerBaseEncoder(cfg)

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

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
    self.out_lienar = RLProjection(
        (self.camera_num + 1) * visual_append_input_shape,
        cfg["encoder_out"],
        proj_function=self.non_linear_func,
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


class LocoTransformerResNet18BaseEncoder(LocoTransformerBaseEncoder):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__(cfg)

    self.front_base = get_small_resnet18(
        pretrained=False, flatten=False, input_channels=self.in_channels
    )
    self.front_up_conv = nn.Conv2d(192, self.token_dim, 1)
    self.per_modal_tokens = 36


class LocoTransformerResNet18Encoder(LocoTransformerEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder = LocoTransformerResNet18BaseEncoder(cfg)


class LocoTransformerWideNatureBaseEncoder(LocoTransformerBaseEncoder):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__(cfg)

    self.front_base = WideNatureEncoder(
        cfg["in_channels"], channels=[32, 64, 128], flatten=False
    )
    self.front_up_conv = nn.Conv2d(128, self.token_dim, 1)
    self.per_modal_tokens = 64


class LocoTransformerWideNatureEncoder(LocoTransformerEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder = LocoTransformerWideNatureBaseEncoder(cfg)


class LocoTransformerCoarseNatureBaseEncoder(LocoTransformerBaseEncoder):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__(cfg)

    self.front_base = CoarseNatureEncoder(
        cfg["in_channels"], channels=[32, 64, 128], flatten=False
    )  # the last frame and current frame
    self.front_up_conv = nn.Conv2d(128, self.token_dim, 1)
    self.per_modal_tokens = 64


class LocoTransformerCoarseNatureEncoder(LocoTransformerEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder = LocoTransformerCoarseNatureBaseEncoder(cfg)


class LocoTransformerSmallNatureBaseEncoder(LocoTransformerBaseEncoder):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__(cfg)

    self.front_base = SmallNatureEncoder(
        cfg["in_channels"], channels=[32, 64, 96], flatten=False
    )  # the last frame and current frame
    self.front_up_conv = nn.Conv2d(96, self.token_dim, 1)
    self.per_modal_tokens = 36


class LocoTransformerSmallNatureEncoder(LocoTransformerEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder = LocoTransformerSmallNatureBaseEncoder(cfg)
