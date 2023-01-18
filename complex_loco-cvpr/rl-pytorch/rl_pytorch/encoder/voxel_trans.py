import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .impala import ImpalaEncoder
from .nature import CoarseNatureEncoder, NatureEncoder, WideNatureEncoder, CompactNatureEncoder
from torch.nn.init import xavier_uniform_, zeros_
from .mlp import RLProjection, MLPBase
from .decoder import NatureVoxelHeightDecoder, NatureVoxelV2HeightDecoder, NatureVoxelV3HeightDecoder
from .decoder import Res18VoxelHeightDecoder, Res18VoxelV2HeightDecoder, Res18VoxelV3HeightDecoder
from .decoder import CoarseNatureVoxelHeightDecoder, CoarseNatureVoxelV2HeightDecoder, CoarseNatureVoxelV3HeightDecoder, SmallNatureImgDecoder
from .submodule import stn, euler2mat
from .voxel import NatureVoxelEncoder, NatureEncoder3D, CompactNatureEncoder3D, CoarseNatureEncoder3D, EncoderTraj, NatureRotate, SmallNatureEncoder3D
from .resnet18_voxel import ResNet18Encoder3D, ResNet18EncoderTraj, ResNet18Rotate

from .decoder import Res18ImgDecoder, CoarseNatureImgDecoder


class NatureVoxelTransEncoder(nn.Module):
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

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    assert self.camera_num == 1, "suport 1 cam for now"

    self.encoder_3d = NatureEncoder3D()
    self.encoder_traj = EncoderTraj(cfg)
    self.rotate = NatureRotate(cfg)

    self.height_decoder_class = NatureVoxelHeightDecoder

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
    )

    self.hidden_states_shape = cfg["hidden_dims"][-1] * 2

    self.add_ln = cfg["add_ln"]
    self.max_pool = cfg["max_pool"]
    # visual_append_input_shape =

    self.token_norm = cfg["token_norm"]
    self.token_dim = cfg["token_dim"]
    self.up_conv = nn.Conv2d(
        self.rotate.output_channel_num * 8,
        self.token_dim, 1
    )

    # if self.token_norm:
    #   self.token_ln = nn.LayerNorm(self.encoder.visual_dim)

    transformer_params = cfg["transformer_params"]
    self.trans_layers = nn.ModuleList()
    for n_head, dim_feedforward in transformer_params:
      att_layer = nn.TransformerEncoderLayer(
          self.token_dim, n_head, dim_feedforward,
          dropout=0
      )
      self.trans_layers.append(att_layer)

    # self.per_modal_tokens = self.encoder.per_modal_tokens
    self.per_modal_tokens = 36

    self.out_lienar = RLProjection(
        (self.camera_num + 1) * self.token_dim,
        cfg["hidden_dims"][-1] * 2,
        proj_function=self.non_linear_func,
    )

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(
        -1, self.in_channels * self.camera_num, self.w, self.h
    )
    b = visual_x.shape[0]

    state_out = self.base(state_x)

    # b, t, c, h, w = clips.size()
    codes = self.encoder_3d(
        visual_x.reshape(-1, 1, self.w, self.h)
    )
    # print(codes.shape)
    _, C, D, H, W = codes.size()
    code_t = codes.view(
        b * self.in_channels, C, D, H, W
    )

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

    voxels = rot_codes.view(
        b, self.in_channels, self.rotate.output_channel_num, D, H, W
    ).mean(1)
    # Voxels (b, c, D, H, W)
    # # voxel shape: B, C, D, H, W
    # flatten_voxel = voxels.view(b, 32 * D, H, W)

    flatten_voxel = voxels.view(b, self.rotate.output_channel_num * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.up_conv(flatten_voxel)

    tokens = flatten_voxel.reshape(
        b, self.token_dim, -1
    ).permute(2, 0, 1)
    # print(tokens.shape)

    out = torch.cat([
        state_out.unsqueeze(0), tokens
    ], dim=0)

    # if self.token_norm:
    #   out = self.token_ln(out)

    for att_layer in self.trans_layers:
      out = att_layer(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    out_state = out[0, ...]
    if self.max_pool:
      out_first = out[1: 1 + self.per_modal_tokens, ...].max(dim=0)[0]
    else:
      out_first = out[1: 1 + self.per_modal_tokens, ...].mean(dim=0)
    out_list = [out_state, out_first]

    out = torch.cat(out_list, dim=-1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    if return_3d_code:
      return out, out_first
    return out


class Res18VoxelTransEncoder(NatureVoxelTransEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder_3d = ResNet18Encoder3D(cfg)
    self.encoder_traj = ResNet18EncoderTraj(cfg)
    self.rotate = ResNet18Rotate(cfg)

    self.height_decoder_class = Res18VoxelHeightDecoder

    self.up_conv = nn.Conv2d(192, self.token_dim, 1)
    self.per_modal_tokens = 36

    self.img_decoder_class = Res18ImgDecoder

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(
        -1, self.in_channels * self.camera_num, self.w, self.h
    )
    b = visual_x.shape[0]

    state_out = self.base(state_x)

    # b, t, c, h, w = clips.size()
    codes = self.encoder_3d(
        visual_x.reshape(-1, 1, self.w, self.h)
    )
    # print(codes.shape)
    _, C, D, H, W = codes.size()
    code_t = codes.view(
        b * self.in_channels, C, D, H, W
    )

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

    # print(rot_codes.shape)
    voxels = rot_codes.view(
        b, self.in_channels, 32, D, H, W
    ).mean(1)
    # Voxels (b, c, D, H, W)
    # # voxel shape: B, C, D, H, W
    # flatten_voxel = voxels.view(b, 32 * D, H, W)

    flatten_voxel = voxels.view(b, 32 * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.up_conv(flatten_voxel)

    tokens = flatten_voxel.reshape(
        b, self.token_dim, -1
    ).permute(2, 0, 1)
    # print(tokens.shape)

    out = torch.cat([
        state_out.unsqueeze(0), tokens
    ], dim=0)

    # if self.token_norm:
    #   out = self.token_ln(out)
    # print(out.shape)

    for att_layer in self.trans_layers:
      out = att_layer(out)
    # (# Patches ** 2, Batch_size, Feature Dim)
    out_state = out[0, ...]
    if self.max_pool:
      out_first = out[1: 1 + self.per_modal_tokens, ...].max(dim=0)[0]
    else:
      out_first = out[1: 1 + self.per_modal_tokens, ...].mean(dim=0)
    out_list = [out_state, out_first]

    out = torch.cat(out_list, dim=-1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    if return_3d_code:
      return out, out_first
    return out

  def compute_latent_code(
      self, x
  ):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    visual_x = visual_x.view(
        -1, self.in_channels * self.camera_num, self.w, self.h
    )
    # visual_x = visual_x[:, -1, :, :]
    b = visual_x.shape[0]

    # b, t, c, h, w = clips.size()
    codes = self.encoder_3d(
        visual_x[:, -1, :, :].reshape(-1, 1, self.w, self.h)
    )
    _, C, D, H, W = codes.size()
    # print(codes.size())
    code_t = codes.unsqueeze(1).repeat(
        1, self.in_channels, 1, 1, 1, 1
    ).view(
        b * self.in_channels, C, D, H, W
    )

    # Video AE code: code_t = codes.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

    frame_ref = visual_x[:, -1:].unsqueeze(
        2).repeat(1, self.in_channels, 1, 1, 1)
    frame_pair = torch.cat([
        # frame_ref,
        # visual_x.view(- 1, self.in_channels, 1, self.w, self.h)
        visual_x.view(- 1, self.in_channels, 1, self.w, self.h),
        visual_x[:, -1:].unsqueeze(2).repeat(1, self.in_channels, 1, 1, 1)
    ], dim=2).reshape(
        b * self.in_channels, 2, self.w, self.h
    )

    poses = self.encoder_traj(frame_pair)
    theta = euler2mat(poses)
    rot_codes = self.rotate(code_t, theta)
    # voxels = rot_codes.view(
    #     b * self.in_channels, 64, D, H, W
    # )
    # voxel shape: B, C, D, H, W
    flatten_voxel = rot_codes.view(
        b * self.in_channels, 32 * D, H, W
    )

    # print(flatten_voxel.shape)
    # flatten_voxel = self.convs(flatten_voxel)
    # flatten_voxel = flatten_voxel.flatten(1)
    # flatten_voxel = self.visual_projector(flatten_voxel)

    # state_out = self.base(state_x)

    # out = torch.cat([flatten_voxel, state_out], dim=-1)
    # out = self.out_lienar(out)
    return flatten_voxel


class CoarseNatureVoxelTransEncoder(NatureVoxelTransEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder_3d = CoarseNatureEncoder3D(cfg)
    self.encoder_traj = EncoderTraj(cfg)
    self.rotate = NatureRotate(cfg)

    self.height_decoder_class = Res18VoxelHeightDecoder

    self.up_conv = nn.Conv2d(
        self.rotate.output_channel_num * self.encoder_3d.depth,
        self.token_dim, 1
    )
    self.per_modal_tokens = 64

    self.img_decoder_class = CoarseNatureImgDecoder

  def compute_latent_code(
      self, x
  ):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    original_state_x_shape = state_x.shape
    visual_x = visual_x.view(
        -1, self.in_channels * self.camera_num, self.w, self.h
    )
    # visual_x = visual_x[:, -1, :, :]
    b = visual_x.shape[0]

    # b, t, c, h, w = clips.size()
    codes = self.encoder_3d(
        visual_x[:, -1, :, :].reshape(-1, 1, self.w, self.h)
    )
    _, C, D, H, W = codes.size()
    # print(codes.size())
    code_t = codes.unsqueeze(1).repeat(
        1, self.in_channels, 1, 1, 1, 1
    ).view(
        b * self.in_channels, C, D, H, W
    )

    # Video AE code: code_t = codes.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)
    frame_ref = visual_x[:, -1:].unsqueeze(
        2).repeat(1, self.in_channels, 1, 1, 1)
    frame_pair = torch.cat([
        # frame_ref,
        # visual_x.view(- 1, self.in_channels, 1, self.w, self.h)
        visual_x.view(- 1, self.in_channels, 1, self.w, self.h),
        visual_x[:, -1:].unsqueeze(2).repeat(1, self.in_channels, 1, 1, 1)
    ], dim=2).reshape(
        b * self.in_channels, 2, self.w, self.h
    )

    poses = self.encoder_traj(frame_pair)
    theta = euler2mat(poses)
    rot_codes = self.rotate(code_t, theta)
    # voxel shape: B, C, D, H, W
    flatten_voxel = rot_codes.view(
        b * self.in_channels, self.rotate.output_channel_num * D, H, W
    )
    return flatten_voxel


class SmallNatureVoxelTransEncoder(CoarseNatureVoxelTransEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)

    self.encoder_3d = SmallNatureEncoder3D(cfg)
    self.up_conv = nn.Conv2d(
        self.rotate.output_channel_num * self.encoder_3d.depth,
        self.token_dim, 1
    )
    self.per_modal_tokens = 36

    self.img_decoder_class = SmallNatureImgDecoder


class CompactNatureVoxelTransEncoder(CoarseNatureVoxelTransEncoder):
  def __init__(
      self,
      cfg
  ):
    super().__init__(cfg)
    self.encoder_3d = CompactNatureEncoder3D(cfg)
    self.up_conv = nn.Conv2d(
        self.rotate.output_channel_num * self.encoder_3d.depth,
        self.token_dim, 1
    )
    self.per_modal_tokens = 16

    self.img_decoder_class = SmallNatureImgDecoder
