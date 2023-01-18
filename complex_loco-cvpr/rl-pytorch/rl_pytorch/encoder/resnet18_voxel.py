from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .submodule import stn, euler2mat, conv
from torch.nn.init import xavier_uniform_, zeros_

from .decoder import Res18VoxelHeightDecoder, Res18VoxelV2HeightDecoder, Res18VoxelV3HeightDecoder
from .decoder import Res18ImgDecoder
from .mlp import RLProjection, MLPBase

from .resnet18 import get_small_resnet18


class ResNet18Encoder3D(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.feature_extraction = get_small_resnet18(
        pretrained=False)
    self.conv3d_1 = nn.ConvTranspose3d(32, 32, 3, stride=1, padding=1)
    self.conv3d_2 = nn.ConvTranspose3d(32, 32, 3, stride=1, padding=1)

  def forward(self, img):
    z_2d = self.feature_extraction(img)
    B, C, H, W = z_2d.shape
    z_3d = z_2d.reshape([-1, 32, 6, H, W])
    z_3d = F.leaky_relu(self.conv3d_1(z_3d))
    z_3d = F.leaky_relu(self.conv3d_2(z_3d))
    return z_3d


class ResNet18EncoderTraj(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    conv_planes = [16, 32, 64, 128, 256]
    self.conv1 = conv(2, conv_planes[0], kernel_size=7)
    self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
    self.conv3 = conv(conv_planes[1], conv_planes[2])
    self.conv4 = conv(conv_planes[2], conv_planes[3])
    self.conv5 = conv(conv_planes[3], conv_planes[4])

    self.pose_pred = nn.Conv2d(conv_planes[4], 6, kernel_size=1, padding=0)

    self.scale_rotate = cfg["scale_rotate"]
    self.scale_translate = cfg["scale_translate"]

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
          zeros_(m.bias)

  def forward(self, input):
    out_conv1 = self.conv1(input)
    out_conv2 = self.conv2(out_conv1)
    out_conv3 = self.conv3(out_conv2)
    out_conv4 = self.conv4(out_conv3)
    out_conv5 = self.conv5(out_conv4)

    pose = self.pose_pred(out_conv5)
    pose = pose.mean(3).mean(2)
    pose = pose.view(pose.size(0), 6)

    pose_r = pose[:, :3] * self.scale_rotate
    pose_t = pose[:, 3:] * self.scale_translate

    pose_final = torch.cat([pose_r, pose_t], 1)

    return pose_final


# class Decoder(nn.Module):
#   def __init__(self, cfg):
#     super(Decoder, self).__init__()
#     self.depth_3d = 32
#     self.conv3 = nn.Conv2d(512, 256, 1)
#     self.upconv3 = nn.ConvTranspose2d(
#         256, 128, kernel_size=4, stride=2, padding=1)
#     self.upconv4 = nn.ConvTranspose2d(
#         128, 64, kernel_size=4, stride=2, padding=1)
#     self.upconv_final = nn.ConvTranspose2d(
#         64, 1, kernel_size=3, stride=1, padding=1)

#   def forward(self, code):
#     code = code.view(-1, code.size(1) * code.size(2),
#                      code.size(3), code.size(4))
#     code = F.leaky_relu(self.conv3(code))
#     code = F.leaky_relu(self.upconv3(code))
#     code = F.leaky_relu(self.upconv4(code))
#     output = self.upconv_final(code)
#     return output


class ResNet18Rotate(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.padding_mode = cfg["padding_mode"]
    self.conv3d_1 = nn.Conv3d(32, 32, 3, padding=1)
    self.conv3d_2 = nn.Conv3d(32, 32, 3, padding=1)

  def forward(self, code, theta):
    rot_code = stn(code, theta, self.padding_mode)
    rot_code = F.leaky_relu(self.conv3d_1(rot_code))
    rot_code = F.leaky_relu(self.conv3d_2(rot_code))
    return rot_code


class ResNet18VoxelEncoder(nn.Module):
  def __init__(
      self,
      cfg
  ):
    super().__init__()

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
    # self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]

    assert self.camera_num == 1, "suport 1 cam for now"

    self.encoder_3d = ResNet18Encoder3D(cfg)
    self.encoder_traj = ResNet18EncoderTraj(cfg)
    self.rotate = ResNet18Rotate(cfg)

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 6, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), self.non_linear_func(),
        nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), self.non_linear_func(),
    ]
    self.convs = nn.Sequential(*layer_list)

    self.height_decoder_class = Res18VoxelHeightDecoder
    self.img_decoder_class = Res18ImgDecoder

    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg["hidden_dims"][-1],
        proj_function=self.non_linear_func,
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
    )

    self.hidden_states_shape = cfg["hidden_dims"][-1] * 2
    self.out_lienar = RLProjection(
        cfg["hidden_dims"][-1] * 2,
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
    # print(visual_x.shape)
    b = visual_x.shape[0]

    # b, t, c, h, w = clips.size()
    codes = self.encoder_3d(
        visual_x.reshape(-1, 1, self.w, self.h)
    )
    _, C, D, H, W = codes.size()
    # print(codes.size())
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
    # print(rot_codes.shape)
    voxels = rot_codes.view(
        b, self.in_channels, 32, D, H, W
    ).mean(1)
    # voxel shape: B, C, D, H, W
    flatten_voxel = voxels.view(b, 32 * D, H, W)

    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)

    state_out = self.base(state_x)

    out = torch.cat([flatten_voxel, state_out], dim=-1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    if return_3d_code:
      return out, voxels
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

    # No gradient for Encoder:
    # code_t = code_t.detach()

    # Video AE code: code_t = codes.unsqueeze(1).repeat(1, t, 1, 1, 1, 1).view(b * t, C, H, W, D)

    frame_ref = visual_x[:, -1:].unsqueeze(
        2).repeat(1, self.in_channels, 1, 1, 1)
    frame_pair = torch.cat([
        frame_ref,
        visual_x.view(- 1, self.in_channels, 1, self.w, self.h)
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


class ResNet18VoxelV2Encoder(ResNet18VoxelEncoder):
  def __init__(self, cfg):
    super().__init__(cfg)
    # self.visual_projector = RLProjection(
    #     in_dim=256,
    #     out_dim=cfg["hidden_dims"][-1]
    # )

    layer_list = [
        nn.Conv2d(
            in_channels=64 * 6, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)

    self.height_decoder_class = Res18VoxelV2HeightDecoder
    self.img_decoder_class = Res18ImgDecoder

  def forward(self, x, return_3d_code=False):
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
    # print(codes.size())
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
        b, self.in_channels, 64, D, H, W
    ).mean(1)
    # voxel shape: B, C, D, H, W

    voxels = voxels.permute(
        0, 1, 3, 2, 4
    )
    flatten_voxel = voxels.reshape(b, 64 * H, D, W)

    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)

    state_out = self.base(state_x)

    out = torch.cat([flatten_voxel, state_out], dim=-1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    if return_3d_code:
      return out, voxels
    return out


class ResNet18VoxelV3Encoder(ResNet18VoxelEncoder):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.visual_projector = RLProjection(
        in_dim=1024,
        out_dim=cfg["hidden_dims"][-1],
        proj_function=self.non_linear_func,
    )
    layer_list = [
        nn.Conv2d(
            in_channels=64 * 6, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.conv1 = nn.Sequential(*layer_list)

    layer_list = [
        nn.Conv2d(
            in_channels=64 * 8, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.conv2 = nn.Sequential(*layer_list)

    self.height_decoder_class = Res18VoxelV3HeightDecoder
    self.img_decoder_class = Res18ImgDecoder

  def forward(self, x, return_3d_code=False):
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
    # print(codes.size())
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
        b, self.in_channels, 64, D, H, W
    ).mean(1)
    # voxel shape: B, C, D, H, W
    flatten_ego_voxel = voxels.reshape(b, 64 * D, H, W)

    bird_voxels = voxels.permute(
        0, 1, 3, 2, 4
    )
    flatten_bird_voxel = bird_voxels.reshape(b, 64 * H, D, W)

    # print(flatten_voxel.shape)
    flatten_bird_voxel = self.conv1(flatten_bird_voxel)
    flatten_ego_voxel = self.conv2(flatten_ego_voxel)
    # print(flatten_voxel.shape)
    flatten_ego_voxel = flatten_ego_voxel.flatten(1)
    flatten_bird_voxel = flatten_bird_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(
        torch.cat([
            flatten_bird_voxel, flatten_ego_voxel
        ], dim=-1)
    )

    state_out = self.base(state_x)

    out = torch.cat([flatten_voxel, state_out], dim=-1)
    out = self.out_lienar(out)

    if len(original_state_x_shape) >= 3:
      out = out.view(*original_state_x_shape[:2], -1)
    if return_3d_code:
      return out, (bird_voxels, voxels)
    return out
