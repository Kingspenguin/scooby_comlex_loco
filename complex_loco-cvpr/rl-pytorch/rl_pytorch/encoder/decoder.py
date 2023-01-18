import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLPBase, RLProjection


class MLPHeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )
    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    out = self.decoder(latent_code)
    height = self.predict(out)
    return height


class NatureVoxelHeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 8, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)
    self.visual_projector = RLProjection(
        in_dim=256,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )
    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )
    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    # voxel shape: B, C, D, H, W

    B, C, D, H, W = latent_code.size()
    flatten_voxel = latent_code.view(B, C * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)
    # print(flatten_voxel.shape)
    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class NatureVoxelV2HeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 6, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)
    self.visual_projector = RLProjection(
        in_dim=256,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )
    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )
    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    # voxel shape: B, C, H, D, W
    # permuted in voxel encoder
    B, C, H, D, W = latent_code.size()
    # latent_code = latent_code.permute(
    #     0, 1, 3, 2, 4
    # )
    flatten_voxel = latent_code.reshape(B, C * H, D, W)
    # flatten_voxel = latent_code.view(B, C * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)
    # print(flatten_voxel.shape)
    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class NatureVoxelV3HeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    layer_list = [
        nn.Conv2d(
            in_channels=64 * 8, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.conv1 = nn.Sequential(*layer_list)

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 8, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.conv2 = nn.Sequential(*layer_list)

    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )

    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )

    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    bird_view_latent, egocentric_latent = latent_code
    # voxel shape: B, C, H, D, W
    # permuted in voxel encoder
    B, C, D, H, W = egocentric_latent.size()

    flatten_ego_voxel = egocentric_latent.reshape(B, C * D, H, W)
    flatten_bird_voxel = bird_view_latent.reshape(B, C * H, D, W)

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

    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class Res18ImgDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.depth_3d = 64
    self.conv3 = nn.Conv2d(192, 256, 1)
    self.upconv3 = nn.ConvTranspose2d(
        256, 128, kernel_size=4, stride=2, padding=1)
    self.upconv4 = nn.ConvTranspose2d(
        128, 64, kernel_size=4, stride=2, padding=1)
    self.upconv_final = nn.ConvTranspose2d(
        64, 1, kernel_size=4, stride=2, padding=1)

  def forward(self, code):
    # print(code.shape)
    # code = code.reshape(-1, code.size(1) * code.size(2),
    #                     code.size(3), code.size(4))
    code = F.leaky_relu(self.conv3(code))
    code = F.leaky_relu(self.upconv3(code))
    code = F.leaky_relu(self.upconv4(code))
    output = self.upconv_final(code)
    output = F.interpolate(output, (64, 64), mode='bilinear')
    return output


class Res18VoxelHeightDecoder(NatureVoxelHeightDecoder):
  def __init__(self, cfg):
    super().__init__(cfg)

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
    self.convs = nn.Sequential(*layer_list)
    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )

  def forward(self, latent_code):
    # voxel shape: B, C, D, H, W

    B, C, D, H, W = latent_code.size()
    flatten_voxel = latent_code.view(B, C * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)
    # print(flatten_voxel.shape)
    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class Res18VoxelV2HeightDecoder(Res18VoxelHeightDecoder):
  def forward(self, latent_code):
    # voxel shape: B, C, H, D, W
    # permuted in voxel encoder
    B, C, H, D, W = latent_code.size()
    # latent_code = latent_code.permute(
    #     0, 1, 3, 2, 4
    # )
    flatten_voxel = latent_code.reshape(B, C * H, D, W)
    # flatten_voxel = latent_code.view(B, C * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)
    # print(flatten_voxel.shape)
    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class Res18VoxelV3HeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
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

    self.visual_projector = RLProjection(
        in_dim=1024,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )

    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )

    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    bird_view_latent, egocentric_latent = latent_code
    # voxel shape: B, C, H, D, W
    # permuted in voxel encoder
    B, C, D, H, W = egocentric_latent.size()

    flatten_ego_voxel = egocentric_latent.reshape(B, C * D, H, W)
    flatten_bird_voxel = bird_view_latent.reshape(B, C * H, D, W)

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

    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class CoarseNatureVoxelHeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 16, out_channels=256,
            kernel_size=5, stride=2, padding=0
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=3, stride=1, padding=0
        ), nn.ReLU(),
        # nn.Conv2d(
        #     in_channels=128, out_channels=64,
        #     kernel_size=5, stride=2, padding=2
        # ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)
    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )
    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )
    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    # voxel shape: B, C, D, H, W

    B, C, D, H, W = latent_code.size()
    flatten_voxel = latent_code.view(B, C * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)
    # print(flatten_voxel.shape)
    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class CoarseNatureVoxelV2HeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 6, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)
    self.visual_projector = RLProjection(
        in_dim=256,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )
    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )
    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    # voxel shape: B, C, H, D, W
    # permuted in voxel encoder
    B, C, H, D, W = latent_code.size()
    # latent_code = latent_code.permute(
    #     0, 1, 3, 2, 4
    # )
    flatten_voxel = latent_code.reshape(B, C * H, D, W)
    # flatten_voxel = latent_code.view(B, C * D, H, W)
    # print(flatten_voxel.shape)
    flatten_voxel = self.convs(flatten_voxel)
    # print(flatten_voxel.shape)
    flatten_voxel = flatten_voxel.flatten(1)
    # print(flatten_voxel.shape)
    flatten_voxel = self.visual_projector(flatten_voxel)
    # print(flatten_voxel.shape)
    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class CoarseNatureVoxelV3HeightDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 6, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.conv1 = nn.Sequential(*layer_list)

    layer_list = [
        nn.Conv2d(
            in_channels=32 * 8, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.conv2 = nn.Sequential(*layer_list)

    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg['encoder_params']["hidden_dims"][-1]
    )

    self.decoder = MLPBase(
        input_shape=cfg['encoder_params']["hidden_dims"][-1],
        hidden_shapes=cfg["height_hidden_dims"],
    )

    self.predict = nn.Linear(
        cfg["height_hidden_dims"][-1],
        cfg["height_dim"]
    )

  def forward(self, latent_code):
    bird_view_latent, egocentric_latent = latent_code
    # voxel shape: B, C, H, D, W
    # permuted in voxel encoder
    B, C, D, H, W = egocentric_latent.size()

    flatten_ego_voxel = egocentric_latent.reshape(B, C * D, H, W)
    flatten_bird_voxel = bird_view_latent.reshape(B, C * H, D, W)

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

    out = self.decoder(flatten_voxel)
    height = self.predict(out)
    return height


class CoarseNatureImgDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.depth_3d = 64
    self.rotate_conv3d = cfg["rotate_conv3d"]
    input_channel = 16 * 8
    if self.rotate_conv3d:
      input_channel = 32 * 8
    self.conv3 = nn.Conv2d(
        input_channel,
        256, 1
    )
    self.upconv3 = nn.ConvTranspose2d(
        256, 128, kernel_size=4, stride=2, padding=1)
    self.upconv4 = nn.ConvTranspose2d(
        128, 64, kernel_size=4, stride=2, padding=1)
    self.upconv_final = nn.ConvTranspose2d(
        64, 1, kernel_size=4, stride=2, padding=1)

  def forward(self, code):
    # print(code.shape)
    # code = code.reshape(-1, code.size(1) * code.size(2),
    #                     code.size(3), code.size(4))
    code = F.leaky_relu(self.conv3(code))
    code = F.leaky_relu(self.upconv3(code))
    code = F.leaky_relu(self.upconv4(code))
    output = self.upconv_final(code)
    return output


class SmallNatureImgDecoder(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.depth_3d = 64
    self.rotate_conv3d = cfg.get("rotate_conv3d", True)
    input_channel = 16 * 6
    if self.rotate_conv3d:
      input_channel = 32 * 6
    self.conv3 = nn.Conv2d(
        input_channel,
        256, 1
    )
    self.upconv3 = nn.ConvTranspose2d(
        256, 128, kernel_size=4, stride=2, padding=1)
    self.upconv4 = nn.ConvTranspose2d(
        128, 64, kernel_size=4, stride=2, padding=1)
    self.upconv_final = nn.ConvTranspose2d(
        64, 1, kernel_size=4, stride=2, padding=1)

  def forward(self, code):
    # print(code.shape)
    # code = code.reshape(-1, code.size(1) * code.size(2),
    #                     code.size(3), code.size(4))
    code = F.leaky_relu(self.conv3(code))
    code = F.leaky_relu(self.upconv3(code))
    code = F.leaky_relu(self.upconv4(code))
    output = self.upconv_final(code)
    output = F.interpolate(output, (64, 64), mode='bilinear')
    return output
