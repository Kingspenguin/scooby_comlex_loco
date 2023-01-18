import torch
import math
import torchvision
import torch.nn as nn
from .mlp import MLPBase, RLProjection
from .decoder import MLPHeightDecoder
from .decoder import Res18ImgDecoder
from .decoder import Res18VoxelHeightDecoder, Res18VoxelV2HeightDecoder, Res18VoxelV3HeightDecoder


def get_small_resnet18(pretrained=True, flatten=False, input_channels=1):
  model = torchvision.models.resnet18(
      pretrained=pretrained
  )
  feature = [
      nn.Conv2d(input_channels, 64, 7, stride=3, padding=3, bias=False),
      # nn.ReLU(),
      # nn.Conv2d(64, 64, 5, stride=2, padding=0, bias=False),
  ] + list(model.children())[1:-4] + [
      nn.Conv2d(128, 192, 3, stride=1, padding=1, bias=False)
  ]
  if flatten:
    feature.append(
        nn.Flatten()
    )
  feature = nn.Sequential(*feature)
  feature.output_dim = 6 * 6 * 192
  return feature


class ResNet18CatEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_base = get_small_resnet18(
        pretrained=False, flatten=False, input_channels=cfg["in_channels"]
    )
    self.flatten = nn.Flatten()

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    layer_list = [
        nn.Conv2d(
            in_channels=192, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), self.non_linear_func(),
        nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), self.non_linear_func(),
    ]
    self.convs = nn.Sequential(*layer_list)

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]

    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg["hidden_dims"][-1],
        proj_function=self.non_linear_func,
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]
    self.out_lienar = RLProjection(
        self.hidden_states_shape,
        self.hidden_states_shape,
        proj_function=self.non_linear_func,
    )

    self.height_decoder_class = MLPHeightDecoder
    self.img_decoder_class = Res18ImgDecoder

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = self.convs(visual_out)
    visual_out = self.flatten(visual_out)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)

    out = self.out_lienar(torch.cat([visual_out, state_out], dim=-1))
    if return_3d_code:
      return out, visual_out
    return out


class ResNet18MultiStepEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))
    self.state_dim = cfg["state_dim"]

    # assert self.camera_num == 1, "suport 1 cam for now"

    self.visual_base = get_small_resnet18(
        pretrained=False, flatten=False, input_channels=1
    )
    self.flatten = nn.Flatten()

    layer_list = [
        nn.Conv2d(
            in_channels=192, out_channels=128,
            kernel_size=4, stride=2, padding=2
        ), self.non_linear_func(),
        nn.Conv2d(
            in_channels=128, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), self.non_linear_func(),
    ]
    self.convs = nn.Sequential(*layer_list)

    self.visual_projector = RLProjection(
        in_dim=512,
        out_dim=cfg["hidden_dims"][-1],
        proj_function=self.non_linear_func,
    )

    self.base = MLPBase(
        input_shape=cfg["state_dim"],
        hidden_shapes=cfg["hidden_dims"],
        activation_func=self.non_linear_func,
        **kwargs
    )

    self.hidden_states_shape = 2 * cfg["hidden_dims"][-1]
    self.out_lienar = RLProjection(
        self.hidden_states_shape,
        self.hidden_states_shape,
        proj_function=self.non_linear_func,
    )

    self.height_decoder_class = Res18VoxelHeightDecoder
    self.img_decoder_class = Res18ImgDecoder

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    # original_state_x_shape = state_x.shape

    state_x = state_x.view(-1, self.state_dim)
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)

    b = visual_x.shape[0]

    visual_out = self.visual_base(
        visual_x.reshape(-1, 1, self.w, self.h)
    )
    _, C, H, W = visual_out.shape
    visual_out = visual_out.reshape(
        b, self.in_channels, C, H, W
    ).mean(1)
    visual_out = self.convs(visual_out)
    visual_out = visual_out.flatten(1)
    visual_out = self.visual_projector(visual_out)

    # visual_out = self.visual_base(visual_x)
    # visual_out = self.flatten(visual_out)
    state_out = self.base(state_x)

    out = self.out_lienar(torch.cat([visual_out, state_out], dim=-1))
    if return_3d_code:
      return out, visual_out
    return out
