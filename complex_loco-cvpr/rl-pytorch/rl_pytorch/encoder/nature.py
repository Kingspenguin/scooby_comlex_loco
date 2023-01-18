import torch
import torch.nn
from rl_pytorch.encoder.mlp import *
from rl_pytorch.encoder.init import *
import math
from .decoder import MLPHeightDecoder


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


class WideNatureEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      channels,
      groups=1,
      flatten=True,
      **kwargs
  ):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super().__init__()
    self.groups = groups
    assert len(channels) == 3
    layer_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=channels[0] * self.groups,
                  kernel_size=7, stride=3, padding=3, padding_mode='reflect'), nn.ReLU(),
        nn.Conv2d(in_channels=channels[0] * self.groups, out_channels=channels[1] * self.groups,
                  kernel_size=5, stride=2, padding=2, padding_mode='reflect'), nn.ReLU(),
        nn.Conv2d(in_channels=channels[1] * self.groups, out_channels=channels[2] * self.groups,
                  kernel_size=5, stride=2, padding=2, padding_mode='reflect'), nn.ReLU(),
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


class CoarseNatureEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      channels,
      groups=1,
      flatten=True,
      **kwargs
  ):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super().__init__()
    self.groups = groups
    assert len(channels) == 3
    layer_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=channels[0] * self.groups,
                  kernel_size=7, stride=3, padding=3, padding_mode='reflect'), nn.ReLU(),
        nn.Conv2d(in_channels=channels[0] * self.groups, out_channels=channels[1] * self.groups,
                  kernel_size=7, stride=3, padding=3, padding_mode='reflect'), nn.ReLU(),
        nn.Conv2d(in_channels=channels[1] * self.groups, out_channels=channels[2] * self.groups,
                  kernel_size=3, stride=1, padding=1, padding_mode='reflect'), nn.ReLU(),
    ]
    if flatten:
      layer_list.append(
          nn.Flatten()
      )
    self.layers = nn.Sequential(*layer_list)

    self.output_dim = 8 * 8 * 128 * self.groups
    self.apply(orthogonal_init)

  def forward(self, x):
    x = x.view(torch.Size(
        [np.prod(x.size()[:-3])]) + x.size()[-3:])
    x = self.layers(x)
    return x


class SmallNatureEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      channels,
      groups=1,
      flatten=True,
      **kwargs
  ):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super().__init__()
    self.groups = groups
    assert len(channels) == 3
    layer_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=channels[0] * self.groups,
                  kernel_size=7, stride=3, padding=3, padding_mode='reflect'), nn.ReLU(),
        nn.Conv2d(in_channels=channels[0] * self.groups, out_channels=channels[1] * self.groups,
                  kernel_size=5, stride=2, padding=2, padding_mode='reflect'), nn.ReLU(),
        nn.Conv2d(in_channels=channels[1] * self.groups, out_channels=channels[2] * self.groups,
                  kernel_size=5, stride=2, padding=2, padding_mode='reflect'), nn.ReLU(),
    ]
    if flatten:
      layer_list.append(
          nn.Flatten()
      )
    self.layers = nn.Sequential(*layer_list)

    self.output_dim = 6 * 6 * channels[2] * self.groups
    self.apply(orthogonal_init)

  def forward(self, x):
    x = x.view(torch.Size(
        [np.prod(x.size()[:-3])]) + x.size()[-3:])
    x = self.layers(x)
    return x


class CompactNatureEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      channels,
      groups=1,
      flatten=True,
      **kwargs
  ):
    """
    input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
    filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
    use_batchnorm: (bool) whether to use batchnorm
    """
    super().__init__()
    self.groups = groups
    layer_list = [
        nn.Conv2d(in_channels=in_channels, out_channels=channels[0] * self.groups,
                  kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=channels[0] * self.groups, out_channels=channels[1] * self.groups,
                  kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=channels[1] * self.groups, out_channels=channels[2] * self.groups,
                  kernel_size=3, stride=1), nn.ReLU(),
    ]
    if flatten:
      layer_list.append(
          nn.Flatten()
      )
    self.layers = nn.Sequential(*layer_list)

    self.output_dim = 4 * 4 * channels[2]
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


class NatureCatEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
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
    self.out_lienar = RLProjection(
        self.hidden_states_shape,
        self.hidden_states_shape
    )

    self.height_decoder_class = MLPHeightDecoder

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)
    visual_out = self.visual_base(visual_x)
    visual_out = self.visual_projector(visual_out)
    state_out = self.base(state_x)

    out = self.out_lienar(torch.cat([visual_out, state_out], dim=-1))
    if return_3d_code:
      return out, visual_out
    return out


class CoarseNatureCatEncoder(nn.Module):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__()
    self.visual_base = CoarseNatureEncoder(
        cfg["in_channels"], channels=[32, 64, 128], flatten=False
    )

    if cfg["activation"] == 'relu':
      self.non_linear_func = nn.ReLU
    elif cfg["activation"] == 'elu':
      self.non_linear_func = nn.ELU

    self.in_channels = cfg["in_channels"]
    self.visual_dim = cfg["visual_dim"]
    self.w = self.h = int(math.sqrt(self.visual_dim // self.in_channels))

    self.state_dim = cfg["state_dim"]

    layer_list = [
        nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)

    self.visual_projector = RLProjection(
        in_dim=2 * 2 * 128,
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

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]
    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)

    visual_out = self.visual_base(visual_x)
    flatten_visual_out = self.convs(visual_out).flatten(1)
    flatten_visual_out = self.visual_projector(flatten_visual_out)
    state_out = self.base(state_x)

    out = self.out_lienar(torch.cat([
        flatten_visual_out, state_out
    ], dim=-1))
    if return_3d_code:
      return out, visual_out
    return out


class SmallNatureCatEncoder(CoarseNatureCatEncoder):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__(cfg)
    self.visual_base = SmallNatureEncoder(
        cfg["in_channels"], channels=[32, 64, 6 * 16], flatten=False
    )

    layer_list = [
        nn.Conv2d(
            in_channels=6 * 16, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)


class CoarseNatureMultiStepEncoder(nn.Module):
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

    self.visual_base = CoarseNatureEncoder(
        1, channels=[32, 64, 128], flatten=False
    )

    layer_list = [
        nn.Conv2d(
            in_channels=128, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)

    self.visual_projector = RLProjection(
        in_dim=2 * 2 * 128,
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

  def forward(self, x, return_3d_code=False):
    state_x, visual_x = x[..., :self.state_dim], x[..., self.state_dim:]

    visual_x = visual_x.view(-1, self.in_channels, self.w, self.h)

    b = visual_x.shape[0]

    visual_out = self.visual_base(
        visual_x.reshape(-1, 1, self.w, self.h)
    )
    _, C, H, W = visual_out.shape
    visual_out = visual_out.reshape(
        b, self.in_channels, C, H, W
    ).mean(1)
    flatten_visual_out = self.convs(visual_out).flatten(1)
    flatten_visual_out = self.visual_projector(flatten_visual_out)
    state_out = self.base(state_x)

    out = self.out_lienar(torch.cat([
        flatten_visual_out, state_out
    ], dim=-1))
    if return_3d_code:
      return out, visual_out
    return out


class SmallNatureMultiStepEncoder(CoarseNatureMultiStepEncoder):
  def __init__(
      self,
      cfg,
      **kwargs
  ):
    super().__init__(cfg)
    self.visual_base = SmallNatureEncoder(
        1, channels=[32, 64, 6 * 16], flatten=False
    )

    layer_list = [
        nn.Conv2d(
            in_channels=6 * 16, out_channels=256,
            kernel_size=4, stride=2, padding=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=256, out_channels=128,
            kernel_size=4, stride=2, padding=1
        ), nn.ReLU(),
    ]
    self.convs = nn.Sequential(*layer_list)


class WideNatureCatEncoder(NatureCatEncoder):
  def __init__(self,
               cfg,
               **kwargs):
    super().__init__(
        cfg,
        **kwargs
    )
    self.visual_base = WideNatureEncoder(
        cfg["in_channels"], channels=[32, 64, 128]
    )
    self.visual_projector = RLProjection(
        in_dim=self.visual_base.output_dim,
        out_dim=cfg["hidden_dims"][-1]
    )

    self.height_decoder_class = MLPHeightDecoder


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
    visual_out = visual_out.view(
        *proprioceptive_non_recurrent_x.shape[:2], -1)
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
        nn.ReLU(), nn.Linear(cfg["hidden_dims"]
                             [-1], cfg["hidden_dims"][-1])
    )
    self.neck_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.neck_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]
                     ), nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"]
                             [-1], cfg["hidden_dims"][-1])
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
        nn.ReLU(), nn.Linear(cfg["hidden_dims"]
                             [-1], cfg["hidden_dims"][-1])
    )
    self.neck_attention = nn.MultiheadAttention(cfg["hidden_dims"][-1], 1)
    self.neck_mlp = nn.Sequential(
        nn.LayerNorm(cfg["hidden_dims"][-1]),
        nn.Linear(cfg["hidden_dims"][-1], cfg["hidden_dims"][-1]),
        nn.ReLU(), nn.Linear(cfg["hidden_dims"]
                             [-1], cfg["hidden_dims"][-1])
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
        nn.ReLU(), nn.Linear(cfg["hidden_dims"]
                             [-1], cfg["hidden_dims"][-1])
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
