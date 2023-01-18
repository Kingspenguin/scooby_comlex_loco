import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_


def xavier_uniform_init(module, gain=1.0):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
    nn.init.xavier_uniform_(module.weight.data, gain)
    nn.init.constant_(module.bias.data, 0)
  return module


class ResidualBlock(nn.Module):
  def __init__(self,
               in_channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(
        in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    out = nn.ReLU()(x)
    out = self.conv1(out)
    out = nn.ReLU()(out)
    out = self.conv2(out)
    return out + x


class ImpalaBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ImpalaBlock, self).__init__()
    self.conv = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
    self.res1 = ResidualBlock(out_channels)
    self.res2 = ResidualBlock(out_channels)

  def forward(self, x):
    x = self.conv(x)
    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    x = self.res1(x)
    x = self.res2(x)
    return x


class Flatten(nn.Module):
  def forward(self, x):
    # base_shape = v.shape[-2]
    return x.view(x.size(0), -1)


class ImpalaEncoder(nn.Module):
  def __init__(
      self,
      in_channels,
      channels,
      flatten=True,
  ):
    super(ImpalaEncoder, self).__init__()
    assert len(channels) == 3
    self.block1 = ImpalaBlock(
        in_channels=in_channels, out_channels=channels[0]
    )
    self.block2 = ImpalaBlock(
        in_channels=channels[0], out_channels=channels[1]
    )
    self.block3 = ImpalaBlock(
        in_channels=channels[1], out_channels=channels[2]
    )

    self.flatten = flatten

    self.output_dim = channels[2] * 8 * 8
    self.apply(xavier_uniform_init)

  def forward(self, x, detach=False):

    view_shape = x.size()[:-3] + torch.Size([-1])
    x = x.view(torch.Size(
        [np.prod(x.size()[:-3])]) + x.size()[-3:])
    # out = self.seq_convs(x)
    # return out

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = nn.ReLU()(x)
    if self.flatten:
      x = Flatten()(x).view(view_shape)
    if detach:
      x = x.detach()
    return x
