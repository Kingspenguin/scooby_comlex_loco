import numpy as np
import torch.nn as nn


def _fanin_init(tensor, alpha=0):
  size = tensor.size()
  if len(size) == 2:
    fan_in = size[0]
  elif len(size) > 2:
    fan_in = np.prod(size[1:])
  else:
    raise Exception("Shape must be have dimension at least 2.")
  # bound = 1. / np.sqrt(fan_in)
  bound = np.sqrt(1. / ((1 + alpha * alpha) * fan_in))
  return tensor.data.uniform_(-bound, bound)


def _uniform_init(tensor, param=3e-3):
  return tensor.data.uniform_(-param, param)


def _constant_bias_init(tensor, constant=0.1):
  tensor.data.fill_(constant)


def layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init):
  weight_init(layer.weight)
  bias_init(layer.bias)


def basic_init(layer):
  layer_init(layer, weight_init=_fanin_init, bias_init=_constant_bias_init)


def uniform_init(layer):
  layer_init(layer, weight_init=_uniform_init, bias_init=_uniform_init)


def _orthogonal_init(tensor, gain=np.sqrt(2)):
  nn.init.orthogonal_(tensor, gain=gain)


def orthogonal_init(layer, scale=np.sqrt(2), constant=0):
  layer_init(
    layer,
    weight_init=lambda x: _orthogonal_init(x, gain=scale),
    bias_init=lambda x: _constant_bias_init(x, 0))


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
