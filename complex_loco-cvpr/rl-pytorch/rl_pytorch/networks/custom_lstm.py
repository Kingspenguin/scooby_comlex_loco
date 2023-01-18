from torch.nn import Parameter
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import torch.jit as jit
import torch
import torch.nn as nn

'''
Some helper classes for writing custom TorchScript LSTMs.
Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.
A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
'''


def script_lnlstm(cfg):
  '''Returns a ScriptModule that mimics a PyTorch LSTM with LayerNorm.'''

  # The following are not implemented.
  input_size, hidden_size, num_layers = 2 * \
      cfg["input_size"], cfg["hidden_size"], cfg["num_layers"]

  return StackedLSTM(num_layers, LSTMLayer,
                     first_layer_args=[
                         LayerNormLSTMCell, input_size, hidden_size],
                     other_layer_args=[LayerNormLSTMCell, hidden_size,
                                       hidden_size])


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class LayerNormLSTMCell(jit.ScriptModule):
  def __init__(self, input_size, hidden_size):
    super(LayerNormLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
    self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
    # The layernorms provide learnable biases

    ln = nn.LayerNorm

    self.layernorm_i = ln(4 * hidden_size)
    self.layernorm_h = ln(4 * hidden_size)
    self.layernorm_c = ln(hidden_size)

  @jit.script_method
  def forward(self, input, state, mask):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    hx, cx = state
    cx = cx * (1 - mask)
    hx = hx * (1 - mask)

    igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
    hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.transpose(-1, -2)))
    gates = (igates + hgates)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
    hy = outgate * torch.tanh(cy)
    # print("weight_ih", self.weight_ih.shape)
    return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
  def __init__(self, cell, *cell_args):
    super(LSTMLayer, self).__init__()
    self.cell = cell(*cell_args)

  @jit.script_method
  def forward(self, input, state, mask):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
    inputs = input.unbind(0)
    masks = mask.unbind(0)
    assert len(inputs) == len(
        masks), 'inputs and masks must have the same length!'
    outputs = torch.jit.annotate(List[Tensor], [])
    for i in range(len(inputs)):
      out, state = self.cell(inputs[i], state, masks[i])
      outputs += [out]
    return torch.stack(outputs), state


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
  layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                         for _ in range(num_layers - 1)]
  return nn.ModuleList(layers)


class StackedLSTM(jit.ScriptModule):
  __constants__ = ['layers']  # Necessary for iterating through self.layers

  def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
    super(StackedLSTM, self).__init__()
    self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                    other_layer_args)

  @jit.script_method
  def forward(self, input, states, mask):
    # type: (Tensor, List[Tuple[Tensor, Tensor]], Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
    # List[LSTMState]: One state per layer
    output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
    output = input
    # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
    i = 0
    for rnn_layer in self.layers:
      state = states[i]
      output, out_state = rnn_layer(output, state, mask)
      output_states += [out_state]
      i += 1
    return output, output_states


def flatten_states(states):
  states = list(zip(*states))
  assert len(states) == 2
  return [torch.stack(state) for state in states]


def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size,
                               num_layers):
  inp = torch.randn(seq_len, batch, input_size)
  states = [LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))
            for _ in range(num_layers)]
  rnn = script_lnlstm(input_size, hidden_size, num_layers)

  # just a smoke test
  out, out_state = rnn(inp, states)
