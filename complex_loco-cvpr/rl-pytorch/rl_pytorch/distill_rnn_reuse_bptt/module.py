import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal


class ActorCriticRNNLM(nn.Module):

  def __init__(
      self, encoder,
      obs_shape, states_shape,
      actions_shape, initial_std,
      model_cfg, asymmetric=False
  ):
    super().__init__()

    self.asymmetric = asymmetric

    if model_cfg is None:
      actor_hidden_dim = [256, 256, 256]
      critic_hidden_dim = [256, 256, 256]
      activation = get_activation("selu")
    else:
      actor_hidden_dim = model_cfg['pi_hid_sizes']
      critic_hidden_dim = model_cfg['vf_hid_sizes']
      activation = get_activation(model_cfg['activation'])

    self.clip_std = model_cfg.get("clip_std", False)
    if self.clip_std:
      self.clip_std_upper = model_cfg["clip_std_upper"]
      self.clip_std_lower = model_cfg["clip_std_lower"]

    assert encoder is not None
    self.encoder = encoder(model_cfg['encoder_params'])

    self.rnn_hidden_size = model_cfg["recurrent"]["hidden_size"]
    self.rnn_num_layers = model_cfg["recurrent"]["num_layers"]
    self.rnn = nn.GRU(
        input_size=self.encoder.hidden_states_shape,
        hidden_size=model_cfg["recurrent"]["hidden_size"],
        num_layers=model_cfg["recurrent"]["num_layers"],
        # batch_first=True
    )

    # # Policy
    actor_layers = [
        nn.Linear(self.encoder.hidden_states_shape, actor_hidden_dim[0]),
        activation
    ]
    self.action_shape = actions_shape
    assert self.asymmetric or actor_hidden_dim[0] == critic_hidden_dim[0]
    for l in range(len(actor_hidden_dim)):
      if l == len(actor_hidden_dim) - 1:
        actor_layers.append(
            nn.Linear(actor_hidden_dim[l], *actions_shape))
      else:
        actor_layers.append(
            nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
        actor_layers.append(activation)
    self.actor = nn.Sequential(*actor_layers)

    # Value function
    critic_layers = []
    if self.asymmetric:
      critic_layers.append(
          nn.Linear(*states_shape, critic_hidden_dim[0]))
      critic_layers.append(activation)
    else:
      critic_layers.append(
          nn.Linear(self.encoder.hidden_states_shape, critic_hidden_dim[0]))
      critic_layers.append(activation)

    for l in range(len(critic_hidden_dim)):
      if l == len(critic_hidden_dim) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
      else:
        critic_layers.append(
            nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
        critic_layers.append(activation)
    self.critic = nn.Sequential(*critic_layers)
    print("actor:")
    print(self.actor)
    print("critic:")
    print(self.critic)
    print("rnn:")
    print(self.rnn)
    if encoder is not None:
      print("encoder:")
      print(self.encoder)

    # Action noise
    self.log_std = nn.Parameter(
        np.log(initial_std) * torch.ones(*actions_shape))

    # Initialize the weights like in stable baselines
    actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
    actor_weights.append(0.01)
    critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
    critic_weights.append(1.0)
    self.init_weights(self.actor, actor_weights)
    self.init_weights(self.critic, critic_weights)
    if encoder is not None:
      self.apply(self.init_visual_weights)

  @staticmethod
  def init_weights(sequential, scales):
    [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
     enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

  @staticmethod
  def init_visual_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
      gain = nn.init.calculate_gain('relu')
      nn.init.orthogonal_(m.weight.data, gain)
      if hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.0)

  def forward(self):
    raise NotImplementedError

  def act(self, observations, states, hidden_states):
    original_x_shape = observations.shape
    observations = observations.view(-1, original_x_shape[-1])
    out = self.encoder(
        observations)
    out = out.unsqueeze(0)
    # print(out.shape, hidden_states.shape)
    rnn_out, new_hidden_states = self.rnn(out, hidden_states)
    rnn_out = rnn_out.squeeze(0)
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
    actions_mean = self.actor(rnn_out)
    # print(original_x_shape[:-1])
    # print(original_x_shape[:-1] + torch.Size(self.action_shape,))
    actions_mean = actions_mean.view(
        *(original_x_shape[:-1] + torch.Size(self.action_shape)))

    if self.clip_std:
      self.log_std.copy_(
          torch.clamp(self.log_std, self.clip_std_lower, self.clip_std_upper)
      )
    # print(self.log_std)
    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    # distribution = Normal(actions_mean, self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
    actions = distribution.sample()
    actions_log_prob = distribution.log_prob(actions)

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(rnn_out)

    value = value.view(*(original_x_shape[:-1] + (1,)))

    # return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), \
    #   self.log_std.repeat(actions_mean.shape[0], 1).detach(), new_hidden_states
    return actions, actions_log_prob.detach(), value.detach(), actions_mean, \
        self.log_std.repeat(
        actions_mean.shape[0], 1).detach(), new_hidden_states

  def act_inference(self, observations, hidden_states):
    original_x_shape = observations.shape
    observations = observations.view(-1, original_x_shape[-1])
    out = self.encoder(
        observations
    )
    out = out.unsqueeze(0)
    rnn_out, new_hidden_states = self.rnn(out, hidden_states)
    rnn_out = rnn_out.squeeze(0)
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)

    actions_mean = self.actor(rnn_out)
    actions_mean = actions_mean.view(
        *(original_x_shape[:-1] + torch.Size(self.action_shape)))
    return actions_mean, new_hidden_states

  def evaluate(self, observations, states, actions, hidden_states, return_rnnout=False):
    original_x_shape = observations.shape
    observations = observations.view(-1, original_x_shape[-1])
    out = self.encoder(observations)

    out = out.unsqueeze(0)
    # print(original_x_shape, out.shape, hidden_states.shape)

    # print(out.shape, hidden_states.shape)
    hidden_states = hidden_states.reshape(self.rnn_num_layers, -1,
                                          self.rnn_hidden_size).contiguous()
    # if len(original_x_shape) >= 3:
    #   # Num Transition, Num Layers, Batch, Hidden
    #   # states = states.permute(1, 0, 2, 3)
    #   states = states.reshape(self.rnn_num_layers, -1,
    #                           self.rnn_hidden_size).contiguous()
    # else:
    #   states = states.reshape(self.rnn_num_layers, -1,
    #                           self.rnn_hidden_size).contiguous()

    rnn_out, new_hidden_states = self.rnn(out, hidden_states)
    rnn_out = rnn_out.squeeze(0)
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
    actions_mean = self.actor(rnn_out)
    actions_mean = actions_mean.view(
        *(original_x_shape[:-1] + torch.Size(self.action_shape)))

    if self.clip_std:
      self.log_std.copy_(
          torch.clamp(self.log_std, self.clip_std_lower, self.clip_std_upper)
      )
    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
    # distribution = Normal(actions_mean, self.log_std.exp())

    actions_log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(rnn_out)

    value = value.view(*(original_x_shape[:-1] + (1,)))

    if return_rnnout:
      return actions_log_prob, entropy.unsqueeze(-1), value, actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1), new_hidden_states, out
    return actions_log_prob, entropy.unsqueeze(-1), value, actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1), new_hidden_states


def get_activation(act_name):
  if act_name == "elu":
    return nn.ELU()
  elif act_name == "selu":
    return nn.SELU()
  elif act_name == "relu":
    return nn.ReLU()
  elif act_name == "crelu":
    return nn.ReLU()
  elif act_name == "lrelu":
    return nn.LeakyReLU()
  elif act_name == "tanh":
    return nn.Tanh()
  elif act_name == "sigmoid":
    return nn.Sigmoid()
  else:
    print("invalid activation function!")
    return None
