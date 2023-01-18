import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal


class ActorCriticLMFull(nn.Module):

  def __init__(
      self, encoder,
      obs_shape, states_shape,
      actions_shape, initial_std,
      model_cfg,
      # asymmetric=False
  ):
    super().__init__()

    # self.asymmetric = asymmetric

    if model_cfg is None:
      actor_hidden_dim = [256, 256, 256]
      critic_hidden_dim = [256, 256, 256]
      activation = get_activation("selu")
    else:
      enc_hidden_dim = model_cfg['enc_hid_sizes']
      actor_hidden_dim = model_cfg['pi_hid_sizes']
      critic_hidden_dim = model_cfg['vf_hid_sizes']
      state_critic_hidden_dim = model_cfg['state_vf_hid_sizes']
      activation = get_activation(model_cfg['activation'])

    self.clip_std = model_cfg.get("clip_std", False)
    if self.clip_std:
      self.clip_std_upper = model_cfg["clip_std_upper"]
      self.clip_std_lower = model_cfg["clip_std_lower"]

    assert encoder is not None
    self.encoder = encoder(model_cfg['encoder_params'])
    # # Policy

    ac_enc_layers = []
    ac_enc_layers.append(
        nn.Linear(self.encoder.hidden_states_shape, enc_hidden_dim[0])
    )
    ac_enc_layers.append(activation)

    for l in range(len(enc_hidden_dim)):
      if l == len(enc_hidden_dim) - 1:
        ac_enc_layers.append(
            nn.Linear(enc_hidden_dim[l], actor_hidden_dim[0]))
      else:
        ac_enc_layers.append(
            nn.Linear(enc_hidden_dim[l], enc_hidden_dim[l + 1]))
      ac_enc_layers.append(activation)
    self.ac_enc = nn.Sequential(*ac_enc_layers)

    actor_layers = []
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
    for l in range(len(critic_hidden_dim)):
      if l == len(critic_hidden_dim) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
      else:
        critic_layers.append(
            nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
        critic_layers.append(activation)
    self.critic = nn.Sequential(*critic_layers)

    # Value function
    state_critic_layers = [
        nn.Linear(*states_shape, state_critic_hidden_dim[0]),
        activation
    ]
    for l in range(len(state_critic_hidden_dim)):
      if l == len(state_critic_hidden_dim) - 1:
        state_critic_layers.append(nn.Linear(state_critic_hidden_dim[l], 1))
      else:
        state_critic_layers.append(
            nn.Linear(state_critic_hidden_dim[l], state_critic_hidden_dim[l + 1]))
        state_critic_layers.append(activation)
    self.state_critic = nn.Sequential(*state_critic_layers)

    print("actor:")
    print(self.actor)
    print("critic:")
    print(self.critic)
    print("state critic:")
    print(self.state_critic)
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

  @ staticmethod
  def init_weights(sequential, scales):
    [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
     enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

  @ staticmethod
  def init_visual_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
      gain = nn.init.calculate_gain('relu')
      nn.init.orthogonal_(m.weight.data, gain)
      if hasattr(m.bias, 'data'):
        m.bias.data.fill_(0.0)

  def forward(self):
    raise NotImplementedError

  def act(self, observations, states):
    out = self.encoder(
        observations)
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
    out = self.ac_enc(out)
    actions_mean = self.actor(out)

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

    # if self.asymmetric:
    value = self.state_critic(states)
    obs_value = self.critic(out)

    return actions.detach(), actions_log_prob.detach(),\
        value.detach(), obs_value.detach(), \
        actions_mean.detach(), self.log_std.repeat(
        actions_mean.shape[0], 1).detach()

  def act_inference(self, observations):
    out = self.encoder(
        observations
    )
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
    out = self.ac_enc(out)
    actions_mean = self.actor(out)
    return actions_mean

  def evaluate(self, observations, states, actions, return_rnnout=False):
    out = self.encoder(observations)
    out = self.ac_enc(out)
    actions_mean = self.actor(out)

    if self.clip_std:
      self.log_std.copy_(
          torch.clamp(self.log_std, self.clip_std_lower, self.clip_std_upper)
      )
    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
    # distribution = Normal(actions_mean, self.log_std.exp())

    actions_log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()

    value = self.state_critic(states)
    obs_value = self.critic(out)

    if return_rnnout:
      return actions_log_prob, entropy.unsqueeze(-1),\
          value, obs_value,\
          actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(
          actions_mean.shape[0], actions_mean.shape[1], 1), out
    return actions_log_prob, entropy.unsqueeze(-1),\
        value, obs_value,\
        actions_mean, self.log_std.unsqueeze(0).unsqueeze(
        1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1)


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
