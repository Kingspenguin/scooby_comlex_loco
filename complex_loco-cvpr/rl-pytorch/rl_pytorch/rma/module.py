import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from rl_pytorch.networks import EFEncoder, AdaptationModule


class RMAActorCritic(nn.Module):

  def __init__(self, encoder, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
    super(RMAActorCritic, self).__init__()

    self.asymmetric = asymmetric

    if model_cfg is None:
      actor_hidden_dim = [256, 256, 256]
      critic_hidden_dim = [256, 256, 256]
      activation = get_activation("selu")
    else:
      actor_hidden_dim = model_cfg['pi_hid_sizes']
      critic_hidden_dim = model_cfg['vf_hid_sizes']
      activation = get_activation(model_cfg['activation'])

    self.adaptation_module = None
    self.ef_encoder = None
    self.encoder = None
    if encoder is not None:
      self.encoder = encoder(model_cfg['encoder_params'])
      self.adaptation_module = AdaptationModule(
        model_cfg['adaptation_module_params'])
    # Policy
      actor_layers = []
      actor_layers.append(
        nn.Linear(self.adaptation_module.state_dim + self.adaptation_module.hidden_dim, actor_hidden_dim[0]))
    else:
      self.ef_encoder = EFEncoder(model_cfg['ef_encoder_params'])
      actor_layers = []
      actor_layers.append(nn.Linear(obs_shape[0] - self.ef_encoder.env_factor_dim +
                          self.ef_encoder.hidden_dim, actor_hidden_dim[0]))
    actor_layers.append(activation)
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
    else:
      if encoder is not None:
        critic_layers.append(self.encoder)
        critic_layers.append(
          nn.Linear(self.encoder.hidden_states_shape, critic_hidden_dim[0]))
      else:
        critic_layers = []
        critic_layers.append(
          nn.Linear(*obs_shape, critic_hidden_dim[0]))
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
    if self.encoder:
      print("visual encoder:")
      print(self.encoder)
    if self.adaptation_module:
      print("adaptation module:")
      print(self.adaptation_module)
    if self.ef_encoder:
      print("env factor encoder:")
      print(self.ef_encoder)

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

  def get_hidden_vector(self, observations_batch, env_factor=None, historical_buffer=None, inner_state_size=0):
    if env_factor is not None:
      if self.ef_encoder is not None:
        return self.ef_encoder(env_factor)
      else:
        raise ValueError("No encoder for env factors")

    if historical_buffer is not None:
      if self.adaptation_module is not None and self.encoder is not None:
        visual_input_batch = observations_batch[:, inner_state_size:]
        # print("visual input batch:", visual_input_batch.shape)
        visual_state_batch = self.encoder(visual_input_batch)
        return self.adaptation_module(historical_buffer, visual_state_batch)
      else:
        raise ValueError("No adaptation module")

  def act(self, observations, states, env_factor=None, historical_buffer=None, inner_state_size=None):
    if self.ef_encoder:
      assert env_factor is not None, "env_factor is None"
      hidden_states = self.ef_encoder(env_factor)
    elif self.adaptation_module:
      assert historical_buffer is not None and self.encoder is not None, "historical_buffer is None"
      visual_states = self.encoder(observations[..., inner_state_size:])
      hidden_states = self.adaptation_module(historical_buffer, visual_states)

    inner_states = observations[..., :inner_state_size]
    actor_input = torch.cat([inner_states, hidden_states], dim=-1)

    actions_mean = self.actor(actor_input)

    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

    actions = distribution.sample()
    actions_log_prob = distribution.log_prob(actions)

    if self.asymmetric:
      # print("states:", states.shape)
      value = self.critic(states)
    else:
      value = self.critic(observations)

    return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

  def act_inference(self, observations, env_factor=None, historical_buffer=None, inner_state_size=None):
    if self.ef_encoder:
      assert env_factor is not None, "env_factor is None"
      hidden_states = self.ef_encoder(env_factor)
    elif self.adaptation_module:
      assert historical_buffer is not None and self.encoder is not None, "historical_buffer is None"
      visual_states = self.encoder(observations[..., inner_state_size:])
      hidden_states = self.adaptation_module(historical_buffer, visual_states)

    inner_states = observations[..., :inner_state_size]
    actor_input = torch.cat([inner_states, hidden_states], dim=-1)
    actions_mean = self.actor(actor_input)
    return actions_mean

  def evaluate(self, observations, states, actions, env_factor=None, historical_buffer=None, inner_state_size=None):
    if self.ef_encoder:
      assert env_factor is not None, "env_factor is None"
      hidden_states = self.ef_encoder(env_factor)
    elif self.adaptation_module:
      assert historical_buffer is not None and self.encoder is not None, "historical_buffer is None"
      visual_states = self.encoder(observations[..., inner_state_size:])
      hidden_states = self.adaptation_module(historical_buffer, visual_states)

    inner_states = observations[..., :inner_state_size]
    actor_input = torch.cat([inner_states, hidden_states], dim=-1)
    actions_mean = self.actor(actor_input)

    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

    actions_log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(observations)

    return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


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
