import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal


class ActorCritic(nn.Module):

  def __init__(self, encoder, obs_shape, states_shape, actions_shape, initial_std, model_cfg, asymmetric=False):
    super(ActorCritic, self).__init__()

    self.asymmetric = asymmetric
    self.add_ln = model_cfg.get("add_ln", False)

    if model_cfg is None:
      actor_hidden_dim = [256, 256, 256]
      critic_hidden_dim = [256, 256, 256]
      activation = get_activation("selu")
    else:
      actor_hidden_dim = model_cfg['pi_hid_sizes']
      critic_hidden_dim = model_cfg['vf_hid_sizes']
      activation = get_activation(model_cfg['activation'])

    self.use_encoder = encoder is not None
    if encoder is not None:
      self.encoder = encoder(model_cfg['encoder_params'])
    # Policy
      actor_layers = []
      actor_layers.append(
          nn.Linear(self.encoder.hidden_states_shape, actor_hidden_dim[0]))
    else:
      actor_layers = []
      actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))

    actor_layers.append(activation)
    if self.add_ln:
      actor_layers.append(nn.LayerNorm(actor_hidden_dim[0]))
    for l in range(len(actor_hidden_dim)):
      if l == len(actor_hidden_dim) - 1:
        actor_layers.append(
            nn.Linear(actor_hidden_dim[l], *actions_shape))
      else:
        actor_layers.append(
            nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
        actor_layers.append(activation)
        if self.add_ln:
          actor_layers.append(nn.LayerNorm(actor_hidden_dim[l + 1]))

    self.actor = nn.Sequential(*actor_layers)

    # Value function
    critic_layers = []
    if self.asymmetric:
      critic_layers.append(
          nn.Linear(*states_shape, critic_hidden_dim[0]))
    else:
      if encoder is not None:
        # critic_layers.append(self.encoder)
        critic_layers.append(
            nn.Linear(self.encoder.hidden_states_shape, critic_hidden_dim[0]))
      else:
        # critic_layers = []
        critic_layers.append(
            nn.Linear(*obs_shape, critic_hidden_dim[0]))
    critic_layers.append(activation)
    if self.add_ln:
      critic_layers.append(nn.LayerNorm(critic_hidden_dim[0]))
    for l in range(len(critic_hidden_dim)):
      if l == len(critic_hidden_dim) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
      else:
        critic_layers.append(
            nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
        critic_layers.append(activation)
        if self.add_ln:
          actor_layers.append(nn.LayerNorm(critic_hidden_dim[l + 1]))
    self.critic = nn.Sequential(*critic_layers)
    print("encoder:")
    print(self.encoder)
    print("actor:")
    print(self.actor)
    print("critic:")
    print(self.critic)

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
      self.apply(self.init_linear_weights)
    self.init_weights(self.actor, actor_weights)
    self.init_weights(self.critic, critic_weights)

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

  @staticmethod
  def init_linear_weights(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
      # print(m)

  def forward(self):
    raise NotImplementedError

  def act(self, observations, states, return_eval_log_prob=False):
    if self.use_encoder:
      out = self.encoder(observations)
    else:
      out = observations
    actions_mean = self.actor(out)

    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

    # std = self.log_std.exp()
    # distribution = Normal(actions_mean, std)

    actions = distribution.sample()
    actions_log_prob = distribution.log_prob(actions)
    eval_log_prob = distribution.log_prob(actions_mean)
    # actions_log_prob = distribution.log_prob(actions).sum(-1, keepdim=True)
    # print(actions_log_prob.shape)

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(out)

    if return_eval_log_prob:
      return actions.detach(), actions_log_prob.detach(), \
          value.detach(), actions_mean.detach(), \
          self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
          eval_log_prob.detach()
    else:
      return actions.detach(), actions_log_prob.detach(), \
          value.detach(), actions_mean.detach(), \
          self.log_std.repeat(actions_mean.shape[0], 1).detach()

  def act_inference(self, observations):
    if self.use_encoder:
      out = self.encoder(observations)
    else:
      out = observations
    actions_mean = self.actor(out)
    return actions_mean

  def act_inference_with_encoder(self, observations):
    if self.use_encoder:
      out = self.encoder(observations)
      actions_mean = self.actor(out)
    else:
      out = None
      actions_mean = self.actor(observations)
    return actions_mean, out

  def evaluate(self, observations, states, actions):
    if self.use_encoder:
      out = self.encoder(observations)
    else:
      out = observations
    actions_mean = self.actor(out)

    covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
    distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

    # std = self.log_std.exp()
    # distribution = Normal(actions_mean, std)

    actions_log_prob = distribution.log_prob(actions)
    # actions_log_prob = distribution.log_prob(actions).sum(-1, keepdim=True)
    entropy = distribution.entropy()
    # entropy = distribution.entropy().sum(-1, keepdim=True)

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(out)

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
