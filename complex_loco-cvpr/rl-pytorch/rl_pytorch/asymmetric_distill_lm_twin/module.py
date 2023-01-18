import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal


class ActorCriticLMTwin(nn.Module):
  def __init__(
      self, encoder,
      obs_shape, states_shape, cpg_phase_info_shape,
      pd_actions_shape, cpg_actions_shape,
      initial_std, model_cfg, asymmetric=False
  ):
    super().__init__()

    self.asymmetric = asymmetric

    if model_cfg is None:
      actor_hidden_dim = [256, 256, 256]
      critic_hidden_dim = [256, 256, 256]
      activation = get_activation("selu")
    else:
      actor_enc_hidden_dim = model_cfg['pi_enc_hid_sizes']
      actor_pd_hidden_dim = model_cfg['pi_pd_hid_sizes']
      actor_cpg_hidden_dim = model_cfg['pi_cpg_hid_sizes']
      critic_hidden_dim = model_cfg['vf_hid_sizes']
      activation = get_activation(model_cfg['activation'])

    self.pd_detach = model_cfg.get("pd_detach", False)
    self.pd_tanh = model_cfg.get("pd_tanh", False)

    self.clip_std = model_cfg.get("clip_std", False)
    if self.clip_std:
      self.clip_std_upper = model_cfg["clip_std_upper"]
      self.clip_std_lower = model_cfg["clip_std_lower"]

    assert encoder is not None
    self.encoder = encoder(model_cfg['encoder_params'])
  # Policy
    actor_enc_layers = []
    actor_enc_layers.append(
        nn.Linear(self.encoder.hidden_states_shape, actor_enc_hidden_dim[0]))
    actor_enc_layers.append(activation)

    assert actor_pd_hidden_dim[0] == actor_cpg_hidden_dim[0]
    for l in range(len(actor_enc_hidden_dim)):
      if l == len(actor_enc_hidden_dim) - 1:
        actor_enc_layers.append(
            nn.Linear(actor_enc_hidden_dim[l], actor_pd_hidden_dim[0]))
      else:
        actor_enc_layers.append(
            nn.Linear(actor_enc_hidden_dim[l], actor_enc_hidden_dim[l + 1]))
      actor_enc_layers.append(activation)
    self.actor_enc = nn.Sequential(*actor_enc_layers)

    actor_cpg_hidden_dim[0] = actor_cpg_hidden_dim[0] + cpg_phase_info_shape
    actor_pd_layers = []
    for l in range(len(actor_pd_hidden_dim)):
      if l == len(actor_pd_hidden_dim) - 1:
        actor_pd_layers.append(
            nn.Linear(actor_pd_hidden_dim[l], *pd_actions_shape))
      else:
        actor_pd_layers.append(
            nn.Linear(actor_pd_hidden_dim[l], actor_pd_hidden_dim[l + 1]))
        actor_pd_layers.append(activation)
    self.actor_pd = nn.Sequential(*actor_pd_layers)

    actor_cpg_layers = []
    for l in range(len(actor_cpg_hidden_dim)):
      if l == len(actor_cpg_hidden_dim) - 1:
        actor_cpg_layers.append(
            nn.Linear(actor_cpg_hidden_dim[l], *cpg_actions_shape))
      else:
        actor_cpg_layers.append(
            nn.Linear(actor_cpg_hidden_dim[l], actor_cpg_hidden_dim[l + 1]))
        actor_cpg_layers.append(activation)
    self.actor_cpg = nn.Sequential(*actor_cpg_layers)

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
    print("actor enc:")
    print(self.actor_enc)
    print("actor pd:")
    print(self.actor_pd)
    print("actor cpg:")
    print(self.actor_cpg)
    print("critic:")
    print(self.critic)
    if encoder is not None:
      print("encoder:")
      print(self.encoder)

    # Action noise
    self.pd_log_std = nn.Parameter(
        np.log(initial_std) * torch.ones(*pd_actions_shape))

    self.cpg_log_std = nn.Parameter(
        np.log(initial_std) * torch.ones(*cpg_actions_shape))

    # Initialize the weights like in stable baselines
    actor_enc_weights = [np.sqrt(2)] * len(actor_enc_hidden_dim)
    actor_enc_weights.append(0.01)
    self.init_weights(self.actor_enc, actor_enc_weights)

    actor_pd_weights = [np.sqrt(2)] * len(actor_pd_hidden_dim)
    actor_pd_weights.append(0.01)
    self.init_weights(self.actor_pd, actor_pd_weights)

    actor_cpg_weights = [np.sqrt(2)] * len(actor_cpg_hidden_dim)
    actor_cpg_weights.append(0.01)
    self.init_weights(self.actor_cpg, actor_cpg_weights)

    critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
    critic_weights.append(1.0)
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

  def act(self, observations, states, phase_infos):
    out = self.encoder(
        observations
    )
    out = self.actor_enc(out)
    if self.pd_detach:
      pd_actions_mean = self.actor_pd(out.detach())
    else:
      pd_actions_mean = self.actor_pd(out)
    if self.pd_tanh:
      pd_actions_mean = torch.tanh(pd_actions_mean)
    cpg_actions_mean = self.actor_cpg(
        torch.cat([out, phase_infos], dim=-1)
    )

    if self.clip_std:
      self.pd_log_std.copy_(
          torch.clamp(self.pd_log_std, self.clip_std_lower,
                      self.clip_std_upper)
      )
      self.cpg_log_std.copy_(
          torch.clamp(self.cpg_log_std, self.clip_std_lower,
                      self.clip_std_upper)
      )

    # print(self.log_std)
    pd_covariance = torch.diag(self.pd_log_std.exp() * self.pd_log_std.exp())
    # distribution = Normal(actions_mean, self.log_std.exp())
    pd_distribution = MultivariateNormal(
        pd_actions_mean, scale_tril=pd_covariance)
    pd_actions = pd_distribution.sample()
    pd_actions_log_prob = pd_distribution.log_prob(pd_actions)

    cpg_covariance = torch.diag(
        self.cpg_log_std.exp() * self.cpg_log_std.exp())
    # distribution = Normal(actions_mean, self.log_std.exp())
    cpg_distribution = MultivariateNormal(
        cpg_actions_mean, scale_tril=cpg_covariance)
    cpg_actions = cpg_distribution.sample()
    cpg_actions_log_prob = cpg_distribution.log_prob(cpg_actions)

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(observations)

    return pd_actions.detach(), pd_actions_log_prob.detach(), \
        cpg_actions.detach(), cpg_actions_log_prob.detach(), \
        value.detach(), \
        pd_actions_mean.detach(), self.pd_log_std.repeat(pd_actions_mean.shape[0], 1).detach(), \
        cpg_actions_mean.detach(), self.cpg_log_std.repeat(
        cpg_actions_mean.shape[0], 1).detach()

  def act_inference(self, observations):
    out = self.encoder(
        observations
    )
    out = self.actor_enc(out)
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
    if self.pd_detach:
      actions_mean = self.actor_pd(out.detach())
    else:
      actions_mean = self.actor_pd(out)
    if self.pd_tanh:
      actions_mean = torch.tanh(actions_mean)
    return actions_mean

  def evaluate(
      self, observations, states, phase_infos, pd_actions, cpg_actions, return_rnnout=False
  ):
    out = self.encoder(observations)
    out = self.actor_enc(out)

    if out.shape[0] == 1:
      out = out.squeeze(0)

    if self.pd_detach:
      pd_actions_mean = self.actor_pd(out.detach())
    else:
      pd_actions_mean = self.actor_pd(out)
    if self.pd_tanh:
      pd_actions_mean = torch.tanh(pd_actions_mean)

    cpg_actions_mean = self.actor_cpg(
        torch.cat([out, phase_infos], dim=-1)
    )

    if self.clip_std:
      self.pd_log_std.copy_(
          torch.clamp(self.pd_log_std, self.clip_std_lower,
                      self.clip_std_upper)
      )
      self.cpg_log_std.copy_(
          torch.clamp(self.cpg_log_std, self.clip_std_lower,
                      self.clip_std_upper)
      )

    # print(self.log_std)
    pd_covariance = torch.diag(self.pd_log_std.exp() * self.pd_log_std.exp())
    pd_distribution = MultivariateNormal(
        pd_actions_mean, scale_tril=pd_covariance)
    pd_actions_log_prob = pd_distribution.log_prob(pd_actions)

    cpg_covariance = torch.diag(
        self.cpg_log_std.exp() * self.cpg_log_std.exp())
    # distribution = Normal(actions_mean, self.log_std.exp())
    cpg_distribution = MultivariateNormal(
        cpg_actions_mean, scale_tril=cpg_covariance)
    cpg_actions_log_prob = cpg_distribution.log_prob(cpg_actions)

    pd_entropy = pd_distribution.entropy()
    cpg_entropy = cpg_distribution.entropy()

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(observations)

    if return_rnnout:
      return pd_actions_log_prob, pd_entropy.unsqueeze(-1), pd_actions_mean, self.pd_log_std.unsqueeze(0).unsqueeze(1).repeat(pd_actions_mean.shape[0], pd_actions_mean.shape[1], 1),\
          cpg_actions_log_prob, cpg_entropy.unsqueeze(-1), cpg_actions_mean, self.cpg_log_std.unsqueeze(0).unsqueeze(1).repeat(cpg_actions_mean.shape[0], cpg_actions_mean.shape[1], 1), \
          value, out
    return pd_actions_log_prob, pd_entropy.unsqueeze(-1), pd_actions_mean, self.pd_log_std.unsqueeze(0).unsqueeze(1).repeat(pd_actions_mean.shape[0], pd_actions_mean.shape[1], 1),\
        cpg_actions_log_prob, cpg_entropy.unsqueeze(-1), cpg_actions_mean, self.cpg_log_std.unsqueeze(
        0).unsqueeze(1).repeat(cpg_actions_mean.shape[0], cpg_actions_mean.shape[1], 1), value


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
