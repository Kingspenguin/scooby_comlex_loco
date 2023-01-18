import numpy as np
# from rl_pytorch.encoder.voxel import HeightDecoder

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal


class ActorCriticLM(nn.Module):

  def __init__(
      self, encoder,
      obs_shape, states_shape,
      actions_shape, initial_std,
      model_cfg, asymmetric=False,
      with_rec_height=False,
      with_rec_img=False,
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

    self.input_inverse_depth = model_cfg.get("input_inverse_depth", False)
    self.input_original_depth = model_cfg.get("input_original_depth", False)

    self.clip_std = model_cfg.get("clip_std", False)
    if self.clip_std:
      self.clip_std_upper = model_cfg["clip_std_upper"]
      self.clip_std_lower = model_cfg["clip_std_lower"]

    assert encoder is not None
    self.encoder = encoder(model_cfg['encoder_params'])
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

    self.with_rec_height = with_rec_height
    if self.with_rec_height:
      self.height_decoder = self.encoder.height_decoder_class(model_cfg)

    self.with_rec_img = with_rec_img
    if self.with_rec_img:
      self.img_decoder = self.encoder.img_decoder_class(
          model_cfg['encoder_params']
      )

    print("actor:")
    print(self.actor)
    print("critic:")
    print(self.critic)
    if encoder is not None:
      print("encoder:")
      print(self.encoder)
    if self.with_rec_height:
      print("height_decoder")
      print(self.height_decoder)

    if self.with_rec_img:
      print("height_decoder")
      print(self.img_decoder)

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

  def act(self, observations, states):
    out = self.encoder(
        observations)
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
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

    if self.asymmetric:
      value = self.critic(states)
    else:
      value = self.critic(out)

    return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach()

  def act_inference(self, observations):
    out = self.encoder(
        observations
    )
    # if rnn_out.shape[0] == 1:
    #   rnn_out = rnn_out.squeeze(0)
    actions_mean = self.actor(out)
    return actions_mean

  def evaluate(
      self, observations, states, actions,
      return_rnnout=False,
      return_decode_height=False,
      return_decode_img=False
  ):
    original_x_shape = observations.shape
    observations = observations.view(-1, original_x_shape[-1])
    if return_decode_height or return_decode_img:
      out, height_feature = self.encoder(
          observations, return_3d_code=return_decode_height or return_decode_img)
    else:
      out = self.encoder(
          observations, return_3d_code=return_decode_height or return_decode_img)
    actions_mean = self.actor(out)
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
      value = self.critic(out)

    value = value.view(*(original_x_shape[:-1] + (1,)))

    rec_results = {}
    if return_decode_img:
      img_feature = height_feature
      if isinstance(img_feature, tuple):
        img_feature = img_feature[1]
      rec_img = self.img_decoder(img_feature)
      # print(height_feature.shape)
      # print(rec_img.shape)
      rec_img = rec_img.view(*(original_x_shape[:-1] + (
          self.encoder.in_channels * self.encoder.camera_num,
          self.encoder.w, self.encoder.h
      )))
      rec_results["rec_img"] = rec_img
    if return_decode_height:
      height = self.height_decoder(height_feature)
      height = height.view(*(original_x_shape[:-1] + (-1,)))
      rec_results["rec_height"] = height

    if return_decode_height or return_decode_img:
      if return_rnnout:
        return actions_log_prob, entropy.unsqueeze(-1), \
            value, actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1), \
            out, rec_results
      return actions_log_prob, entropy.unsqueeze(-1), \
          value, actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1), \
          rec_results

    if return_rnnout:
      return actions_log_prob, entropy.unsqueeze(-1), value, actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1), out
    return actions_log_prob, entropy.unsqueeze(-1), value, actions_mean, self.log_std.unsqueeze(0).unsqueeze(1).repeat(actions_mean.shape[0], actions_mean.shape[1], 1)

  def process_depth_label(self, depth):
    if self.input_original_depth:
      # depth[depth < 0.09] = 3
      depth = torch.clamp(depth, 0.1, 3)
      return 1 / depth
    if self.input_inverse_depth:
      depth = torch.clamp(depth, 1 / 3, 1 / 0.1)
      return depth
    # convert back to original depth
    original_depth = torch.exp(depth ** 2) - 1
    original_depth = torch.clamp(original_depth, 0.1, 3)
    inverse_depth = 1 / (original_depth + 1e-5)
    return inverse_depth

  def get_original_depth(self, depth):

    if self.input_inverse_depth or self.input_original_depth:
      clipped_depth = torch.clamp(depth, 1 / 3, 1 / 0.1)
      return 1 / clipped_depth
    # convert back to original depth
    clipped_depth = torch.clamp(depth, 1 / 3, 1 / 0.1)
    original_depth = 1 / clipped_depth
    # original_depth = torch.exp(clipped_depth ** 2) - 1
    # inverse_depth = 1 / (original_depth + 1e-5)
    return original_depth

  def compute_rec_losses(
      self, x, labels, with_rec_height, with_rec_img
  ):
    latent_code = self.encoder.compute_latent_code(x)
    losses = {}
    if with_rec_height:
      rec_height = self.height_decoder(latent_code)
      if with_rec_height:
        height_aux_loss = ((
            rec_height - labels["original_height"].detach()
        ) ** 2).mean()
        losses["height_aux_loss"] = height_aux_loss

    if with_rec_img:
      rec_img = self.img_decoder(latent_code)
      # print(rec_img.shape, labels["original_img"].detach().shape)
      b, c, h, w = labels["original_img"].shape
      labels["original_img"] = labels["original_img"].detach().reshape(
          b * c, 1, h, w
      )
      labels["original_img"] = self.process_depth_label(labels["original_img"])

      img_aux_loss = torch.abs(
          rec_img - labels["original_img"]
      ).mean()
      losses["img_aux_loss"] = img_aux_loss
    return losses

  def get_decoded_img(
      self, x
  ):
    # Assume Batch Size = 1
    latent_code = self.encoder.compute_latent_code(x)
    latent_code = latent_code[0:1, ...]
    rec_img = self.img_decoder(latent_code)
    rec_img = self.get_original_depth(rec_img)
    return rec_img


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
