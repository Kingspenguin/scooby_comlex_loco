import torch
from torch import autograd
from rl_pytorch.networks import get_network
import logging
import numpy as np
import json
import os

from torch.optim.adam import Adam


class GAIL:
  def __init__(self, num_envs, device, learner_cfg, control_dt):

    self.discriminator = get_network("discriminator", learner_cfg)
    self.discriminator.to(device)
    self.num_envs = num_envs
    self.nsteps = learner_cfg["nsteps"]
    self.nminibatches = learner_cfg["nminibatches"]
    self.device = device
    self.optimizer = Adam(
        self.discriminator.parameters(), lr=learner_cfg["lr"])
    self.max_grad_norm = learner_cfg["max_grad_norm"]
    self.noptepochs = learner_cfg["noptepochs"]
    self.dof = learner_cfg["dof"]
    self.transtion_buffer = torch.zeros(
        (self.nsteps, self.num_envs, 2 * self.dof)).to(device)  # 2 * dof means we have dof pos + dof vel, for current and next pose
    self.step = 0
    self.motion_names = learner_cfg["motion_names"]
    self.control_dt = control_dt
    self.load_ref_motions()

  def load_ref_motions(self):
    self._frames = []
    for motion_names in self.motion_names:
      motion_file_names = os.path.join(os.path.dirname(os.path.abspath(
          __file__)), '../../data/a1_motions', motion_names + ".txt")
      logging.info("Loading motion from: {:s}".format(motion_names))
      with open(motion_file_names, "r") as f:
        motion_json = json.load(f)
        frames = torch.as_tensor(
            motion_json["Frames"], dtype=torch.float32, device=self.device
        )[..., -self.dof:]
        frame_dt = motion_json["FrameDuration"]

        l = frames.shape[0]
        dt = 0
        filtered_frames = []

        while dt < (l - 1) * frame_dt:
          idx = int(dt / frame_dt)
          filtered_frames.append(
              frames[idx: idx + 1]
          )
          dt += self.control_dt
        frames = torch.cat(filtered_frames, dim=0)

        frames = torch.cat([
            frames[..., 3: 6],
            frames[..., 0: 3],
            frames[..., 9: 12],
            frames[..., 6: 9],
        ], dim=1)
        print(l, frames.shape)

        current_frames = frames[:-1, :]
        next_frames = frames[1:, :]
        frames = torch.cat([
            current_frames, next_frames
        ], dim=-1).view(-1, self.dof * 2)

        self._frames.append(frames)
        logging.info("Loaded motion from {:s}.".format(motion_names))
    self._frames = torch.cat(self._frames, dim=0)

  def sample_expert(self):
    expert_indices = torch.randint(
        0, len(self._frames), size=(self.nsteps * self.num_envs, )).to(self.device)
    expert_frames = self._frames[expert_indices].view(
        self.nsteps, self.num_envs, -1)
    return expert_frames

  # def sample_transitions(self, nminibatches):
  def sample_transitions(self):
    """
    Sample transitions.
    """
    # indices = np.random.randint(
    #     0, self.nsteps, size=nminibatches)
    indices = np.arange(self.nsteps)
    np.random.shuffle(indices)
    return self.transtion_buffer[indices]

  def save_transition(self, current_pose, next_pose):
    """
    Save transition.
    """
    assert self.step < self.nsteps, "GAIL transition buffer overflow!"
    # print("store", self.step)
    transition = torch.cat((current_pose, next_pose), dim=-1)
    self.transtion_buffer[self.step].copy_(transition)
    self.step += 1

  def clear(self):
    self.step = 0

  def check_update(self):
    """
    Check if we need to update the discriminator.
    """
    return self.step >= self.nsteps

  def calc_grad2(self, d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.mean(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.sum() / batch_size
    return reg

  def update(self):
    """
    Update discriminator.
    """
    self.discriminator.train()
    total_loss = 0
    for _ in range(self.noptepochs):
      expert_frames = torch.chunk(
          self.sample_expert(), self.nminibatches, 0
      )
      agent_frames = torch.chunk(
          self.sample_transitions(), self.nminibatches, 0
      )

      for expert_frame, agent_frame in zip(expert_frames, agent_frames):
        assert expert_frame.shape[-1] == agent_frame.shape[-1]
        expert_frame = expert_frame.view(
            self.nsteps * self.num_envs // self.nminibatches, -1)
        agent_frame = agent_frame.view(
            self.nsteps * self.num_envs // self.nminibatches, -1)

        expert_frame.requires_grad_()

        self.optimizer.zero_grad()
        expert_logits = self.discriminator(expert_frame)
        expert_loss = torch.sum(
            (expert_logits - 1) ** 2, dim=-1).mean()
        agent_logits = self.discriminator(agent_frame)
        agent_loss = torch.sum((agent_logits + 1) ** 2, dim=-1).mean()
        loss = 0.5 * (expert_loss + agent_loss)

        l_reg = self.calc_grad2(d_out=expert_logits, x_in=expert_frame)
        l_reg = 10 * l_reg
        l_reg.backward(retain_graph=True)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                       self.max_grad_norm)
        self.optimizer.step()
        total_loss += loss.item()
    self.clear()
    mean_loss = total_loss / self.noptepochs / self.nminibatches
    print("#" * 25, "GAIL update", "#" * 25)
    print(" " * 20, "Mean loss: {:.4f}".format(mean_loss))
    print()
    return mean_loss

  def reward(self, current_pose: torch.Tensor, next_pose: torch.Tensor):
    """
    Compute reward.
    """
    with torch.no_grad():
      transition = torch.cat((current_pose, next_pose), dim=-1)
      pred_logits = self.discriminator(transition).squeeze(-1)
      pred_scores = 1 - 0.25 * (pred_logits - 1) ** 2
      reward = torch.clamp(pred_scores, 0.0, 1.0)
      return reward

  def reward(self):
    """
    Compute reward.
    """
    with torch.no_grad():
      transition = self.transtion_buffer
      pred_logits = self.discriminator(transition)  # .squeeze(-1)
      pred_scores = 1 - 0.25 * (pred_logits - 1) ** 2
      reward = torch.clamp(pred_scores, 0.0, 1.0)
      return reward
