# Reference for RMA: https://arxiv.org/pdf/2107.04034.pdf
import torch
import torch.nn as nn


class EFEncoder(nn.Module):
  # Environment factored encoder in RMA paper
  # But we use the height map input version
  def __init__(self, cfg):
    super(EFEncoder, self).__init__()
    self.cfg = cfg
    self.env_factor_dim = cfg["privilege_dim"] + cfg["heights_dim"] # privilege dimension
    self.hidden_dim = cfg["hidden_dim"]  # hidden dimension
    fc1 = nn.Linear(self.env_factor_dim, self.hidden_dim)
    fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.fc = nn.Sequential(fc1, nn.ELU(), fc2, nn.ELU(), fc3)

  def forward(self, x):
    return self.fc(x)


class AdaptationModule(nn.Module):
  # Adaptation module in RMA paper
  # But we use the visual input version
  def __init__(self, cfg):
    super(AdaptationModule, self).__init__()
    self.cfg = cfg
    self.state_dim = cfg["state_dim"]  # state dimension
    self.historical_steps = cfg["historical_steps"]  # historical steps
    self.hidden_dim = cfg["hidden_dim"]  # hidden dimension
    fc1 = nn.Linear(
      self.state_dim, self.hidden_dim)
    fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.fc = nn.Sequential(fc1, nn.ELU(), fc2)
    conv_layers = [
      nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=4, stride=4),
      nn.ELU(),
      nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=3),
      nn.Flatten()
    ]
    self.conv = nn.Sequential(*conv_layers)
    self.final_layers = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ELU(),
                                      nn.Linear(self.hidden_dim, self.hidden_dim))
    
  def forward(self, x, visual_state):
    x = self.fc(x)
    x = x.view(x.shape[0], self.hidden_dim, self.historical_steps)
    x = self.conv(x)
    x = torch.cat([x, visual_state], dim=-1)
    x = self.final_layers(x)
    return x
    