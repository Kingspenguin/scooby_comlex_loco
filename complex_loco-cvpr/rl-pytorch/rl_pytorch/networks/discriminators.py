import torch


class Discriminator(torch.nn.Module):
  def __init__(self, cfg):
    super(Discriminator, self).__init__()
    self.cfg = cfg
    self.state_dim = cfg["state_dim"]  # state dimension
    self.hidden_dim = cfg["hidden_dim"]  # hidden dimension

    self.fc1 = torch.nn.Linear(
      self.state_dim * 2, self.hidden_dim)
    self.fc2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
    self.fc3 = torch.nn.Linear(self.hidden_dim, 1)

  def forward(self, x):
    # x = x.view(-1, self.state_dim + self.state_dim)
    x = torch.nn.functional.relu(self.fc1(x))
    x = torch.nn.functional.relu(self.fc2(x))
    x = self.fc3(x)
    return x
