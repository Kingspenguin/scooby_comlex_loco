# Task specific functions that determine:
# - Combination of rewards
# - Terminal conditions
# - Reference data
# ...
import os
import yaml
import torch
import json


class BaseTaskWrapper(object):

  def __init__(self, device, cfg):
    """Initializes the task wrappers."""
    self.device = device
    self.cfg = cfg
    self.task_name = cfg["task"]["name"]
    self.file_name = self.task_name + '.yaml'
    self.testing = cfg.get("testing", False)
    self.test_finished = False
    if self.testing:
      self.test_episodes = cfg.get("test_episodes", 10)
      self.statistics_logdir = cfg.get("statistics_logdir", "statistics")
      self.test_id = cfg.get("test_id", "all")
      self.reachable_distance_ratio = torch.zeros(
          (cfg["env"]["numEnvs"], self.test_episodes), device=device)
      self.traverse_success_ratio = torch.zeros(
          (cfg["env"]["numEnvs"], self.test_episodes), device=device)
      self.current_test_episode = torch.zeros(
          (cfg["env"]["numEnvs"],), device=device, dtype=torch.long)
      self.finished_test_episodes = torch.zeros(
          (cfg["env"]["numEnvs"],), device=device, dtype=torch.long)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', self.file_name), 'r') as f:
      self.task_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return

  def check_termination(self, task):
    """Checks if the episode is over."""
    reset_buf = torch.zeros_like(task.reset_buf)
    # falling = task.root_states[:, 2] < -4.0
    falling = task.root_states[:, 2] < task.robot_env_low
    reset_buf |= falling
    if self.testing:
      success_buf = (
          task.robot_reachable_distance - (task.root_states[:, 0] - task.robot_origin[:, 0]) < 0.5)
      reset_buf |= success_buf
    return reset_buf

  def get_statistics(self, task):
    """Gets the statistics of the current episode."""
    if self.test_finished or torch.all(self.current_test_episode >= self.test_episodes):
      return
    reset_buf = torch.logical_and(
        task.reset_buf, (self.finished_test_episodes == 0))
    finished_robot_x_pos = (task.root_states[:, 0] - task.robot_origin[:, 0])
    finished_robot_x_ratio = torch.clamp(
        finished_robot_x_pos / (task.robot_reachable_distance - 0.5), 0, 1)
    finished_robot_x_success = (
        task.robot_reachable_distance - finished_robot_x_pos < 0.5).float()
    self.traverse_success_ratio[reset_buf, self.current_test_episode[reset_buf]
                                ] = finished_robot_x_success[reset_buf]
    self.reachable_distance_ratio[reset_buf, self.current_test_episode[reset_buf]
                                  ] = finished_robot_x_ratio[reset_buf]
    self.current_test_episode[reset_buf] += 1
    self.finished_test_episodes = (
        self.current_test_episode >= self.test_episodes).float()

    if torch.all(self.current_test_episode >= self.test_episodes):
      print("Finished testing")
      print("Average reachable distance: {:2f},  std: {:2f}".format(
          self.reachable_distance_ratio.mean(), self.reachable_distance_ratio.mean(0).std()))
      print("Average traverse success: {:2f},  std: {:2f}".format(
          self.traverse_success_ratio.mean(), self.traverse_success_ratio.mean(0).std()))
      # Save the results
      if not os.path.exists(self.statistics_logdir):
        os.makedirs(self.statistics_logdir, exist_ok=True)
      with open(os.path.join(self.statistics_logdir, self.test_id + '.json'), 'w') as f:
        json.dump({
            "reachable_distance_ratio": [self.reachable_distance_ratio.mean().item(), self.reachable_distance_ratio.mean(0).std().item()],
            "traverse_success_ratio": [self.traverse_success_ratio.mean().item(), self.traverse_success_ratio.mean(0).std().item()],
        }, f)
      self.test_finished = True
