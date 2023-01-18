from tasks.task_wrappers.base_task_wrapper import BaseTaskWrapper
import torch


class PaceForwardTask(BaseTaskWrapper):

  def __init__(self, device, cfg):
    """Initializes the task."""
    super().__init__(device, cfg)

  def check_termination(self, task):
    """ Check if environments need to be reset
            """
    task.reset_buf = torch.any(torch.norm(
      task.contact_forces[:, task.termination_contact_indices, :], dim=-1) > 1., dim=1)
    task.time_out_buf = task.progress_buf > task.max_episode_length
    super_check = super().check_termination(task)
    task.reset_buf |= super_check
    task.reset_buf |= task.time_out_buf
    task.reset_buf |= task.env_wrapper.check_termination(task)
