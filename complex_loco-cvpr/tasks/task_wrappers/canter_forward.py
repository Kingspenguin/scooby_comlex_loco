from tasks.task_wrappers.base_task_wrapper import BaseTaskWrapper
import torch


class CanterForwardTask(BaseTaskWrapper):

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
    if task.multi_env:
      termination_list = []
      for env_wrapper in task.env_wrapper_list:
        termination_list.append(
          env_wrapper.check_termination(task).unsqueeze(0)
        )
      termination = torch.cat(termination_list, dim=0)
      termination = torch.gather(
        termination, 0,
        task.chosen_env
        # torch.repeat_interleave(
        #     self.chosen_env.unsqueeze(2), 3, 2
        # )
      ).squeeze(0)
      task.reset_buf |= termination
    else:
      task.reset_buf |= task.env_wrapper.check_termination(task)
