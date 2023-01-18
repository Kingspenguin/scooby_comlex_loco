import numpy as np


class EnvCurriculumScheduler(object):
  def __init__(self, cfg):
    self.first_randomization = True
    self.last_update_step = self.last_step = 0
    self.curriculum_params = cfg["env_curriculum"]["curriculum_params"]

  def schedule(self, task):
    rand_freq = self.curriculum_params.get("frequency", 1)

    self.last_step = task.gym.get_frame_count(task.sim)

    do_schedule = (
      self.last_step - self.last_update_step) >= rand_freq
    if do_schedule:
      self.last_update_step = self.last_step
