import numpy as np


class RewardRandomizer(object):
  def __init__(self, cfg):
    self.first_randomization = True
    self.last_rand_step = self.last_step = 0
    self.randomization_params = cfg["randomize_reward"]["randomization_params"]
    return

  def apply_randomizations(self, task):
    rand_freq = self.randomization_params.get("frequency", 1)

    self.last_step = task.gym.get_frame_count(task.sim)

    do_rew_randomize = (
      self.last_step - self.last_rand_step) >= rand_freq
    if do_rew_randomize:
      self.last_rand_step = self.last_step

    scale_params = self.randomization_params["reward_scale"]
    for k, v in scale_params.items():
      v_range = v["range"]
      task.reward_scales[k] = np.random.uniform(
        low=v_range[0], high=v_range[1]) * task.dt
