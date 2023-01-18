"""Gait pattern planning module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Sequence

from copy import deepcopy
import math
import os
import inspect

import torch
# from torch import Tensor
import numpy as np

A1_TROTTING = [
    0,
    1,
    1,
    0,
]

_NOMINAL_STANCE_DURATION = 0.2
_NOMINAL_DUTY_FACTOR = 0.6

# this file is copied from python\rlgpu\tasks\controller\gait_generator.py
# for consistency with the RL environment


class GaitGenerator():
  """Generates gaits for quadruped robots.

  A flexible open-loop gait generator. Each leg has its own cycle and duty
  factor. And the state of each leg alternates between stance and swing. One can
  easily formuate a set of common quadruped gaits like trotting, pacing,
  pronking, bounding, etc by tweaking the input parameters.
  """

  def __init__(
      self,
      stance_duration: float = _NOMINAL_STANCE_DURATION,
      duty_factor: float = _NOMINAL_DUTY_FACTOR,
      initial_leg_state: Sequence[int] = A1_TROTTING,
      initial_leg_phase: Sequence[float] = [0.9, 0, 0, 0.9]
  ):
    self._num_legs = 4

    # make them into tensor
    self._stance_duration = np.array([stance_duration] * 4)
    self._duty_factor = np.array([duty_factor] * 4)
    self.full_cycle_period = self._stance_duration / self._duty_factor
    self._swing_duration = self._stance_duration / \
        self._duty_factor - self._stance_duration

    if len(initial_leg_phase) != self._num_legs:
      raise ValueError(
          "The number of leg phases should be the same as number of legs.")

    self._initial_leg_phase = np.array(initial_leg_phase)

    if len(initial_leg_state) != self._num_legs:
      raise ValueError(
          "The number of leg states should be the same of number of legs.")

    self._initial_leg_state = np.array(initial_leg_state)

    self._next_leg_state = np.zeros_like(self._initial_leg_state)

    self._initial_state_ratio_in_cycle = np.zeros_like(self._initial_leg_state)

    self._initial_state_ratio_in_cycle = ((1 - duty_factor) * (
        self._initial_leg_state == 0) + duty_factor * (self._initial_leg_state != 0))

    print((self._initial_leg_state == 0))
    self._next_leg_state = (self._initial_leg_state == 0).astype(np.int32)

    # The normalized phase within swing or stance duration.
    self._normalized_phase = np.zeros((self._num_legs))

    self._leg_state = deepcopy(self._initial_leg_state)
    self._desired_leg_state = deepcopy(self._initial_leg_state)

    self._phase_info = np.zeros((self._num_legs, 2))

    self._time_phase_info = np.zeros((self._num_legs, 2))

  def reset(self):
    # The normalized phase within swing or stance duration.
    self._normalized_phase[:] = 0
    self._leg_state = deepcopy(self._initial_leg_state)
    self._desired_leg_state = deepcopy(self._initial_leg_state)
    # self._phase_info[reset_idx] = 0

  @property
  def desired_leg_state(self):
    """The desired leg SWING/STANCE states.

    Returns:`
      The SWING/STANCE states for all legs.

    """
    return self._desired_leg_state

  @property
  def leg_state(self):
    """The leg state after considering contact with ground.

    Returns:
      The actual state of each leg after accounting for contacts.
    """
    return self._leg_state

  @property
  def swing_duration(self):
    return self._swing_duration

  @property
  def stance_duration(self):
    return self._stance_duration

  @property
  def normalized_phase(self):
    """The phase within the current swing or stance cycle.

    Reflects the leg's phase within the current swing or stance stage. For
    example, at the end of the current swing duration, the phase will
    be set to 1 for all swing legs. Same for stance legs.

    Returns:
      Normalized leg phase for all legs.

    """
    return self._normalized_phase

  def update(self, current_time, residual_phase):
    augmented_time = current_time + self._initial_leg_phase * self.full_cycle_period
    # print(current_time)
    self.phase_in_full_cycle = np.fmod(
        augmented_time, self.full_cycle_period
    ) / self.full_cycle_period

    self.time_phase_in_full_cycle = self.phase_in_full_cycle.copy()

    ratio = self._initial_state_ratio_in_cycle
    self.phase_in_full_cycle += residual_phase
    indices = np.nonzero(
        self.phase_in_full_cycle < ratio
    )[0]
    non_indices = np.nonzero(
        self.phase_in_full_cycle >= ratio
    )[0]

    self._normalized_phase[indices] = (
        self.phase_in_full_cycle / ratio)[indices]
    # print("1", (
    #   self.phase_in_full_cycle / ratio)[indices])
    self._normalized_phase[non_indices] = (
        (self.phase_in_full_cycle - ratio) / (1 - ratio)
    )[non_indices]
    # print("2", self.phase_in_full_cycle)
    self._desired_leg_state[indices] = self._initial_leg_state[indices]
    self._desired_leg_state[non_indices] = self._next_leg_state[non_indices]
    self._leg_state = self._desired_leg_state
    # print(self._normalized_phase)
    self._phase_info[:, 0] = np.cos(self.phase_in_full_cycle * math.pi * 2)
    self._phase_info[:, 1] = np.sin(self.phase_in_full_cycle * math.pi * 2)

    self._time_phase_info[:, 0] = np.cos(
        self.time_phase_in_full_cycle * math.pi * 2)
    self._time_phase_info[:, 1] = np.sin(
        self.time_phase_in_full_cycle * math.pi * 2)

  @ property
  def phase_info(self):
    return self._phase_info

  @property
  def time_phase_info(self):
    return self._time_phase_info
