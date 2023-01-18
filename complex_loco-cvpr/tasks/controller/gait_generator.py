"""Gait pattern planning module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Sequence
from torch import Tensor
from copy import deepcopy
import torch
import math
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


A1_TROTTING = [
    0,
    1,
    1,
    0,
]

_NOMINAL_STANCE_DURATION = 0.2
_NOMINAL_DUTY_FACTOR = 0.6


class GaitGenerator():
  """Generates gaits for quadruped robots.

  A flexible open-loop gait generator. Each leg has its own cycle and duty
  factor. And the state of each leg alternates between stance and swing. One can
  easily formuate a set of common quadruped gaits like trotting, pacing,
  pronking, bounding, etc by tweaking the input parameters.
  """

  def __init__(
      self,
      num_envs: int,
      device: Any,
      stance_duration: float = _NOMINAL_STANCE_DURATION,
      duty_factor: float = _NOMINAL_DUTY_FACTOR,
      initial_leg_state: Sequence[int] = A1_TROTTING,
      initial_leg_phase: Sequence[float] = [0.9, 0, 0, 0.9]
  ):
    self._num_legs = 4
    self._num_envs = num_envs
    self._device = device

    # make them into tensor, repeat for each env
    self._stance_duration = torch.as_tensor(
        [stance_duration], device=self._device).repeat(self._num_envs, 4)
    self._duty_factor = torch.as_tensor(
        [duty_factor], device=self._device).repeat(self._num_envs, 4)
    self._swing_duration = self._stance_duration / \
        self._duty_factor - self._stance_duration
    # print(self._swing_duration, self._stance_duration)
    # exit()
    if len(initial_leg_phase) != self._num_legs:
      raise ValueError(
          "The number of leg phases should be the same as number of legs.")
    self._initial_leg_phase = torch.as_tensor(
        [initial_leg_phase], device=self._device).repeat(self._num_envs, 1)
    if len(initial_leg_state) != self._num_legs:
      raise ValueError(
          "The number of leg states should be the same of number of legs.")
    self._initial_leg_state = torch.as_tensor(
        [initial_leg_state], device=self._device).repeat(self._num_envs, 1)
    self._next_leg_state = torch.zeros_like(
        self._initial_leg_state, device=self._device)
    self._initial_state_ratio_in_cycle = torch.zeros_like(
        self._initial_leg_state, device=self._device)

    self._initial_state_ratio_in_cycle = ((1 - duty_factor) * (
        self._initial_leg_state == 0) + duty_factor * (self._initial_leg_state != 0)).float()
    self._next_leg_state = (self._initial_leg_state == 0).long()

    # The normalized phase within swing or stance duration.
    self._normalized_phase = torch.zeros(
        (num_envs, self._num_legs), device=self._device)
    self._leg_state = deepcopy(self._initial_leg_state)
    self._desired_leg_state = deepcopy(self._initial_leg_state)
    self._phase_info = torch.zeros(
        (num_envs, self._num_legs, 2), device=self._device)

    self._time_phase_info = torch.zeros(
        (num_envs, self._num_legs, 2), device=self._device)

  def reset(self, reset_idx):
    # The normalized phase within swing or stance duration.
    self._normalized_phase[reset_idx] = 0
    self._leg_state[reset_idx] = deepcopy(self._initial_leg_state)[reset_idx]
    self._desired_leg_state[reset_idx] = deepcopy(
        self._initial_leg_state)[reset_idx]
    # self._phase_info[reset_idx] = 0

  @property
  def desired_leg_state(self) -> Tensor:
    """The desired leg SWING/STANCE states.

    Returns:
      The SWING/STANCE states for all legs.

    """
    return self._desired_leg_state

  @property
  def leg_state(self) -> Tensor:
    """The leg state after considering contact with ground.

    Returns:
      The actual state of each leg after accounting for contacts.
    """
    return self._leg_state

  @property
  def swing_duration(self) -> Tensor:
    return self._swing_duration

  @property
  def stance_duration(self) -> Tensor:
    return self._stance_duration

  @property
  def normalized_phase(self) -> Tensor:
    """The phase within the current swing or stance cycle.

    Reflects the leg's phase within the current swing or stance stage. For
    example, at the end of the current swing duration, the phase will
    be set to 1 for all swing legs. Same for stance legs.

    Returns:
      Normalized leg phase for all legs.

    """
    return self._normalized_phase

  def update(self, current_time: Tensor, residual_phase: Tensor):
    full_cycle_period = self._stance_duration / self._duty_factor
    augmented_time = current_time.unsqueeze(-1) + \
        self._initial_leg_phase * full_cycle_period
    # print("aug time:", augmented_time)
    augmented_time = augmented_time
    # print(current_time)
    self.phase_in_full_cycle = torch.fmod(augmented_time,
                                          full_cycle_period) / full_cycle_period
    # print("phase in full cycle:", self.phase_in_full_cycle)
    ratio = self._initial_state_ratio_in_cycle
    # print("Ratio:", ratio)
    self.time_phase_in_full_cycle = self.phase_in_full_cycle.clone()
    self.phase_in_full_cycle += residual_phase
    # print("phase in full cycle:", self.phase_in_full_cycle)
    indices = torch.nonzero(self.phase_in_full_cycle < ratio, as_tuple=True)[0]
    non_indices = torch.nonzero(
        self.phase_in_full_cycle >= ratio, as_tuple=True)[0]
    # print("indices:", indices, non_indices)
    self._normalized_phase[indices] = (
        self.phase_in_full_cycle / ratio)[indices]
    # print("1", (
    #   self.phase_in_full_cycle / ratio)[indices])
    self._normalized_phase[non_indices] = ((
        self.phase_in_full_cycle - ratio) / (1 - ratio))[non_indices]
    # print("2", self.phase_in_full_cycle)
    # print("normed phase:", self._normalized_phase)
    self._desired_leg_state[indices] = self._initial_leg_state[indices]
    self._desired_leg_state[non_indices] = self._next_leg_state[non_indices]
    self._leg_state = self._desired_leg_state
    # print("Desired leg state", self._desired_leg_state)
    # print(self._normalized_phase)
    self._phase_info[:, :, 0] = torch.cos(
        self.phase_in_full_cycle * math.pi * 2)
    self._phase_info[:, :, 1] = torch.sin(
        self.phase_in_full_cycle * math.pi * 2)

    self._time_phase_info[:, :, 0] = torch.cos(
        self.time_phase_in_full_cycle * math.pi * 2)
    self._time_phase_info[:, :, 1] = torch.sin(
        self.time_phase_in_full_cycle * math.pi * 2)

  @property
  def phase_info(self):
    return self._phase_info

  @property
  def time_phase_info(self):
    return self._time_phase_info
