"""State estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cgitb import reset
from copy import deepcopy
from typing import Any

from utils.torch_jit_utils import quat_rotate_inverse, quat_apply
import torch
from torch import Tensor

_DEFAULT_WINDOW_SIZE = 20


class MovingWindowFilter(object):
  """A stable O(1) moving filter for incoming data streams.

  We implement the Neumaier's algorithm to calculate the moving window average,
  which is numerically stable.

  """

  def __init__(self, num_envs: int, device: Any, window_size: int):
    """Initializes the class.

    Args:
      window_size: The moving window size.
    """
    assert window_size > 0
    self._num_envs = num_envs
    self._window_size = window_size
    self._device = device
    self._current_size = torch.zeros(
      (self._num_envs), device=self._device, dtype=torch.long)
    self._value_deque = torch.zeros(
      (self._num_envs, self._window_size), device=self._device)
    self._sum = torch.zeros((self._num_envs), device=self._device)
    self._correction = torch.zeros((self._num_envs), device=self._device)

  def reset(self, reset_idx):
    self._current_size[reset_idx] = 0
    self._value_deque[reset_idx] = 0
    self._sum[reset_idx] = 0
    self._correction[reset_idx] = 0

  def _neumaier_sum(self, value: Tensor) -> Tensor:
    """Update the moving window sum using Neumaier's algorithm.

    For more details please refer to:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    Args:
      value: The new value to be added to the window.
    """
    new_sum = self._sum + value
    bigger_indices = torch.nonzero(
      torch.abs(self._sum) >= torch.abs(value), as_tuple=True)[0]
    smaller_indices = torch.nonzero(
      torch.abs(self._sum) < torch.abs(value), as_tuple=True)[0]
    self._correction[bigger_indices] += (
      (self._sum - new_sum) + value)[bigger_indices]
    self._correction[smaller_indices] += (
      (value - new_sum) + self._sum)[smaller_indices]
    self._sum = new_sum

  def calculate_average(self, new_value: Tensor) -> Tensor:
    """Computes the moving window average in O(1) time.

    Args:
      new_value: The new value to enter the moving window.

    Returns:
      The average of the values in the window.

    """
    unfilled_indices = torch.nonzero(
      self._current_size < self._window_size, as_tuple=True)[0]
    value_deque = deepcopy(self._value_deque)
    value_deque[unfilled_indices] = 0
    self._neumaier_sum(-value_deque[..., 0])
    self._neumaier_sum(new_value)
    self.append_deque(new_value)
    return (self._sum + self._correction) / self._window_size

  def append_deque(self, new_value: Tensor):
    unfilled_indices = torch.nonzero(
      self._current_size < self._window_size, as_tuple=True)[0]
    filled_indices = torch.nonzero(
      self._current_size >= self._window_size, as_tuple=True)[0]
    if unfilled_indices.shape[0] > 0:
      self._value_deque[unfilled_indices,
                        self._current_size[unfilled_indices]] = new_value[unfilled_indices]
      self._current_size[unfilled_indices] += 1
    if filled_indices.shape[0] > 0:
      self._value_deque[filled_indices] = torch.cat(
        [self._value_deque[filled_indices, 1:], new_value[filled_indices].unsqueeze(-1)], dim=-1)


class COMVelocityEstimator(object):
  """Estimate the CoM velocity using on board sensors.


  Requires knowledge about the base velocity in world frame, which for example
  can be obtained from a MoCap system. This estimator will filter out the high
  frequency noises in the velocity so the results can be used with controllers
  reliably.

  """

  def __init__(
      self,
      num_envs: int = 1,
      device: Any = None,
      window_size: int = _DEFAULT_WINDOW_SIZE,
  ):
    self._num_envs = num_envs
    self._device = device
    self._window_size = window_size
    self.init()

  @property
  def com_velocity_body_frame(self) -> Tensor:
    """The base velocity projected in the body aligned inertial frame.

    The body aligned frame is a intertia frame that coincides with the body
    frame, but has a zero relative velocity/angular velocity to the world frame.

    Returns:
      The com velocity in body aligned frame.
    """
    return self._com_velocity_body_frame

  @property
  def com_velocity_world_frame(self) -> Tensor:
    return self._com_velocity_world_frame

  def init(self):
    # We use a moving window filter to reduce the noise in velocity estimation.
    self._velocity_filter_x = MovingWindowFilter(
      self._num_envs, self._device, window_size=self._window_size)
    self._velocity_filter_y = MovingWindowFilter(
      self._num_envs, self._device, window_size=self._window_size)
    self._velocity_filter_z = MovingWindowFilter(
      self._num_envs, self._device, window_size=self._window_size)
    self._com_velocity_world_frame = torch.zeros(
      (self._num_envs, 3), device=self._device)
    self._com_velocity_body_frame = torch.zeros(
      (self._num_envs, 3), device=self._device)

  def reset(self, reset_idx):
    self._velocity_filter_x.reset(reset_idx)
    self._velocity_filter_y.reset(reset_idx)
    self._velocity_filter_z.reset(reset_idx)
    self._com_velocity_world_frame[reset_idx] = 0
    self._com_velocity_body_frame[reset_idx] = 0

  def update(self, robot_states):
    base_quat = robot_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(
      base_quat, robot_states[:, 7:10])
    vx = self._velocity_filter_x.calculate_average(base_lin_vel[..., 0])
    vy = self._velocity_filter_y.calculate_average(base_lin_vel[..., 1])
    vz = self._velocity_filter_z.calculate_average(base_lin_vel[..., 2])
    self._com_velocity_world_frame = torch.stack([vx, vy, vz], dim=-1)
    self._com_velocity_body_frame = quat_apply(
      base_quat, self._com_velocity_world_frame)
