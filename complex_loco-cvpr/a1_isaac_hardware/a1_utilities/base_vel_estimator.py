"""Estimates base velocity for A1 robot from accelerometer readings."""
import collections
import numpy as np
from filterpy.kalman import KalmanFilter

"""Moving window filter to smooth out sensor readings."""


class MovingWindowFilter(object):
  """A stable O(1) moving filter for incoming data streams.

  We implement the Neumaier's algorithm to calculate the moving window average,
  which is numerically stable.

  """

  def __init__(self, window_size: int, dim: int = 3):
    """Initializes the class.

    Args:
      window_size: The moving window size.
    """
    assert window_size > 0
    self._window_size = window_size
    self._value_deque = collections.deque(maxlen=window_size)
    # The moving window sum.
    self._sum = np.zeros(dim)
    # The correction term to compensate numerical precision loss during
    # calculation.
    self._correction = np.zeros(dim)

  def _neumaier_sum(self, value: np.ndarray):
    """Update the moving window sum using Neumaier's algorithm.

    For more details please refer to:
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

    Args:
      value: The new value to be added to the window.
    """

    new_sum = self._sum + value
    self._correction = np.where(
        np.abs(self._sum) >= np.abs(value),
        self._correction + (self._sum - new_sum) + value,
        self._correction + (value - new_sum) + self._sum)
    self._sum = new_sum

  def calculate_average(self, new_value: np.ndarray) -> np.ndarray:
    """Computes the moving window average in O(1) time.

    Args:
      new_value: The new value to enter the moving window.

    Returns:
      The average of the values in the window.

    """
    deque_len = len(self._value_deque)
    if deque_len < self._value_deque.maxlen:
      pass
    else:
      # The left most value to be subtracted from the moving sum.
      self._neumaier_sum(-self._value_deque[0])

    self._neumaier_sum(new_value)
    self._value_deque.append(new_value)

    return (self._sum + self._correction) / self._window_size


class A1RobotStateEstimator:
  """Estimates base velocity of A1 robot.
  The velocity estimator consists of a state estimator for CoM velocity.
  Two sources of information are used:
  The integrated reading of accelerometer and the velocity estimation from
  contact legs. The readings are fused together using a Kalman Filter.
  """

  def __init__(
      self,
      robot,
      obs_interval,
      accelerometer_variance=np.array(
          [1.42072319e-05, 1.57958752e-05, 8.75317619e-05]),
      sensor_variance=np.array([0.33705298, 0.14858707, 0.68439632]) * 0.03,
      initial_variance=0.1
  ):
    """Initiates the velocity estimator.
    See filterpy documentation in the link below for more details.
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    Args:
      robot: the robot class for velocity estimation.
      accelerometer_variance: noise estimation for accelerometer reading.
      sensor_variance: noise estimation for motor velocity reading.
      initial_covariance: covariance estimation of initial state.
    """
    self.robot = robot
    self.obs_interval = obs_interval
    self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
    self.filter.x = np.zeros(3)
    self._initial_variance = initial_variance
    self._accelerometer_variance = accelerometer_variance
    self._sensor_variance = sensor_variance
    self.filter.P = np.eye(3) * self._initial_variance  # State covariance
    self.filter.Q = np.eye(3) * accelerometer_variance
    self.filter.R = np.eye(3) * sensor_variance

    self.filter.H = np.eye(3)  # measurement function (y=H*x)
    self.filter.F = np.eye(3)  # state transition matrix
    self.filter.B = np.eye(3)
    self.reset()

  def reset(self):
    self.filter.x = np.zeros(3)
    self.filter.P = np.eye(3) * self._initial_variance
    self._last_timestamp = 0
    self._last_base_velocity_sim = np.zeros(3)
    self._estimated_velocity = self.filter.x.copy()

  def _compute_delta_time(self, robot_state):
    del robot_state  # unused
    if self._last_timestamp == 0.:
      # First timestamp received, return an estimated delta_time.
      delta_time_s = self.obs_interval
    else:
      delta_time_s = self.robot.time_since_reset - self._last_timestamp
    self._last_timestamp = self.robot.time_since_reset
    return delta_time_s

  def _get_velocity_observation(self):
    base_orientation = self.robot.base_orientation_quat
    rot_mat = self.robot.pybullet_client.getMatrixFromQuaternion(
        base_orientation)
    rot_mat = np.array(rot_mat).reshape((3, 3))
    observed_velocities = []
    foot_contact = self.robot.foot_contacts

    for leg_id in range(4):
      if foot_contact[leg_id]:
        jacobian = self.robot.compute_foot_jacobian(leg_id)
        # Only pick the jacobian related to joint motors
        joint_velocities = self.robot.motor_velocities[
            leg_id * 3:(leg_id + 1) * 3]
        leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
        base_velocity_in_base_frame = -leg_velocity_in_base_frame[:3]
        observed_velocities.append(rot_mat.dot(base_velocity_in_base_frame))

    return observed_velocities

  def update(self, robot_state):
    """Propagate current state estimate with new accelerometer reading."""
    delta_time_s = self._compute_delta_time(robot_state)
    sensor_acc = np.array(robot_state.imu.accelerometer)
    base_orientation = self.robot.base_orientation_quat
    rot_mat = self.robot.pybullet_client.getMatrixFromQuaternion(
        base_orientation)
    rot_mat = np.array(rot_mat).reshape((3, 3))
    calibrated_acc = rot_mat.dot(sensor_acc) + np.array([0., 0., -9.8])
    self.filter.predict(u=calibrated_acc * delta_time_s)

    observed_velocities = self._get_velocity_observation()

    if observed_velocities:
      observed_velocities = np.mean(observed_velocities, axis=0)
      # multiplier = np.clip(
      #     1 + (np.sqrt(observed_velocities[0]**2 + \
      #     observed_velocities[1]**2) -
      #          0.3), 1, 1.3)
      # observed_velocities[0] *= 1.3
      self.filter.update(observed_velocities)

    self._estimated_velocity = self.filter.x.copy()

  @property
  def estimated_velocity(self):
    return self._estimated_velocity.copy()
