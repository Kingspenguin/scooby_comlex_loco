from mpc_controller import com_velocity_estimator
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller


def process_controller(task):
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(task)

  state_estimator = com_velocity_estimator.COMVelocityEstimator(
    robot_task=task, window_size=20)

  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
    task,
    gait_generator,
    state_estimator,
    desired_speed=desired_speed,
    desired_twisting_speed=desired_twisting_speed,
    desired_height=0.24,
    foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
    task,
    gait_generator,
    state_estimator,
    desired_speed=desired_speed,
    desired_twisting_speed=desired_twisting_speed,
    desired_body_height=0.24)

  controller = locomotion_controller.LocomotionController(
    task,
    gait_generator=gait_generator,
    state_estimator=state_estimator,
    swing_leg_controller=sw_controller,
    stance_leg_controller=st_controller,)

  return controller
