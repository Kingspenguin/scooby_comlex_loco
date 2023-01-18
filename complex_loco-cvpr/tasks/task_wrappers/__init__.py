from tasks.task_wrappers.base_task_wrapper import BaseTaskWrapper
from tasks.task_wrappers.moving_forward import MovingForwardTask
from tasks.task_wrappers.pace_forward import PaceForwardTask
from tasks.task_wrappers.trot_forward import TrotForwardTask
from tasks.task_wrappers.canter_forward import CanterForwardTask
from tasks.task_wrappers.following_command import FollowingCommandTask

all_task_wrappers = {
    "base": BaseTaskWrapper,
    "moving_forward": MovingForwardTask,
    "moving_forward_air0.8_p10": MovingForwardTask,
    "moving_forward_air0.8_p10_c10": MovingForwardTask,
    "pace_forward": PaceForwardTask,
    "pace_forward_air0.4": PaceForwardTask,
    "pace_forward_air0.6": PaceForwardTask,
    "pace_forward_air0.8": PaceForwardTask,
    "pace_forward_air0.8_p04": PaceForwardTask,
    "pace_forward_air0.8_p10": PaceForwardTask,
    "pace_forward_air0.8_p10_c10": PaceForwardTask,
    "pace_forward_air1.0": PaceForwardTask,
    "pace_forward_air1.0_p04": PaceForwardTask,
    "pace_forward_air1.0_p10": PaceForwardTask,
    "pace_forward_air2.0": PaceForwardTask,
    "pace_forward_air4.0": PaceForwardTask,
    "trot_forward": TrotForwardTask,
    "canter_forward": CanterForwardTask,
    "following_command": FollowingCommandTask
}


def build_task_wrapper(name, device, cfg):
  return all_task_wrappers[name](device, cfg)
