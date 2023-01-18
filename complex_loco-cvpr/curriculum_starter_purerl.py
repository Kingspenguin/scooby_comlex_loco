import time
import os
import sys
import subprocess
import argparse


def checkNotFinish(popen_list):
  for eachpopen in popen_list:
    if eachpopen.poll() == None:
      return True
  return False

# --headless
# --num_envs 256
# --cfg_train cfg/student_draw_curves/train/2cam_multiply.yaml
# --cfg_env cfg/student_draw_curves/2cam.yaml
# --teacher_logdir teachers/asymmetric_distill/cpg/
# --teacher_resume 30000
# --logdir /minghao-isaac/may6-student-curves/mask_base_vel/2cam_multiply/
# --resume 10000
# --max_iterations 12000
# --add_label mask_base_vel_corl_2cam_multiply
# --seed 1
# --mask_base_vel;


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--file', type=str,
                    default='train_asymmetric_distill_recurrent.py')
parser.add_argument("--interval", type=int, default=1000,)
parser.add_argument("--num_envs", type=int, default=512,)
parser.add_argument("--max_iter", type=int, default=30000,)
parser.add_argument('--cfg_train', type=str, default='ruihan')
parser.add_argument('--cfg_env', type=str, default='ruihan')
parser.add_argument('--teacher_logdir', type=str, default='ruihan')
parser.add_argument("--teacher_resume", type=int, default=30000,)
parser.add_argument('--logdir', type=str, default='ruihan')
parser.add_argument('--add_label', type=str, default='ruihan')
# # parser.add_argument( "--resume", type=int, default=None,)
parser.add_argument('--seed_base', type=int, default=0,
                    help='random seed (default: (0,))')
parser.add_argument('--learn_value_by_self', type=bool, default=False,
                    help='value function')
# parser.add_argument('--env_name', type=str, default="HalfCheetah-v2")
# parser.add_argument('--name_space', type=str, default="rl-multitask")
# parser.add_argument('--log', action='store_true')
# parser.add_argument('--vmpo', action='store_true')
# parser.add_argument('--repo', type=str, default="torchmetarl")
# parser.add_argument('--pod_name', type=str, default=None)
# parser.add_argument('--env', type=str, default=None)

args = parser.parse_args()


def run_command(command):
  p = subprocess.Popen(command, shell=True)
  while True:
    if p.poll() != None:
      break


def run_exp():
  start_command = "python {} --headless --num_envs {} --cfg_train {} --cfg_env {} --logdir {} --max_iterations {} --add_label {} --seed {} "
  base_command = "python {} --headless  --num_envs {} --cfg_train {} --cfg_env {} --logdir {} --resume {} --max_iterations {} --add_label {} --seed {} --wandb_resume True"
  num_exp = len(list(range(0, args.max_iter, args.interval)))
  for idx, start_iter in enumerate(range(0, args.max_iter, args.interval)):
    if start_iter == 0:
      command = start_command.format(
          args.file,
          args.num_envs,
          args.cfg_train, args.cfg_env, args.logdir, start_iter +
          args.interval, args.add_label,
          args.seed_base * num_exp + idx
      )
    else:
      command = base_command.format(
          args.file,
          args.num_envs,
          args.cfg_train, args.cfg_env, args.logdir,
          start_iter,
          start_iter + args.interval, args.add_label,
          args.seed_base * num_exp + idx
      )
    print(command)
    run_command(command)


if __name__ == "__main__":
  run_exp()
