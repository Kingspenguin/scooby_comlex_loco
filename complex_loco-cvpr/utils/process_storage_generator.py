# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from rl_pytorch.ppo import ActorCritic, get_encoder
from rl_pytorch.storage_generator import StorageGenerator


def process_storage_generator(args, env, cfg_train, teacher_logdir):
  learn_cfg = cfg_train["learn"]
  is_testing = learn_cfg["test"]
  chkpt = learn_cfg["resume"]
  encoder_type = cfg_train["policy"]["encoder_type"]
  # Override resume and testing flags if they are passed as parameters.
  if not is_testing:
    is_testing = args.test
  if args.resume > 0:
    chkpt = args.resume

  teacher_resume = args.teacher_resume

  storage_generator = StorageGenerator(
    vec_env=env,
    actor_critic_class=ActorCritic,
    height_encoder=get_encoder(encoder_type),
    vis_encoder=get_encoder(encoder_type),
    num_teacher_transitions=learn_cfg["nsteps_teacher"],
    init_noise_std=learn_cfg.get("init_noise_std", 0.3),
    model_cfg=cfg_train["policy"],
    device=env.rl_device,
    teacher_log_dir=teacher_logdir,
    asymmetric=False,
    teacher_resume=teacher_resume,
    vidlogdir=args.vidlogdir,
    vid_log_step=args.vid_log_step,
    log_video=args.log_video,
    storage_save_path=args.storage_save_path,
  )
  return storage_generator
