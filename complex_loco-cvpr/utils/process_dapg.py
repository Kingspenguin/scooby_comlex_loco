# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from rl_pytorch.ppo import ActorCritic, get_encoder
from rl_pytorch.dapg import DAPGTrainer


def process_dapg_trainer(args, env, cfg_train, teacher_logdir, student_logdir):
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

  """Set up the PPO system for training or inferencing."""
  dapg_trainer = DAPGTrainer(
    vec_env=env,
    actor_critic_class=ActorCritic,
    height_encoder=get_encoder(encoder_type),
    vis_encoder=get_encoder(encoder_type),
    num_transitions_per_env=learn_cfg["nsteps"],
    num_learning_epochs=learn_cfg["noptepochs"],
    num_mini_batches=learn_cfg["nminibatches"],
    clip_param=learn_cfg["cliprange"],
    gamma=learn_cfg["gamma"],
    lam=learn_cfg["lam"],
    init_noise_std=learn_cfg.get("init_noise_std", 0.3),
    value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
    bc_coef=learn_cfg.get("bc_coef", 1.0),
    entropy_coef=learn_cfg["ent_coef"],
    learning_rate=learn_cfg["optim_stepsize"],
    max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
    use_clipped_value_loss=learn_cfg.get(
      "use_clipped_value_loss", False),
    schedule=learn_cfg.get("schedule", "fixed"),
    desired_kl=learn_cfg.get("desired_kl", None),
    model_cfg=cfg_train["policy"],
    device=env.rl_device,
    sampler=learn_cfg.get("sampler", 'sequential'),
    student_log_dir=student_logdir,
    is_testing=is_testing,
    print_log=learn_cfg["print_log"],
    apply_reset=False,
    asymmetric=False,
    vidlogdir=args.vidlogdir,
    vid_log_step=args.vid_log_step,
    log_video=args.log_video,
    storage_load_path=args.storage_load_path
  )

  if is_testing:
    print("Loading model from {}/model_{}.pt".format(student_logdir, chkpt))
    dapg_trainer.test_student("{}/model_{}.pt".format(student_logdir, chkpt))
  elif chkpt > 0:
    print("Loading model from {}/model_{}.pt".format(student_logdir, chkpt))
    dapg_trainer.student_load("{}/model_{}.pt".format(student_logdir, chkpt))

  return dapg_trainer
