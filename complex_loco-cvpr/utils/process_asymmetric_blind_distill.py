# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from rl_pytorch.ppo import ActorCritic, get_encoder
from rl_pytorch.asymmetric_blind_distill import AsymmetricBlindDistillTrainer


def process_asymmetric_blind_distill_trainer(args, env, cfg_train, teacher_logdir, student_logdir):
  learn_cfg = cfg_train["learn"]
  is_testing = learn_cfg["test"]
  chkpt = learn_cfg["resume"]
  teacher_encoder_type = cfg_train["policy"]["teacher_encoder_params"]["encoder_type"]
  student_encoder_type = cfg_train["policy"]["student_encoder_params"]["encoder_type"]
  # Override resume and testing flags if they are passed as parameters.
  if not is_testing:
    is_testing = args.test
  if args.resume > 0:
    chkpt = args.resume
  teacher_resume = args.teacher_resume
  """Set up the PPO system for training or inferencing."""
  trainer = AsymmetricBlindDistillTrainer(vec_env=env,
                                     actor_critic_class=ActorCritic,
                                     teacher_encoder=get_encoder(
                                       teacher_encoder_type),
                                     student_encoder=get_encoder(
                                       student_encoder_type),
                                     num_transitions_per_env=learn_cfg["nsteps"],
                                     num_learning_epochs=learn_cfg["noptepochs"],
                                     num_mini_batches=learn_cfg["nminibatches"],
                                     clip_param=learn_cfg["cliprange"],
                                     gamma=learn_cfg["gamma"],
                                     lam=learn_cfg["lam"],
                                     init_noise_std=learn_cfg.get(
                                       "init_noise_std", 0.3),
                                     value_loss_coef=learn_cfg.get(
                                       "value_loss_coef", 2.0),
                                     bc_loss_coef=learn_cfg.get(
                                       "bc_loss_coef", 1.0),
                                     entropy_coef=learn_cfg["ent_coef"],
                                     learning_rate=learn_cfg["optim_stepsize"],
                                     max_grad_norm=learn_cfg.get(
                                       "max_grad_norm", 2.0),
                                     use_clipped_value_loss=learn_cfg.get(
                                       "use_clipped_value_loss", False),
                                     schedule=learn_cfg.get(
                                       "schedule", "fixed"),
                                     desired_kl=learn_cfg.get(
                                       "desired_kl", None),
                                     model_cfg=cfg_train["policy"],
                                     device=env.rl_device,
                                     sampler=learn_cfg.get(
                                       "sampler", 'sequential'),
                                     teacher_log_dir=teacher_logdir,
                                     student_log_dir=student_logdir,
                                     teacher_resume=teacher_resume,
                                     is_testing=is_testing,
                                     print_log=learn_cfg["print_log"],
                                     apply_reset=False,
                                     asymmetric=False,
                                     vidlogdir='video',
                                     log_video=args.log_video,
                                     vid_log_step=args.vid_log_step,
                                     local=args.local,
                                     learn_value_by_self=args.learn_value_by_self,
                                     mask_base_vel=args.mask_base_vel,
                                     )

  if is_testing:
    print("Loading model from {}/model_{}.pt".format(student_logdir, chkpt))
    trainer.test_student("{}/model_{}.pt".format(student_logdir, chkpt))
  elif chkpt > 0:
    print("Loading model from {}/model_{}.pt".format(student_logdir, chkpt))
    trainer.student_load("{}/model_{}.pt".format(student_logdir, chkpt))

  return trainer
