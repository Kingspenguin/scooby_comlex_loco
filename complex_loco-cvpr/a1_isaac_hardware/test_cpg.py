import torch
import numpy as np
from control_loop_execution.cpg_rl_policy_wrapper import CPGPolicyWrapper


control_freq = 20
get_image_interval = 1
save_dir_name = None
policyComputer = CPGPolicyWrapper(
    None,
    {},
    get_image_interval,
    save_dir_name=save_dir_name,
    no_tensor=False,
    state_only=True,
    action_scale=[0.1, 0.25, 0.25],
    phase_scale=0.1,
    use_foot_contact=False,
    use_com_vel=True,
    use_command=True,
    save_log=False,
    num_hist=2,
    phase_dt=1 / control_freq,
    Kp=60, Kd=0.8
)

with np.printoptions(precision=4, suppress=True):
  policyComputer.set_cpg_mode(True)
  action = torch.Tensor([
      0.0178, 0.1636, 0.2500, -0.0372, -0.0600, -0.2003, 0.0135, -0.1757,
      -0.2450, -0.0503, 0.0468, 0.2487, -0.0455, 0.0189, -0.0056, -0.0188]
  )
  policyComputer.current_joint_angle = np.array([
      0.4145, 1.0613, -1.7449, -0.5280, 0.7669, -2.0535, -0.0609, 0.9305,
      -1.8407, -0.0125, 1.0548, -1.4432
  ])
  print("action scale:", policyComputer.action_scale)
  target_joint_pos = policyComputer.get_cpg_action(action)
  print(target_joint_pos)

  print(policyComputer.gait_generator.phase_info.reshape(-1))
