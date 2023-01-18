from tasks.learner.gail import GAIL

all_learners = {"gail": GAIL}


def build_learner(name, num_envs, device, cfg, control_dt):
  learner_cfg = cfg["learners"][name]
  return all_learners[name](
      num_envs=num_envs, device=device, learner_cfg=learner_cfg, control_dt=control_dt
  )
