from analysis_utils import get_wandb_data, smooth_data
from cmx import doc
from pathlib import Path

import pandas as pd
import wandb

import matplotlib.pyplot as plt

TEACHER_GROUP = "terrain_update_2_multi_pace_forward_air0.8_p10_privilege_ppo_t_p300_25hz_elu_v126_32_095_random"

SMOOTH_PARA = 10


def main():

  summary_list, config_list, name_list, history_list = get_wandb_data(
      TEACHER_GROUP, "isaac-locomotion", "MultiViewLocomotion",
  )

  with doc.table().figure_row() as r:
    plt.title("Privileged Teacher Training")
    plt.plot(
        history_list[0]["Total timesteps:"],
        smooth_data(history_list[0]["Eval Mean reward:"], SMOOTH_PARA),
        label="Privileged Teacher"
    )

    plt.xlabel('Samples')
    plt.legend(loc='best', ncol=1)
    # r.savefig(f"{Path(__file__).stem}/teacher_curve.png")
    r.savefig(f"{Path(__file__).stem}/teacher_curve.pdf")


if __name__ == '__main__':
  doc @ """
    # Comparison With PPO
    """

  main()

  doc.flush()
