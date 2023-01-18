from cmx import doc
from pathlib import Path

import pandas as pd
import wandb

import matplotlib.pyplot as plt

from analysis_utils import get_wandb_data, smooth_data

NVM_GROUP = "terrain_vis1c_17_4_up2_multi_10_moving_forward_air0.8_p10_nat_1cam_bcaux_001_p300_10_v149_5128dr02_bch17_1cnatvts"
NACNN_GROUP = "terrain_vis1c_17_4_up2_multi_10_moving_forward_air0.8_p10_rescnn_1cam_bc_p300_10_v149_5128dr02_bch17_1cnatc"
NACNN_RNN_GROUP = "terrain_vis1c_17_4_up2_multi_10_moving_forward_air0.8_p10_rescnn_1cam_rnnbc_p300_10_v149_5128dr02_bch17_1cnatc"
LT_GROUP = "terrain_vis1c_17_4_up2_multi_10_moving_forward_air0.8_p10_reslocot_1cam_bc_p300_10_v149_5128dr02_bch17_1cnatt"
MST_GROUP = "terrain_vis1c_17_4_up2_multi_10_moving_forward_air0.8_p10_resmst_1cam_bc_p300_10_v149_5128dr02_bch17_1cnatmst"

SMOOTH_PARA = 10


def main():

  _, _, _, ours_history_list = get_wandb_data(
      NVM_GROUP, "isaac-locomotion", "MultiViewLocomotion",
  )

  _, _, _, nacnn_history_list = get_wandb_data(
      NACNN_GROUP, "isaac-locomotion", "MultiViewLocomotion",
  )

  _, _, _, mst_history_list = get_wandb_data(
      MST_GROUP, "isaac-locomotion", "MultiViewLocomotion",
  )

  _, _, _, nacnn_rnn_history_list = get_wandb_data(
      NACNN_RNN_GROUP, "isaac-locomotion", "MultiViewLocomotion",
  )

  _, _, _, lt_history_list = get_wandb_data(
      LT_GROUP, "isaac-locomotion", "MultiViewLocomotion",
  )

  with doc.table().figure_row() as r:
    plt.title("Student Distillation")
    plt.plot(
        ours_history_list[0]["Total timesteps:"],
        smooth_data(ours_history_list[0]["Eval Mean reward:"], SMOOTH_PARA),
        label="NVM"
    )
    plt.plot(
        nacnn_history_list[0]["Total timesteps:"],
        smooth_data(nacnn_history_list[0]["Eval Mean reward:"], SMOOTH_PARA),
        label="NaiveCNN"
    )

    plt.plot(
        nacnn_rnn_history_list[0]["Total timesteps:"],
        smooth_data(
            nacnn_rnn_history_list[0]["Eval Mean reward:"], SMOOTH_PARA),
        label="NaiveCNN-RNN"
    )

    plt.plot(
        lt_history_list[0]["Total timesteps:"],
        smooth_data(lt_history_list[0]["Eval Mean reward:"], SMOOTH_PARA),
        label="LocoTransformer"
    )

    plt.plot(
        mst_history_list[0]["Total timesteps:"],
        smooth_data(mst_history_list[0]["Eval Mean reward:"], SMOOTH_PARA),
        label="Multi-Step-NaiveCNN"
    )

    plt.xlabel('Samples')
    plt.ylabel('Avg Episode Return')
    plt.ylim(bottom=0)
    plt.xlim(0, 2.0e7)
    plt.legend(loc='best', ncol=2, fontsize=14)
    # r.savefig(f"{Path(__file__).stem}/distillation_curve.png")
    r.savefig(f"{Path(__file__).stem}/distillation_curve.pdf")


if __name__ == '__main__':
  doc @ """
    # Comparison With PPO
    """

  main()

  def plot_line():
    pass

  doc.flush()
