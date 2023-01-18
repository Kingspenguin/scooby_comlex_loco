from .discriminators import Discriminator
from .visual_rma_networks import EFEncoder, AdaptationModule
from .custom_lstm import script_lnlstm

functional_networks = {
  "discriminator": Discriminator,
  "ef_encoder": EFEncoder,
  "adaptation_module": AdaptationModule,
  "lnlstm": script_lnlstm
}


def get_network(name, cfg):
  if name not in functional_networks:
    raise ValueError("{} is not a valid network name".format(name))
  return functional_networks[name](cfg)
