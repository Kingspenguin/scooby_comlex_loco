from .impala import ImpalaEncoder
from .locotransformer import *
from .voxel import *
from .nature import *
from .mlp import *
from .resnet18_voxel import ResNet18VoxelEncoder
from .resnet18_voxel import ResNet18VoxelV2Encoder
from .resnet18_voxel import ResNet18VoxelV3Encoder
from .resnet18 import ResNet18CatEncoder, ResNet18MultiStepEncoder
from .voxel_trans import NatureVoxelTransEncoder, Res18VoxelTransEncoder
from .voxel_trans import CoarseNatureVoxelTransEncoder, SmallNatureVoxelTransEncoder, CompactNatureVoxelTransEncoder


def get_encoder(encoder_type):
  if encoder_type == "mlp":
    return None
  if encoder_type == "height_mlp":
    return MLPHeightEncoder
  if encoder_type == "multi_height_mlp":
    return MLPMultiHeightEncoder
  elif encoder_type == "pure_nature":
    return NaturePureEncoder
  elif encoder_type == "nature":
    return NatureFuseEncoder
  elif encoder_type == "nature_cat":
    return NatureCatEncoder
  elif encoder_type == "nature_add":
    return NatureFuseAddEncoder
  elif encoder_type == "nature_multiply":
    return NatureFuseMultiplyEncoder
  elif encoder_type == "nature_lstm":
    return NatureFuseLSTMEncoder
  elif encoder_type == "nature_lstm_v2":
    return NatureFuseLSTMEncoderV2
  elif encoder_type == "nature_lstm_1cam_multiply":
    return NatureFuseLSTM1CamMultiplyEncoder
  elif encoder_type == "nature_separate_lstm":
    return NatureFuseLSTMSeparateEncoder
  elif encoder_type == "nature_sep":
    return NatureSepFuseEncoder
  elif encoder_type == "nature_att":
    return NatureAttEncoder
  elif encoder_type == "nature_lstm_add":
    return NatureFuseLSTMAddEncoder
  elif encoder_type == "nature_lstm_multiply":
    return NatureFuseLSTMMultiplyEncoder
  elif encoder_type == "nature_lstm_multiply_v2":
    return NatureFuseLSTMMultiplyEncoderV2
  elif encoder_type == "nature_lstm_concat_v2":
    return NatureFuseLSTMConcatEncoderV2
  elif encoder_type == "nature_lstm_all_multiply":
    return NatureFuseLSTMAllMultiplyEncoder
  elif encoder_type == "nature_lstm_multiply_attention_v2":
    return NatureFuseLSTMMultiplyAttentionEncoderV2
  elif encoder_type == "nature_lstm_multiply_all_attention_v2":
    return NatureFuseLSTMMultiplyAllAttentionEncoderV2
  elif encoder_type == "locotransformer_lstm":
    return LocoTransformerLSTMEncoder
  elif encoder_type == "locotransformer":
    return LocoTransformerEncoder
  elif encoder_type == "locotransformer_v2":
    return LocoTransformerV2Encoder
  elif encoder_type == "impala_voxel":
    return ImpalaVoxelEncoder
  elif encoder_type == "nature_voxel":
    return NatureVoxelEncoder
  elif encoder_type == "nature_voxel_trans":
    return NatureVoxelTransEncoder
  elif encoder_type == "nature_voxel_v2":
    return NatureVoxelV2Encoder
  elif encoder_type == "nature_voxel_v3":
    return NatureVoxelV3Encoder
  elif encoder_type == "coarse_nature_cat":
    return CoarseNatureCatEncoder
  elif encoder_type == "coarse_nature_multi_cat":
    return CoarseNatureMultiStepEncoder
  elif encoder_type == "coarse_nature_voxel":
    return CoarseNatureVoxelEncoder
  elif encoder_type == "coarse_nature_voxel_trans":
    return CoarseNatureVoxelTransEncoder
  elif encoder_type == "coarse_nature_voxel_v2":
    return CoarseNatureVoxelV2Encoder
  elif encoder_type == "coarse_nature_voxel_v3":
    return CoarseNatureVoxelV3Encoder
  elif encoder_type == "small_nature_cat":
    return SmallNatureCatEncoder
  elif encoder_type == "small_nature_multi_cat":
    return SmallNatureMultiStepEncoder
  elif encoder_type == "small_nature_voxel":
    return SmallNatureVoxelEncoder
  elif encoder_type == "small_nature_voxel_trans":
    return SmallNatureVoxelTransEncoder
  # elif encoder_type == "compact_nature_cat":
  #   return CompactNatureCatEncoder
  # elif encoder_type == "compact_nature_multi_cat":
  #   return CompactNatureMultiStepEncoder
  elif encoder_type == "compact_nature_voxel":
    return CompactNatureVoxelEncoder
  elif encoder_type == "compact_nature_voxel_trans":
    return CompactNatureVoxelTransEncoder
  elif encoder_type == "resnet18_cat":
    return ResNet18CatEncoder
  elif encoder_type == "resnet18_multi_cat":
    return ResNet18MultiStepEncoder
  elif encoder_type == "resnet18_voxel":
    return ResNet18VoxelEncoder
  elif encoder_type == "resnet18_voxel_trans":
    return Res18VoxelTransEncoder
  elif encoder_type == "resnet18_voxel_v2":
    return ResNet18VoxelV2Encoder
  elif encoder_type == "resnet18_voxel_v3":
    return ResNet18VoxelV3Encoder
  elif encoder_type == "locotransformer_resnet18":
    return LocoTransformerResNet18Encoder
  elif encoder_type == "locotransformer_coarse_nature":
    return LocoTransformerCoarseNatureEncoder
  elif encoder_type == "locotransformer_small_nature":
    return LocoTransformerSmallNatureEncoder
  else:
    raise NotImplementedError
