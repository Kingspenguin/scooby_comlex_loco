import torch.nn as nn
import torch
from rl_pytorch.encoder.nature import CoarseNatureEncoder, WideNatureEncoder, SmallNatureEncoder, NatureEncoder, CompactNatureEncoder

enc = CoarseNatureEncoder(4, channels=[32, 64, 128], flatten=False)
img = torch.rand((1, 4, 64, 64))
print(enc(img).shape)

layer_list = [
    nn.Conv2d(
        in_channels=32 * 16, out_channels=256,
        kernel_size=5, stride=2, padding=0
    ), nn.ReLU(),
    nn.Conv2d(
        in_channels=256, out_channels=128,
        kernel_size=3, stride=1, padding=0
    ), nn.ReLU(),
    # nn.Conv2d(
    #     in_channels=128, out_channels=64,
    #     kernel_size=5, stride=2, padding=2
    # ), nn.ReLU()
]
convs = nn.Sequential(*layer_list)
img = torch.rand((1, 512, 11, 11))
print(convs(img).shape)


enc = WideNatureEncoder(4, channels=[32, 64, 32 * 6], flatten=False)
img = torch.rand((1, 4, 64, 64))
print(enc(img).shape)

layer_list = [
    nn.Conv2d(
        in_channels=32 * 6, out_channels=256,
        kernel_size=4, stride=2, padding=2
    ), nn.ReLU(),
    nn.Conv2d(
        in_channels=256, out_channels=128,
        kernel_size=4, stride=2, padding=1
    ), nn.ReLU(),
    # nn.Conv2d(
    #     in_channels=128, out_channels=64,
    #     kernel_size=5, stride=2, padding=2
    # ), nn.ReLU()
]
convs = nn.Sequential(*layer_list)
img = torch.rand((1, 192, 6, 6))
print(convs(img).shape)


enc = SmallNatureEncoder(4, channels=[32, 64, 32 * 6], flatten=False)
img = torch.rand((1, 4, 64, 64))
print(enc(img).shape)

# layer_list = [
#     nn.Conv2d(
#         in_channels=32 * 6, out_channels=256,
#         kernel_size=4, stride=2, padding=2
#     ), nn.ReLU(),
#     nn.Conv2d(
#         in_channels=256, out_channels=128,
#         kernel_size=4, stride=2, padding=1
#     ), nn.ReLU(),
#     # nn.Conv2d(
#     #     in_channels=128, out_channels=64,
#     #     kernel_size=5, stride=2, padding=2
#     # ), nn.ReLU()
# ]
# convs = nn.Sequential(*layer_list)
# img = torch.rand((1, 192, 6, 6))
# print(convs(img).shape)

enc = CompactNatureEncoder(4, channels=[32, 64, 32 * 4], flatten=False)
img = torch.rand((1, 4, 64, 64))
print(enc(img).shape)
