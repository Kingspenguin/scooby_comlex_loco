import time
import torch
from rl_pytorch.encoder.impala import ImpalaEncoder
from rl_pytorch.encoder.resnet18 import get_small_resnet18

impala_encoder = ImpalaEncoder(in_channels=1, channels=[
                               32, 64, 128], flatten=False).to("cuda:0")
print(impala_encoder)

res18 = get_small_resnet18(pretrained=False, input_channels=1).to("cuda:0")

random_input = torch.rand(32, 1, 64, 64).to("cuda:0")

t = time.time()
for _ in range(1000):
  out = impala_encoder(random_input)
print("Impala Time:", time.time() - t)
print(out.shape)


t = time.time()
for _ in range(1000):
  out = res18(random_input)
print("Res18 Time:", time.time() - t)
print(out.shape)
