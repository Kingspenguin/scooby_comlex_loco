

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# import torch
# import torchrl.networks.nets as networks
# import torchrl.networks.base as networks_base
# import torchrl.policies as policies
# from torchrl.utils import get_params
# from rich4loco.get_env import get_env
# import pycuda
import glob
import os

import copy

class TRTPolicyWrapper():
    # By Default Use FP16
  def __init__(
          self, 
          engine_path,
          # policy,
          state_dim,
          visual_dim,
          act_dim,
          tanh_action,
          batch_size=1,
  ) -> None:
  # self.pf = policy
  # self.d_input
    self.tanh_action = tanh_action

    # need to set input and output precisions to FP16 to fully enable it
    dummy_input = np.random.rand(
        batch_size, state_dim + np.prod(visual_dim)
    ).astype(np.float16)

    import pycuda.autoinit
    # cuda.init()
    self.cuda_cxt = cuda.Device(0).make_context()

    f = open(engine_path, "rb")
    self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

    self.engine = self.runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()

    # print(pf(d_t))
    # print(d_t.dtype)
    # print(dummy_input.dtype)
    input_batch = np.array(dummy_input)
    self.output = np.empty([batch_size, act_dim], dtype=np.float16)

    # allocate device memory
    self.d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

    self.bindings = [int(self.d_input), int(self.d_output)]

    self.stream = cuda.Stream()

  def predict(self, batch): # result gets copied into output
    # transfer input data to device
    # print(batch.shape, batch.dtype)
    batch = batch.astype(np.float16)
    # print(batch.shape, batch.dtype)
    cuda.memcpy_htod_async(self.d_input, batch, self.stream)
    # execute model
    # print("Copy")
    self.context.execute_async_v2(self.bindings, self.stream.handle, None)
    # print("execute")
    # transfer predictions back
    cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
    # print("Copy")
    # syncronize threads
    self.stream.synchronize()

    # print("Sync")
    output = copy.deepcopy(self.output)
    # print(output, output.shape)
    # print("Out")
    return output

  def eval_act(self, batch):
    mean = self.predict(batch)
    mean = np.squeeze(mean)
    if self.tanh_action:
        mean = np.tanh(mean)
    return mean
