

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

import glob
import os

import copy


class LocotransTRTPolicyWrapper():
    # By Default Use FP16
  def __init__(
      self,
      engine_path,
      state_dim,
      visual_dim,
      act_dim,
      batch_size=1,
  ) -> None:
    self.act_dim = act_dim
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

    input_batch = np.array(dummy_input)
    self.output = np.empty(
        [batch_size, act_dim],
        dtype=np.float16
    )

    # allocate device memory
    self.d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

    self.bindings = [int(self.d_input), int(self.d_output)]

    self.stream = cuda.Stream()

  def predict(self, batch):  # result gets copied into output
    # transfer input data to device
    batch = batch.astype(np.float16)
    cuda.memcpy_htod_async(self.d_input, batch, self.stream)
    # execute model
    self.context.execute_async_v2(self.bindings, self.stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
    # syncronize threads
    self.stream.synchronize()

    output = copy.deepcopy(self.output)
    action = output
    return action

  def act_inference(self, obs):
    act = self.predict(obs)
    # act = np.squeeze(act)
    return act


class LocotransTRTGRUPolicyWrapper():
    # By Default Use FP16
  def __init__(
      self,
      engine_path,
      state_dim,
      visual_dim,
      h_dim,
      h_num_layers,
      act_dim,
      batch_size=1,
  ) -> None:
    self.act_dim = act_dim
    self.h_dim = h_dim
    self.h_num_layers = h_num_layers
    # need to set input and output precisions to FP16 to fully enable it
    dummy_input = np.random.rand(
        batch_size, state_dim + np.prod(visual_dim) + h_dim * h_num_layers
    ).astype(np.float16)

    import pycuda.autoinit
    # cuda.init()
    self.cuda_cxt = cuda.Device(0).make_context()

    f = open(engine_path, "rb")
    self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    self.engine = self.runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()

    input_batch = np.array(dummy_input)
    self.output = np.empty(
        [batch_size, act_dim + h_dim * h_num_layers],
        dtype=np.float16
    )

    # allocate device memory
    self.d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

    self.bindings = [int(self.d_input), int(self.d_output)]

    self.stream = cuda.Stream()

  def predict(self, batch):  # result gets copied into output
    # transfer input data to device
    batch = batch.astype(np.float16)
    cuda.memcpy_htod_async(self.d_input, batch, self.stream)
    # execute model
    self.context.execute_async_v2(self.bindings, self.stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
    # syncronize threads
    self.stream.synchronize()

    output = copy.deepcopy(self.output)
    action = output[:, :self.act_dim]
    h = output[:, self.act_dim:]
    return action, h

  def act_inference(self, obs, h):
    batch = np.concatenate([
        obs.reshape(-1), h.reshape(-1)
    ])
    act, h = self.predict(batch)
    act = np.squeeze(act)
    return act, h
