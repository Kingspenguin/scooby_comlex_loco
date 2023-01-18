from cv2 import blur
import pyrealsense2 as rs
import os
import numpy as np
import cv2
import datetime
import time
import threading
from a1_utilities.logger import StateLogger
from a1_utilities.logger import VisualLogger
import multiprocessing as mp
from multiprocessing import Manager
mp.set_start_method('spawn', force=True)


RGB_WIDTH = 320
RGB_HEIGHT = 180
RGB_RATE = 30

DEPTH_WIDTH = 424
DEPTH_HEIGHT = 240
DEPTH_RATE = 30


def process_depth(depth, target_shape):
  depth_dim = depth.shape
  horizontal_clip = int(depth_dim[0] / 10)
  resized_depth_image = cv2.resize(
      depth[:, horizontal_clip:],
      dsize=target_shape,
      interpolation=cv2.INTER_NEAREST
  )
  blurred_depth_image = cv2.medianBlur(resized_depth_image, 3)
  # blurred_depth_image[~np.isfinite(blurred_depth_image)] = 0
  # blurred_depth_image[blurred_depth_image > 5] = 5
  # blurred_depth_image = -1.0 * blurred_depth_image
  # return resized_depth_image
  return blurred_depth_image


def process_depth_old(depth, target_shape):
  depth_dim = depth.shape
  horizontal_clip = int(depth_dim[0] / 10)
  resized_depth_image = cv2.resize(
      depth[:, horizontal_clip: -horizontal_clip],
      dsize=target_shape,
      interpolation=cv2.INTER_NEAREST
  )
  blurred_depth_image = cv2.medianBlur(resized_depth_image, 3)
  return blurred_depth_image
# We do not use RGB rightnow


def process_rgb(rgb):
  return rgb


class A1RealSense:
  '''
  A1RealSense:
    wrapper of the pyrealsense functions for streaming RealSense Reading
  '''

  def __init__(
      self,
      target_shape=(64, 64),
      use_depth=True,
      use_rgb=False,
      save_frames=False,
      save_dir_name=None,
      update_rate=1,
      idx=0,
  ):
    self.use_depth = use_depth
    self.use_rgb = use_rgb

    self.target_shape = target_shape
    if self.use_rgb:
      # Where we keep track of rgbd frame
      self.rgb_frame = np.zeros(target_shape + (3,))
    if self.use_depth:
      # Where we keep track of rgbd frame
      self.depth_frame = np.zeros(target_shape + (1,))

    self.continue_execution = False

    self.save_frames = save_frames
    self.save_dir_name = save_dir_name

    self.idx = idx

    self.manager = Manager()
    self.shared_dict = self.manager.dict()

    self.update_rate = update_rate

  def start_thread(self):
    assert self.find_camera()
    # self.main_thread = threading.Thread(target=self.streaming)
    self.main_process = mp.Process(
        target=self.streaming,
        args=(
            self.shared_dict,
            self.idx,
            self.use_rgb, self.use_depth,
            self.target_shape,
            self.save_frames,
            self.save_dir_name,
            self.update_rate
        )
    )
    self.continue_execution = True
    self.shared_dict["execution"] = True
    self.main_process.start()

  def stop_thread(self):
    self.continue_execution = False
    self.shared_dict["execution"] = False
    self.main_process.join()
    # if self.save_frames:
    #   self.depth_logger.write()
    print("Realsense thread terminated")

  def find_camera(self):
    self.ctx = rs.context()
    print(self.ctx.query_devices())
    print(self.idx)
    self.device = self.ctx.query_devices()[self.idx]
    self.device_product_line = str(
        self.device.get_info(rs.camera_info.product_line))
    print("Found: ", self.device_product_line)
    self.found_rgb = False
    self.found_depth = False
    for s in self.device.sensors:
      if s.get_info(rs.camera_info.name) == 'RGB Camera':
        self.found_rgb = True
      if s.get_info(rs.camera_info.name) == 'Stereo Module':
        self.found_depth = True
    assert self.found_depth or (not self.use_depth)
    assert self.found_rgb or (not self.use_rgb)
    return (self.found_depth or (not self.use_depth)) \
        and self.found_rgb or (not self.use_rgb)

  @staticmethod
  def streaming(
      shared_dict,
      idx,
      use_rgb, use_depth,
      target_shape,
      save_frames,
      save_dir_name,
      update_rate
  ):
    ctx = rs.context()
    device = ctx.query_devices()[idx]
    device_product_line = str(
        device.get_info(rs.camera_info.product_line)
    )
    print("Found: ", device_product_line)
    found_rgb = False
    found_depth = False
    for s in device.sensors:
      if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
      if s.get_info(rs.camera_info.name) == 'Stereo Module':
        found_depth = True
    assert found_depth or (not use_depth)
    assert found_rgb or (not use_rgb)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(
        device.get_info(rs.camera_info.serial_number)
    )
    if use_rgb:
      config.enable_stream(
          rs.stream.color,
          RGB_WIDTH, RGB_HEIGHT,
          rs.format.bgr8, RGB_RATE
      )
    if use_depth:
      config.enable_stream(
          rs.stream.depth,
          DEPTH_WIDTH, DEPTH_HEIGHT,
          rs.format.z16,
          DEPTH_RATE
      )
    profile = pipeline.start(config)
    print("started: ", idx)

    hole_filling_filter = rs.hole_filling_filter(2)

    if save_frames:
      depth_logger = VisualLogger(
          save_path=os.path.join(save_dir_name, "depth")
      )
    depth_scale = profile.get_device(
    ).first_depth_sensor().get_depth_scale()
    shared_dict["depth_scale"] = depth_scale
    frame_count = -1
    try:
      while shared_dict["execution"]:
        frames = pipeline.wait_for_frames()
        frame_count += 1

        if frame_count % update_rate != 0:
          continue

        if use_rgb:
          color_frame = frames.get_color_frame()
          assert color_frame
          # convert images to numpy arrays
          color_image = np.asanyarray(color_frame.get_data())
          rgb_frame = process_rgb(color_image, target_shape)
          shared_dict["rgb_frame"] = rgb_frame

        if use_depth:
          depth_frame = frames.get_depth_frame()
          assert depth_frame
          depth_frame = hole_filling_filter.process(depth_frame)
          # convert images to numpy arrays
          depth_image = np.asanyarray(depth_frame.get_data())
          depth_frame = process_depth(depth_image, target_shape)
          if save_frames:
            depth_logger.record(depth_frame * depth_scale)
            depth_logger.record_img(depth_frame * depth_scale)
          shared_dict["depth_frame"] = depth_frame

    finally:
      pipeline.stop()

  def get_depth_scale(self):
    return self.shared_dict["depth_scale"]

  def get_depth_frame(self):
    # depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
    return self.shared_dict["depth_frame"]

  def get_rgb_frame(self):
    return self.shared_dict["rgb_frame"]


def display(camera):
    # convert rgbd to two rgb frames for display
  depth_image = camera.get_depth_frame()
  color_image = camera.get_rgb_frame()
  depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
      depth_image, alpha=0.03), cv2.COLORMAP_BONE)  # JET)
  images = np.hstack((color_image, depth_colormap))
  cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
  cv2.imshow('RealSense', depth_colormap)
  cv2.waitKey(1)


if __name__ == '__main__':
  a1_camera = A1RealSense(use_depth=True)
  a1_camera.start_thread()

  time.sleep(3)
  while True:
    display()