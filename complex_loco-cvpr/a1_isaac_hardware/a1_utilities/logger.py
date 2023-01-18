import numpy as np
import cv2
import os
import time
import pathlib
from collections import deque
import threading
import os


class VisualLogger():
    def __init__(
        self,
        save_path
    ):        
        self.idx = 0
        self.data_save_path = save_path
        pathlib.Path(self.data_save_path).mkdir(parents=True, exist_ok=True)

    def record_img(self, data):
        # self.idx += 1
        data = (
            255.0 * data.astype(np.float32) / data.max()
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(
                self.data_save_path,
                "{}_{}.jpg".format(self.idx, time.time())
            ), data
        )

    def record(self, data):
        self.idx += 1
        with open(
            os.path.join(
                self.data_save_path,
                "{}_{}.npy".format(self.idx, time.time())
            ), "wb"
        ) as f:
            np.save(
                f, data
            )

class StateLogger():
    def __init__(self, data_example, duration, frequency, data_save_name):
        self.dtype = data_example.dtype
        self.maxlength = int(duration * frequency * 2.0)

        self.data_i = 0

        self.shape = data_example.shape + (self.maxlength,)

        self.timestamp = np.zeros(self.maxlength, dtype=np.float64)
        self.data = np.zeros(self.shape, self.dtype)

        self.data_save_name = data_save_name
       
    def record(self, data):
        self.timestamp[self.data_i] = time.time()
        self.data[...,self.data_i] = data
        self.data_i += 1

    def write(self):
        np.savez(self.data_save_name, self.timestamp[...,0:self.data_i], self.data[...,0:self.data_i])
        print("Saved as %s" % self.data_save_name)
        

# class Information:
#   def __init__(self, name):
#     self.name = name
#     self.data = None
#     self.timestamp = 0
#     self.save_bool = False
#     self.data_index = 0
#     self.dirname = name+"_dir"
#     if not os.path.exists(self.dirname):
#       os.makedirs(self.dirname)

# class InformationSaver:
#   def __init__(self, control_freq=20):
#     self.control_freq = control_freq
#     self.continue_thread = False
#     self.info_dict = dict() # name of information corresponding to index
  
#   def register_information_class(self, info_name):
#     self.info_dict[info_name] = Information(info_name)

#   def register_information(self, info_name, information):
#     if (info_name in self.info_dict):
#         self.info_dict[info_name].data = information
#         self.info_dict[info_name].save_bool = True
#         self.info_dict[info_name].timestamp = time.time()
#         # data index incremented in saving
#     else:
#         raise RuntimeError("Information name not registered")
        
#   def start_thread(self):
#     self.thread = threading.Thread(target=self.main_function)
#     self.continue_thread = True
#     self.thread.start()

#   def stop_thread(self):
#     self.continue_thread = False
#     self.thread.join()
#     print("Information Saver thread joined")

#   def main_function(self):
#     count = 0
#     thread_start_time = time.time()
#     while self.continue_thread:
#       # Loop through all of the information to save
#       start_time = time.time()
#       for info in self.info_dict.values():
#         if info.save_bool is True:
#           filename = os.path.join(info.dirname, 
#                         "data_%03d_%.4f.npz" % (info.data_index, info.timestamp))
#           np.savez(filename, info.data)
#           info.save_bool = False
#           info.data_index += 1

#       # Running at certain rate?
#       count += 1
#       t1 = time.time()
#       time.sleep(max(0, (1 / self.control_freq) - (t1 -start_time)))
