import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

#def median_blur(img, kernal_shape):
    


if __name__=="__main__":
  pipeline = rs.pipeline()
  ctx = rs.context()
  device = ctx.query_devices()[0]
  
  # enable stream
  config = rs.config()
  config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 320, 180, rs.format.bgr8, 30)

  # start stream
  profile = pipeline.start(config)
  depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

  # set up histogram
  bins = 100
  fig,ax = plt.subplots()
  ax.set_title('distribution of depth values in frame')
  ax.set_xlabel('depth intensity')
  ax.set_ylabel('percentage')
  lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), label='intensity')
  ax.set_ylim(0,1)
  plt.ion()
  plt.show()
  numPixels = 64*64

  # collect frame
  frames_collection = np.zeros((0,64,64))
  counter = 0
  try:
    while True: #counter < 10:
      frames = pipeline.wait_for_frames()
      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()
      if not depth_frame:
        break

      depth_image = np.asanyarray(depth_frame.get_data())
      color_image = np.asanyarray(color_frame.get_data())
      depth_dim = depth_image.shape
      # let's apply filtering here as a test
      horizontal_clip = int(depth_dim[0]/10)
      depth_image = cv2.resize(depth_image[:,2*horizontal_clip:], dsize=(64,64), interpolation=cv2.INTER_NEAREST)

      depth_image = cv2.medianBlur(depth_image, 5)
      # scaling for network
      frame_meters = depth_image * depth_scale
      
      # regard distance = 0 as distance measure failure and default to max distance
      frame_meters[frame_meters < 1e-3] = +np.inf

      #frame_meters = cv2.medianBlur(frame_meters, 3)

      frame_meters = np.clip(frame_meters, a_min=0.3, a_max=7)
      frame_meters = np.sqrt(np.log(frame_meters+1))
     
    

      # add this frame to the frames
      new_axis_frame = np.expand_dims(frame_meters, axis=0)
      frames_collection = np.vstack((new_axis_frame, frames_collection)) 

      frame = (255.0 * frame_meters.astype(np.float32) / frame_meters.max()).astype(np.uint8)
    

      # show frame
      #cv2.namedWindow('RealSense',cv2.WINDOW_AUTOSIZE)
      cv2.imwrite("Depth_frame/%03d.jpg" % counter, frame)
      #cv2.imshow('RealSense', frame)
      #cv2.waitKey(1)
      # obtain histogram of frame values
      frame_values = frame.flatten()
      histogram = cv2.calcHist([frame], [0], None, [bins], [0,255]) / numPixels
      lineGray.set_ydata(histogram)
      fig.canvas.draw()
      counter = counter + 1
  finally:
    pipeline.stop()
    cv2.destroyAllWindows() 

    mean_frame = np.mean(frames_collection,axis=0)
    print(mean_frame)
    plt.plot(mean_frame.flatten(),np.linspace(0,64*64,64*64))

    
