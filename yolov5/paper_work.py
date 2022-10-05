import cv2
import torch

import matplotlib.pyplot as plt

from PIL import Image
from roboflow import Roboflow

from utils.plots import Annotator, colors

import pyrealsense2 as rs
import numpy as np

# # roboflow
# rf = Roboflow(api_key="wX2vbQQYJZ6AHTgk2j3l")
# project = rf.workspace().project("test-6omc6")
# dataset = project.version(1).download("yolov5")


# Custom Model Load 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/kimi/work/yolov5/runs/train/exp11/weights/best.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Original model 

# # Images detection 
# img1 = Image.open('iphone_data/pencil.jpeg')  # PIL image
# img1 = img1.resize((640, 480), Image.NEAREST)

# img2 = Image.open('iphone_data/multiple_clip_white.jpeg')  # PIL image
# img2 = img2.resize((640, 480), Image.NEAREST)

# # img_resize_lanczos.save('data/dst/sample_pillow_resize_lanczos.jpg')

# # img2 = cv2.imread('test2.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
# imgs = [img1, img2]  # batch of images

# # # Inference
# results = model(imgs)  # includes NMS

# results.show()
# print(results.pandas().xyxy[0])




# # get pos in image
# xmin = results.xyxy[0][0][0]
# xmin = xmin.to('cpu')
# ymin = results.xyxy[0][0][1]
# ymin = ymin.to('cpu')
# xmax = results.xyxy[0][0][2]
# xmax = xmax.to('cpu')
# ymax = results.xyxy[0][0][3]
# ymax = ymax.to('cpu')

# # get center pos
# ave_x = (xmin+xmax)/2
# ave_x = ave_x.numpy()
# ave_y = (ymin+ymax)/2
# ave_y = ave_y.numpy()

# img3 = results.render()
# img3 = np.asarray(img3)

# # cv2.imshow('',img3[0])
# cv2.waitKey(0)


# Webcam detection
cap = cv2.VideoCapture(4)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(cap.isOpened()):
    ret, frame = cap.read()

    results = model(frame, size=640)
    
    # results.print()
    img = results.render()
    img = np.asarray(img)
    
    cv2.imshow('Office Item recognition', img[0])

    key=cv2.waitKey(1)
    if not ret and key==27:
    	break
    
cap.release()
cv2.destroyAllWindows()


'''
# Intel realsense connect
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        
        results = model(color_image, size=640)
        results.print()
        img = results.render()
        img = np.asarray(img)
        # results.print()

        if len(results.pandas().xyxy[0].index) == 0:
 	       	pass
        else:
        	# get pos in image
            xmin = results.xyxy[0][0][0]
            xmin = xmin.to('cpu')
            ymin = results.xyxy[0][0][1]
            ymin = ymin.to('cpu')
            xmax = results.xyxy[0][0][2]
            xmax = xmax.to('cpu')
            ymax = results.xyxy[0][0][3]
            ymax = ymax.to('cpu')
	        
            # get center pos
            ave_x = (xmin+xmax)/2
            ave_x = ave_x.numpy()
            ave_y = (ymin+ymax)/2
            ave_y = ave_y.numpy()

            dist = depth_frame.get_distance(ave_x, ave_y)
            print("dist: ", dist)
            depth_pixel = [ave_x, ave_y]
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)    
            print("depth point: ", depth_point)


        
        
        


        # # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', img[0])
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()

'''