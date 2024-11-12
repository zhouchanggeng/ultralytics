'''
Author: zhouchanggeng
Date: 2024-11-12 10:46:56
LastEditTime: 2024-11-12 11:34:22
LastEditors: zhouchanggeng
Description: 
FilePath: /ultralytics/scripts/pose.py
Copyright (c) 2024 jiaxun.com, Inc. All Rights Reserved
'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from ultralytics import YOLO
# from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/workspace/Research/ultralytics/checkpoints/pose/yolo11l-pose.pt")
image_path = "/root/zcg/workspace/Research/ultralytics/ultralytics/assets/bus.jpg"

results = model(image_path)

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk