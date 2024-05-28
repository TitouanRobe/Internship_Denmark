from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

model = YOLO("yolov8m.pt")
dict_classes = model.model.names

import yaml

data = {'train' :  './Road_Sign_Detection/train/images',
        'val' :  './Road_Sign_Detection/valid/images',
        'test' :  './Road_Sign_Detection/test/images',
        'nc': 73,
        'names': ['-', 'Barrier Ahead', 'Cattle', 'Caution', 'Cycle Crossing', 'Dangerous Dip', 'Eating Place', 'Falling Rocks', 'Ferry', 'First Aid Post', 'Give Way', 'Horn Prohibited', 'Hospital', 'Hump', 'Left Hair Pin Bend', 'Left Reverse Bend', 'Left hand curve', 'Light Refreshment', 'Men at Work', 'Narrow Bridge', 'Narrow road ahead', 'No Parking', 'No Stopping', 'No Thorough Road', 'No Thorough SideRoad', 'Parking Lot Cars', 'Parking Lot Cycle', 'Parking Lot Scooter and MotorCycle', 'Parking This side', 'Pedestrian Crossing', 'Pedestrian Prohibited', 'Petrol Pump- Gas Station', 'Public Telephone', 'Resting Place', 'Right Hair Pin Bend', 'Right Hand Curve', 'Right Reverse Bend', 'Road Wideness Ahead', 'Round About', 'School Ahead', 'Slippery Road', 'Speed Limit -10-', 'Speed Limit -100-', 'Speed Limit -110-', 'Speed Limit -120-', 'Speed Limit -130-', 'Speed Limit -140-', 'Speed Limit -150-', 'Speed Limit -160-', 'Speed Limit -20-', 'Speed Limit -25-', 'Speed Limit -35-', 'Speed Limit -45-', 'Speed Limit -48-', 'Speed Limit -5-', 'Speed Limit -50-', 'Speed Limit -55-', 'Speed Limit -60-', 'Speed Limit -65-', 'Speed Limit -70-', 'Speed Limit -75-', 'Speed Limit -8-', 'Speed Limit -80-', 'Speed Limit -90-', 'Speed Limit 3', 'Speed Limit 30', 'Speed limit -15-', 'Speed limit -40-', 'Steep Ascent', 'Steep Desecnt', 'Stop', 'Straight Prohibitor No Entry', 'walking']
        }

with open('./Road_Sign_Detection/data.yaml', 'w') as f:
    yaml.dump(data, f)

# read the content in .yaml file
with open('./Road_Sign_Detection/data.yaml', 'r') as f:
    road_yaml = yaml.safe_load(f)
    

print(len(model.names))

model.train(data="./data.yaml", epochs=50, batch=8)