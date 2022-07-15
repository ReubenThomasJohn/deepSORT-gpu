# Use python 3.8.10

# Import the required libraries
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# This is a function that takes our bounding boxes, and converts them into a list data type. 
from helpers.convert_boxes import convert_boxes

# Here are the core deep-sort functions. 

# Pre-processing contains the code for the non-maxima suppresion
from deep_sort import preprocessing
# nn-matching contains all the code to implement the cost function to associate tracks
from deep_sort import nn_matching
# Here is a class to hold all the information in a single detection from yolo()
from deep_sort.detection import Detection
# Here is the Tracker class to hold all information regarding a tracked object. This is the key class - make sure to open up the files and understand
# the methods implemented
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# we import the yolo class we created in the previous project
from deep_sort.yoloV5 import YOLO_Fast

class DeepSORT:

    def __init__(self, class_names_file='Tracking_DeepSORT/data/labels/coco.names', 
    yolo_model='./Tracking_DeepSORT/deep_sort/onnx_models/yolov5m.onnx',
    model_filename='Tracking_DeepSORT/model_data/mars-small128.pb')

        self.class_names = [c.strip() for c in open(os.path.abspath('Tracking_DeepSORT/data/labels/coco.names')).readlines()]
        self.yolo = YOLO_Fast(sc_thresh=.5, nms_thresh=.45, cnf_thresh=.45, model=yolo_model)

        # cost-related hyperparameters
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 0.8

        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)

        from collections import deque
        self.pts = [deque(maxlen=30) for _ in range(1000)] 

    def do_object_detection(self, img_in, visualise=False):
        boxes, scores, classes, nums = self.yolo.object_detection(img_in, visualise = False)
        boxes = np.array([boxes]) 
        classes = classes[0]

        names = []
        for i in range(len(classes)):
            names.append(self.class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img_in, boxes[0])
        # we extract the features corresponding to each bounding box detected. 
        features = self.encoder(img_in, converted_boxes)

        # this is a list of Detection() objects, that contain information for the bbox, score, class, and feature corresponding to each detection
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                    zip(converted_boxes, scores[0], names, features)]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        # we perform NMS on all the detections, and extract indices of only the boxes to be kept. That is, the overlapping boxes are removed
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Kalman Filter step


    