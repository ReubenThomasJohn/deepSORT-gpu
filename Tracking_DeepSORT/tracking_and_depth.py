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

from deep_sort.yoloV5_dnn import YOLO_DNN

class DeepSORT():

    def __init__(self, class_names_file='Tracking_DeepSORT/data/labels/coco.names', 
    yolo_model='./Tracking_DeepSORT/deep_sort/onnx_models/yolov5m.onnx',
    model_filename='Tracking_DeepSORT/model_data/mars-small128.pb', visualize=True):

        self.class_names = [c.strip() for c in open(os.path.abspath(class_names_file)).readlines()]
        self.yolo = YOLO_DNN(sc_thresh=.5, nms_thresh=.45, cnf_thresh=.45, model=yolo_model)

        # cost-related hyperparameters
        self.max_cosine_distance = 0.5
        self.nn_budget = None
        self.nms_max_overlap = 0.8

        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(self.metric)

        from collections import deque
        self.pts = [deque(maxlen=30) for _ in range(1000)] 

        self.viz = visualize

        self.counter = []

        if self.viz==True:
            # These lines are simply for visualization purposes. 
            cmap = plt.get_cmap('tab20b')
            self.colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    def do_object_detection(self, img_in):
        t1 = time.time()
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
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Kalman Filter step
        self.tracker.predict()
        self.tracker.update(detections)

        if self.viz == True:
            current_count = int(0)

            # looping over each tracked obstacle
            center_arr = []
            for track in self.tracker.tracks:
                # if the track has not been confirmed yet, has not been updated for more than 1 frame, skip it.
                if not track.is_confirmed() or track.time_since_update >1:
                    continue
                # convert the bbox from tlwh to tlbr
                bbox = track.to_tlbr()
                # extract the tracks class using the get_class() method
                class_name = track.get_class()
                color = self.colors[int(track.track_id) % len(self.colors)]
                color = [i * 255 for i in color]

                # visualization
                cv2.rectangle(img_in, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
                cv2.rectangle(img_in, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(img_in, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                center_arr.append(center)
                self.pts[track.track_id].append(center)

                for j in range(1, len(self.pts[track.track_id])):
                    if self.pts[track.track_id][j-1] is None or self.pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*2)
                    cv2.line(img_in, (self.pts[track.track_id][j-1]), (self.pts[track.track_id][j]), color, thickness)

                height, width, _ = img_in.shape
                cv2.line(img_in, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
                cv2.line(img_in, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)

                center_y = int(((bbox[1])+(bbox[3]))/2)

                if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):
                    if class_name == 'car' or class_name == 'truck':
                        self.counter.append(int(track.track_id))
                        current_count += 1

            total_count = len(set(self.counter))
            cv2.putText(img_in, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
            cv2.putText(img_in, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)


            fps = 1./(time.time()-t1)
            cv2.putText(img_in, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)

        return img_in, center_arr


vid = cv2.VideoCapture('Tracking_DeepSORT/data/video/MOT16-13-raw.webm')
deepsort  = DeepSORT(class_names_file='Tracking_DeepSORT/data/labels/coco.names', 
    yolo_model='./Tracking_DeepSORT/deep_sort/onnx_models/yolov5m.onnx',
    model_filename='Tracking_DeepSORT/model_data/mars-small128.pb', visualize=True)
while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    
    img = cv2.resize(img, (640,640)) # We resize to (640, 640) since YOLOv5 was trained on this shape
    img_copy = np.copy(img)
    img_out, c = deepsort.do_object_detection(img_copy)
    cv2.namedWindow('output')
    # cv2.resizeWindow('output', 1424, 1068)
    cv2.imshow('output', img_out)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()



    