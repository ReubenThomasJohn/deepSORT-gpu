
from yoloV5 import YOLO_Fast
import cv2



def main():
    vid = cv2.VideoCapture('./Single-Multiple-Custom-Object-Detection-and-Tracking/data/video/MOT16-13-raw.mp4')
    ret, frame = vid.read()
    print(ret)
    image_height, image_width = frame.shape[0], frame.shape[1]
    print("Original Dims:", image_height, image_width)
    frame = cv2.resize(frame, (640, 640))
    frame_height, frame_width = frame.shape[0], frame.shape[1]
    print("Changed Dims:", frame_height, frame_width)
    yolo = YOLO_Fast(sc_thresh=.5, nms_thresh=.45, cnf_thresh=.45, model='./Single-Multiple-Custom-Object-Detection-and-Tracking/deep_sort/onnx_models/yolov5m.onnx')
    boxes, scores, classes, nums, tracking_boxes = yolo.object_detection(frame, visualise = True)
    print('boxes:', boxes)
    print(nums, len(classes[0]))
    assert nums == len(classes[0])
    # print(len(boxes))
    cv2.imshow('Output', yolo.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()