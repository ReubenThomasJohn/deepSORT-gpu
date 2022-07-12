'''
A simple script to check if our object-detector is working well...
'''
from deep_sort.yoloV5 import YOLO_Fast
import cv2

def main():
    vid = cv2.VideoCapture('./Single-Multiple-Custom-Object-Detection-and-Tracking/data/video/MOT16-13-raw.mp4')
    ret, frame = vid.read()
    print(ret)
    image_height, image_width = frame.shape[0], frame.shape[1]
    print(image_height, image_width)
    frame = cv2.resize(frame, (640, 640))
    yolo = YOLO_Fast(sc_thresh=.5, nms_thresh=.45, cnf_thresh=.45, model='./Single-Multiple-Custom-Object-Detection-and-Tracking/deep_sort/onnx_models/yolov5s.onnx')
    _, _, classes, nums= yolo.object_detection(frame, visualise = True)

    assert nums == len(classes[0])
    cv2.imshow('Output', yolo.image)
    cv2.imshow('Output', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()

