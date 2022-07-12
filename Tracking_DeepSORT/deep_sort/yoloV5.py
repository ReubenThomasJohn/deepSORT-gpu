import cv2
import numpy as np

'''
Notes: 
1. We are creating a YOLO() class that performs object detection, and non-maxima suppression
2. We will create it in such a way, that we can integrate use it in the upcoming
tracking project as well.
'''
class YOLO_Fast(): # outputs tlbr bounding boxes
    '''
    YOLOv5 outputs 25200 detections. Each detection is an array containing the following format
    [cx, cy, w, h, confidence, scores for each class]. Hence, each detection will be an array of len (5 + no. of classes)
    Here, (cx, cy) is the pixel location for the center of the detected bbox.
    (w, h) are the width and height of the bbox.
    confidence is the probability that YOLO thinks the detected box really has an object inside.
    class scores are the probabilities that the object detected belongs to each corresponding class.
    The bounding boxes are in the format (top, left), (bottom, right)
    '''
    
    # Constructor
    # We create  a constructor that takes in score, confidence and nms thresholds, along with which model to use. 
    def __init__(self, sc_thresh=.5, nms_thresh=.45, cnf_thresh=.45, model="./Single-Multiple-Custom-Object-Detection-and-Tracking/deep_sort/onnx_models/yolov5s.onnx"):
        # Our model (YOLOv5) architecture expects a 640px by 640px image as input
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        
        # These are the thresholds that are used to perform the object detection
        self.SCORE_THRESHOLD = sc_thresh
        self.NMS_THRESHOLD = nms_thresh
        self.CONFIDENCE_THRESHOLD = cnf_thresh
        
        # Drawing labels and rectangles
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.BLUE = (255, 178, 50)
        self.YELLOW = (0, 255, 255)
        
        # Network & Classes
        classesFile = "./Tracking_DeepSORT/data/labels/coco.names"
        self.classes = None
        # A handy way to read all the classes from a file, without needed to hardcode each one
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            # print("Classes: ", self.classes)
    
        print("Number of Classes: ", len(self.classes))
        # We use the YOLOv5m model. YOLOv5 has smaller and larger models which you can see from the Ultralytics/yolov5 github.
        # modelWeights = model # We also use a .onnx file. Converting from PyTorch to ONNX is simple, and
        # can be done using the export.py function from the Ultralytics/yolov5 github
        print('----------------------')
        print("Loaded model: ", model)
        # Loading in the model using the cv2.dnn class
        self.net = cv2.dnn.readNet(model)
        
    def pre_process(self, input_image):
        '''We create this function, that takes in the 640x640 image, converts it to a blob since this is what the 
        network from cv2.dnn requires. Then, the input is sent into the network, and one forward pass is done. Finally, the outputs 
        are retrieved using the getUnconnectedLayersNames() function.
        Read more about blobFromImage here: https://www.geeksforgeeks.org/how-opencvs-blobfromimage-works/
        '''
        self.image = input_image
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0,0,0], 1, crop=False)
        self.net.setInput(blob)

        self.outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())

    def post_process(self):
        '''
        One forward pass gives us an output as an array, that has a shape=(25200, no.of classes). That is, for each image,
        25200 predictions, per class are made. We need to extract only the useful and important information from this massive
        array. We use the post_process() function for this. 
        '''
        # Lists to hold respective values while unwrapping.
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.tracking_boxes = []
        self.classes_scores = []

        # Rows.
        rows = self.outputs[0].shape[1]

        image_height, image_width = self.image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        # Iterate through 25200 detections.
        for r in range(rows):
            row = self.outputs[0][0][r]
            confidence = row[4]

            # Discard bad detections and continue.
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]

                # Get the index of max class score.
                class_id = np.argmax(classes_scores)

                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    self.confidences.append(confidence)
                    self.class_ids.append(class_id)
                    self.classes_scores.append(classes_scores[class_id])

                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    left = int((cx - w/2) * x_factor) 
                    top = int((cy - h/2) * y_factor)
                    right = int((cx + w/2) * x_factor)
                    bottom = int((cy + h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    self.boxes.append(box)

        self.boxes = np.array(self.boxes)  # nms needs tlbr not tlwh 

    def tlwh2tlbr(self):
        '''
        A func that converts bboxes in the tlwh (top left, width height) format to the 
        tlbr format (top left, bottom right)
        '''
        tlbr_boxes = self.boxes.copy()
        try:
            tlbr_boxes[:,2:] += self.boxes[:,:2]
        except:
            tlbr_boxes = []

        return tlbr_boxes 

    def tlbr2tlwh(self):
        '''
        A function to convert tlbr to tlwh bboxes
        '''
        tlwh_boxes = self.boxes.copy()
        tlwh_boxes[:,2:] -= self.boxes[:,:2]
        return tlwh_boxes


    def non_max_suppression_fast(self):
        '''
        A quick and efficient way to perform NMS using vectorization.
        This function requires the bboxes to be in the tlbr format
        '''
        # print('Rejecting overlapping boxes...')
        self.boxes = self.tlwh2tlbr()
        # if there are no boxes, return an empty list
        if len(self.boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if self.boxes.dtype.kind == "i":
            self.boxes = self.boxes.astype("float")
        # initialize the list of picked indexes	
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = self.boxes[:,0]
        y1 = self.boxes[:,1]
        x2 = self.boxes[:,2]
        y2 = self.boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]] 

            # # Supress/hide the warning
            # np.seterr(invalid='ignore')
            
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                    np.where(overlap > self.NMS_THRESHOLD)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        self.boxes = self.boxes[pick].astype("int")
        self.boxes = self.tlbr2tlwh()
        self.class_ids = np.array(self.class_ids)[pick]
        assert len(self.boxes) == len(self.class_ids)
        self.classes_scores = np.array(self.classes_scores)[pick]
        self.picks = pick 
    
    def drawNMSBoxes(self):
        self.boxes = self.tlwh2tlbr()
        for box, pick, classId in zip(self.boxes, self.picks, self.class_ids):
            label = '%.2f' % (self.confidences[pick])
            if len(self.class_ids)>0:
                assert(classId < len(self.classes))
                labeltoDraw = '%s:%s' % (self.classes[classId], label)
    #       box = boxes[i]
            left = box[0]
            top = box[1]
            right = box[2]
            bottom = box[3]
            cv2.rectangle(self.image, (left, top), (right, bottom), self.BLUE, 3*self.THICKNESS)

            #Display the label at the top of the bounding box
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.putText(self.image, labeltoDraw, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=1)
            
            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(self.image, label, (20, 40), self.FONT_FACE, self.FONT_SCALE, self.YELLOW, 1, cv2.LINE_AA)    
        
    def object_detection(self, input_image, visualise=False):
        self.pre_process(input_image)
        self.post_process()      
        self.non_max_suppression_fast()
        if visualise == True:
            self.drawNMSBoxes()

        return self.boxes, np.array([self.classes_scores]), np.array([self.class_ids]), len(self.class_ids)