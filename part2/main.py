import cv2
import time #import time 
import numpy as np

#import fps calculation and print to frame window from fps file
from fps import calculate_fps
from fps import write_to_frame

#get coco class names 
from cocoNames import coco_names


# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0, 0, 255)

#load the yolo model 
# model_path = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/part1/yolov4-tiny.cfg"
# model_weights = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/part1/yolov4-tiny.weights"

model_path = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/part1/yolov4.cfg"
model_weights = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/part1/yolov4.weights"

onnx_file = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/yolov5s.onnx"
# model = cv2.dnn.readNet(model_path, model_weights)
model = cv2.dnn.readNet(onnx_file)
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

#thresholds 
conf_threshold = 0.5
nms_threshold = 0.4
score_threshold = 0.5

def draw_label(im, label, x, y):

    #get text size
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    dim, baseline = text_size[0], text_size[1]

    #create a rectangle 
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED)

    #display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)


def pre_process(input_image, net):
      # Create a 4D blob from a frame.
      blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
 
      # Sets the input to the network.
      net.setInput(blob)
 
      # Run the forward pass to get output of the output layers.
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs    


def post_process(input_image, outputs):
      # Lists to hold respective values while unwrapping.
      class_ids = []
      confidences = []
      boxes = []
      # Rows.
      rows = outputs[0].shape[1]
      image_height, image_width = input_image.shape[:2]
      # Resizing factor.
      x_factor = image_width / INPUT_WIDTH
      y_factor =  image_height / INPUT_HEIGHT
      # Iterate through detections.
      for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

        # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
      indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, 3*THICKNESS)
            # Class label.                      
            label = "{}:{:.2f}".format(coco_names[class_ids[i]], confidences[i])             
            # Draw label.             
            draw_label(input_image, label, left, top)
      return input_image                

    
#get camera source
cap = cv2.VideoCapture(1) # 1 for iVCam  

#variables for previous frame and current frame 
prev_frame_time = 0 
current_frame_time = 0

while(True):    

    #read frames from source
    ret, frame = cap.read()
        
    #calculate frames per second 
    current_frame_time = time.time()
    
    #compute fps
    fps = calculate_fps(prev_frame_time, current_frame_time)
    #update previous frame time to current frame time
    prev_frame_time = current_frame_time
    #print to frames window converted to string format
    write_to_frame(frame, fps)

    #modelWeights = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/yolov5s.onnx"
    modelWeights = "C:/Users/almal/Desktop/T869COMP/assignments/assignment 3/best (1).onnx"

    net = cv2.dnn.readNet(modelWeights)
    # Process image.
    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)

    cv2.imshow("YOLO Detection", img)
     
    #update frames and exit on q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#close camera source
cap.release()
cv2.destroyAllWindows()