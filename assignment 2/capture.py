import cv2
import time #import time 
import numpy as np

#import the ransac function from the ransac python file
from ransac import ransac

#get camera source
cap = cv2.VideoCapture(0) # 1 for iVCam
    
#limit the resolution to 640x480 for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    

#variables for previous frame and current frame 
prev_frame_time = 0 
current_frame_time = 0

while(True):

    #read frames from source
    ret, frame = cap.read()

    #convert to grey scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #calculate frames per second 
    current_frame_time = time.time()
    #process time for one frame the difference between current and previous frame times
    frame_time = current_frame_time - prev_frame_time
    print("frame time: ", frame_time, " seconds") 

    fps = 1 / frame_time #per 1 second
    # print("fps: ", fps)

    #update previous frame time to current frame time
    prev_frame_time = current_frame_time

    #print to frames window converted to string format
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    #find edges with canny edge detection
    edges = cv2.Canny(frame, 70, 200)

    #take all x and y coordinates of all edge pixels and put them in an array - edges are binary, 1 by canny 
    edge_pixels = np.column_stack(np.where(edges > 0))
    edge_pixels = edge_pixels[:, ::-1] #swap columns to get x,y 

    #perform ransac algorithm on edge pixels 
    best_model, best_inliers = ransac(edge_pixels, 10000, 10000, 3)

    if best_model is not None:
        slope, intersection = best_model

        #identify start and ending point for the line of the best ransac model 
        x_start = 0
        y_start = int(slope * x_start + intersection)
        x_end = gray.shape[1]  
        y_end = int(slope * x_end + intersection)

        #draw the line on the gray frame
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
    
    #show edges frames 
    cv2.imshow('edges', edges)
     
    #show frames
    cv2.imshow('frame', frame) 

    #update frames and exit on q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#close camera source
cap.release()
cv2.destroyAllWindows()




         



       