import cv2
import time #import time 
import numpy as np

#get camera source
cap = cv2.VideoCapture(0) # 1 for iVCam

#variables for previous frame and current frame 
prev_frame_time = 0 
current_frame_time = 0

while(True):

    #get the tick count from the start of processing of each frame...
    start_count = cv2.getTickCount()

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

    #minMaxLoc to locate brightest (and darkest) pixel in the grey scale video 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

    # #initialize variables for brightest pixel and its location
    maximum_value = 0
    maximum_location = (0, 0)

    # #use nested for-loop going through pixel by pixel to find the brightest pixel
    for i in range(0, gray.shape[0]):
        for j in range(0, gray.shape[1]):
            if gray[i, j] > maximum_value:
                maximum_value = gray[i, j]
                maximum_location = (i, j)

    #mark the brightest pixel with a circle - using nested for loop 
    cv2.circle(gray, maximum_location, 10, (0, 0, 255), 2)

    #mark the brightest pixel with a circle - using minMaxLoc
    cv2.circle(gray, max_loc, 10, (0, 0, 255), 2)

    #compute and find the reddest pixel in each frame defined by the HSV color space with hue=red, saturation = 100 
    # and value = 100
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert BGR to HSV
    lower_red = np.array([0, 150, 150])
    upper_red = np.array([0, 200, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red) #bandpass filter on specified color range 
   
    #find the reddest point
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red_mask)

    #mark the reddest pixel with a sqaure
    cv2.rectangle(frame, max_loc, (max_loc[0] + 10, max_loc[1] + 10), (0, 0, 0), 2)

    #print to frames window converted to string format
    cv2.putText(gray, 'FPS: ' + str(int(fps)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    # #FPS for color image 
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    #show frames
    cv2.imshow('frame', gray) #for grey scale
    cv2.imshow('frame', frame) #for color

    #...to the end of processing of each frame 
    end_count = cv2.getTickCount()

    #latency in seconds -> the number of clock ticks between start and end of processing for each frame
    latency = ((end_count - start_count) / cv2.getTickFrequency())*1000 #tickFrequency = the number of clock ticks per second
    print("latency: ", latency, " ms")

    #update frames and exit on q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#close camera source
cap.release()
cv2.destroyAllWindows()