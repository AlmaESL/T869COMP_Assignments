import cv2
import numpy as np
import time

from fps import calculate_fps, write_to_frame
from canny import canny_edge_detection
from Hough import get_Hough_lines, draw_hough_lines

prev_frame_time = 0 
current_frame_time = 0

# Capture video from the default camera
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #compute fps
    #calculate frames per second 
    current_frame_time = time.time()
    
    #compute fps
    fps = calculate_fps(prev_frame_time, current_frame_time)
    #update previous frame time to current frame time
    prev_frame_time = current_frame_time
    #print to frames window converted to string format
    write_to_frame(frame, fps)

    #canny edge detection
    edges = canny_edge_detection(gray, 100, 200, apertureSize=3)

    #hough transform 
    lines = get_Hough_lines(edges, 1, np.pi / 180, 130)
    draw_hough_lines(frame, lines, 1000)

    #show edges and lines
    cv2.imshow('Edges and Lines', frame)
    #show edges frames 
    cv2.imshow('Edges', edges)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

