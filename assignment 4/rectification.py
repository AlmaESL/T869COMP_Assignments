import cv2
import numpy as np
import time

from fps import calculate_fps, write_to_frame
from canny import canny_edge_detection
from Hough import get_Hough_lines, draw_hough_lines
from intersections import compute_intersections

def rectify_image(frame, intersections, width_scale, height_scale):

    #compute rectification only if there are 4 intersections
    if len(intersections) >= 4:

        #sort the intersections ie corners
        intersections = sorted(intersections, key=lambda x: x[0] + x[1])
        top_left, top_right, bottom_right, bottom_left = intersections[:4]

        print("corners: ", top_left, top_right, bottom_right, bottom_left)

        #add the corner points to an array
        top_left = np.array(top_left)
        top_right = np.array(top_right)
        bottom_right = np.array(bottom_right)
        bottom_left = np.array(bottom_left)


        #specify the corners of the source image
        pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])
        #print the corners of the source image

        width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)) * width_scale
        height = max(np.linalg.norm(top_left - bottom_left), np.linalg.norm(top_right - bottom_right)) * height_scale

        #specify the corners of the destination image
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        #compute the transformation matrix
        matrix, _ = cv2.findHomography(pts1, pts2)
    
        
        #apply the transformation
        result = cv2.warpPerspective(frame, matrix, (int(width), int(height)))

        return result
    return frame

cap = cv2.VideoCapture(1)

prev_frame_time = 0 
current_frame_time = 0

#precess every 2nd frame to prevent cpu crash 
frame_count = 0
process_every_nth_frame = 3

while True:
    #read frames from video 
    ret, frame = cap.read()
    
    #compute fps
    current_frame_time = time.time()
    fps = calculate_fps(prev_frame_time, current_frame_time)
    prev_frame_time = current_frame_time
    write_to_frame(frame, fps)

    #precess every 2nd frame to prevent cpu crashing
    if frame_count % process_every_nth_frame == 0:

        #work on binary images for edge detection and hough transform
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #apply canny edge detection
        edges = canny_edge_detection(gray, 100, 200, apertureSize=3)

        #apply hough transform to get hough lines 
        lines = get_Hough_lines(edges, 1, np.pi / 180, 130)

        #extract and draw top 4 prominent hough lines
        if lines is not None:

            #sort the lines after prominence
            lines = sorted(lines, key=lambda x: abs(x[0][0]), reverse=True)

            #extract the top 4 most poriment lines
            most_4_lines = lines[:4]

            #draw the lines
            for line in most_4_lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #compute intersections
            intersections = compute_intersections(most_4_lines)

            #rectify image
            rectified_image = rectify_image(frame.copy(), intersections, 0.5, 0.5)

            #display rectified image
            cv2.imshow('Rectified Image', rectified_image)

        cv2.imshow('Edges', edges)
    cv2.imshow('Edges and Lines', frame)

    #increment frame count
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
