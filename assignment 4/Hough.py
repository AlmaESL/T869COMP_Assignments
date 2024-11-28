import cv2
import numpy as np

def get_Hough_lines(edges, rho, theta, threshold):
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    return lines

#draw four most prominent hough lines 
def draw_hough_lines(frame, lines, frame_width):

    if lines is not None:
        #sort the lines after prominence
        lines = sorted(lines, key=lambda x: abs(x[0][0]), reverse=True)

        #extract the top 4 most poriment lines
        most_4_lines = lines[:4]

        for line in most_4_lines:
            #convert end points from hough space to 2D euclidean space
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            #draw lines on frame, extend them to frame width 
            x1 = int(x0 + frame_width * (-b))
            y1 = int(y0 + frame_width * (a))
            x2 = int(x0 - frame_width * (-b))
            y2 = int(y0 - frame_width * (a))

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
