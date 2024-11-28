import cv2
import numpy as np

from fps import calculate_fps, write_to_frame
from canny import canny_edge_detection
from Hough import get_Hough_lines, draw_hough_lines
from intersections import compute_intersections

def rectify_image(source_image, target_image):
    
    #work on binary images for edge detection and hough transform
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    #canny edge detection
    edges = canny_edge_detection(gray, 100, 200, apertureSize=3)

    #hough transform
    lines = get_Hough_lines(edges, 1, np.pi / 180, 130)

    #extract and draw top 4 prominent hough lines
    if lines is not None:
        lines = sorted(lines, key=lambda x: abs(x[0][0]), reverse=True)
        most_4_lines = lines[:4]

        # for line in most_4_lines:
        #     rho, theta = line[0]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))

        #     cv2.line(source_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        #compute intersections
        intersections = compute_intersections(most_4_lines)

        #only do rectification if there are 4 intersections
        if len(intersections) >= 4:

            #sort intersections
            intersections = sorted(intersections, key=lambda x: x[0] + x[1])

            #compute corners for rectification, only use top 4 lines
            top_left, top_right, bottom_right, bottom_left = intersections[:4]

            print("corners: ", top_left, top_right, bottom_right, bottom_left)

            #add the corner points to an array
            top_left = np.array(top_left)
            top_right = np.array(top_right)
            bottom_right = np.array(bottom_right)
            bottom_left = np.array(bottom_left)

            #specify the corners of the source image
            pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

            #get the height and width of the target image
            height, width = target_image.shape[:2]
            
            #specify the corners of the target image
            pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

            #compute the transformation matrix
            matrix, _ = cv2.findHomography(pts1, pts2)
            
            #warp the source image
            result = cv2.warpPerspective(source_image, matrix, (width, height))

            return result

    return source_image


#load source and target images
source_image = cv2.imread('C:/Users/almal/Desktop/T869COMP/assignments/assignment 4/20241128202406.jpg')
target_image = cv2.imread('C:/Users/almal/Desktop/T869COMP/assignments/assignment 4/20241128202352.jpg')

#rectify the source image to match the target image
rectified_image = rectify_image(source_image, target_image)

#source image display 
cv2.imshow('Source Image', source_image)
#target image display
cv2.imshow('Target Image', target_image)
# display rectified image
cv2.imshow('Rectified Image', rectified_image)

cv2.waitKey(0)
cv2.destroyAllWindows()