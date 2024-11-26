import numpy as np
import cv2

def canny_edge_detection(frame, threshold1 = 70, threshold2 = 200):
     
     #find edges with canny edge detection
     edges = cv2.Canny(frame, threshold1, threshold2)

     #take all x and y coordinates of all edge pixels and put them in an array - edges are binary, 1 by canny 
     edges_array = np.column_stack(np.where(edges > 0))
     edges_array = edges_array[:, ::-1] #swap columns to get x,y 

     return edges, edges_array