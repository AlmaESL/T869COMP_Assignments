import numpy as np
import cv2

def canny_edge_detection(frame, threshold1, threshold2, apertureSize):
     edges = cv2.Canny(frame, threshold1, threshold2, apertureSize=apertureSize)
     return edges