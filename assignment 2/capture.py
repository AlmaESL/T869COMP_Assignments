import cv2
import time #import time 

#import fps calculation and print to frame window from fps file
from fps import calculate_fps
from fps import write_to_frame
#import canny edge detection from canny file
from canny import canny_edge_detection
#import the ransac functionality from the ransac python file
from ransac import ransac
from ransac import draw_ransac_line

def main():
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
        
        #calculate frames per second 
        current_frame_time = time.time()
    
        #compute fps
        fps = calculate_fps(prev_frame_time, current_frame_time)
        #update previous frame time to current frame time
        prev_frame_time = current_frame_time

        #print to frames window converted to string format
        write_to_frame(frame, fps)

        edges, edges_array = canny_edge_detection(frame, 70, 200)

        #perform ransac algorithm on edge pixels 
        best_model, best_inliers = ransac(edges_array, 10000, 10000, 3)

        if best_model is not None:
            draw_ransac_line(frame, best_model, frame.shape[1])
    
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

#run main from command line
if __name__ == "__main__":
    main()
       