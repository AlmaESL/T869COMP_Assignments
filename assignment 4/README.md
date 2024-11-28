Task 1
doc_scanner contains the code for task1, runs on live-video 

Task 2
rectification.py is for live video stream, processes every 2'nd frame to save cpu 
test.py runs rectification on still images

helper functions 
fps.py: computes and displays fps for live-video
canny.py: implements line detection with cv2.canny
Hough.py: implements finding and prniting the 4 most prominent hough lines from edges 
intersection.py: computes line intersection points