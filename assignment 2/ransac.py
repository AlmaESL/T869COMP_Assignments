import numpy as np
import cv2

#function implmenting the RANSAC algorithm to detect a line
def ransac(edge_pixels, n_iterations = 100, neighborhood_size = 100, sample_size = 1):

    best_model = None
    best_inliers = []
    max_inliers = 0

    #consider only every k'th pixel coordinaete in edge pixels coordinates array
    subsampled_pixels = edge_pixels[::sample_size]

    #repeat the ransac algorithm number of iteration times 
    for i in range(n_iterations):

        #sample 2 random points
        random_points = subsampled_pixels[np.random.choice(subsampled_pixels.shape[0], 2, replace=False)]
        (x1, y1), (x2, y2) = random_points

        #avoid division by 0 
        if x1 == x2:
            continue

        #compute line between the 2 points y = kx + m
        k = (y2 - y1) / (x2 - x1)
        m = y1 - k * x1

        #compute the distance to all edge pixels from the line
        distances = np.abs(k * subsampled_pixels[:, 0] - subsampled_pixels[:, 1] + m) / np.sqrt(k**2 + 1)

        #compute number of inliers  ie pixels within the given neighborhood of the line
        inliers = subsampled_pixels[distances < neighborhood_size]

        #update the best model if the number of inliers is higher than previous best -> we want to cover as many pixels 
        #as possible from the line 
        if len(inliers) > max_inliers:
            best_model = (k, m)
            best_inliers = inliers
            max_inliers = len(inliers)

    return best_model, best_inliers



def draw_ransac_line(frame, model, width):

    #slope and intersection of the ransac model (line)
    slope, intersection = model

    #find start and end point for the line of the ransac model, limited to the frame width
    x_start, x_end = 0, width
    y_start = int(slope * x_start + intersection)
    y_end = int(slope * x_end + intersection)

    #draw the line on the frame
    cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
