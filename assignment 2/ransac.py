import numpy as np

#function implmenting the RANSAC algorithm to detect a line
def ransac(edge_pixels, n_iterations = 100, neighborhood_size = 100, sample_size = 1):

    best_model = None
    best_inliers = []
    max_inliers = 0

    #consider only every k'th pixel coordinaete in edge pixels coordinates array
    subsampled_pixels = edge_pixels[::sample_size]

    #repeat the ransac algorithm number of iteration times 
    for i in range(n_iterations):

        #sample 2 random pixel coordinates from the edge pixels
        random_points = subsampled_pixels[np.random.choice(subsampled_pixels.shape[0], 2, replace=False)]
        (x1, y1), (x2, y2) = random_points

        #avoid division by 0 
        if x1 == x2:
            continue

        #compute line between the 2 pixel coordinates y = kx + m
        k = (y2 - y1) / (x2 - x1)
        m = y1 - k * x1

        #compute the distance to all edge pixel coordinates from the line
        distances = np.abs(k * subsampled_pixels[:, 0] - subsampled_pixels[:, 1] + m) / np.sqrt(k**2 + 1)

        #compute number of inliers ie pixel coordinates within the given neighborhood of the line
        inliers = subsampled_pixels[distances < neighborhood_size]

        #update the best model if the number of inliers is higher than previous best -> we want to cover as many pixels 
        #as possible with the line 
        if len(inliers) > max_inliers:
            best_model = (k, m)
            best_inliers = inliers
            max_inliers = len(inliers)

    return best_model, best_inliers
