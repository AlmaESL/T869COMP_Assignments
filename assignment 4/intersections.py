import numpy as np

def compute_intersections(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
            b = np.array([[rho1], [rho2]])
            if np.linalg.det(A) != 0:
                x, y = np.linalg.solve(A, b)
                intersections.append((int(x[0]), int(y[0])))
    return intersections