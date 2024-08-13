import numpy as np

def rot(vec, theta):
    arr = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]]
    )
    return np.dot(arr, vec)
