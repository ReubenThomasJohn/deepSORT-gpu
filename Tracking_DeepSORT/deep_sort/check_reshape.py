import numpy as np
import cv2

arr = np.eye(6,3)

print(arr)

reshaped_arr = arr.reshape(1, 3, 6)

print(reshaped_arr)
