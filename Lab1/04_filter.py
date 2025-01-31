import cv2
import numpy as np

image = cv2.imread('lena.tif')

if image is None:
    print('Could not read image')

filter1 = np.array([[0, -2, 0],
                    [-2, 8, -2],
                    [0, -2, 0]])

filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=filter1)

cv2.imwrite('filtered_image.tif', filtered_image)