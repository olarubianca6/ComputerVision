import cv2
import numpy as np


def blur_image(image, kernel, image_name):
    blurred_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imwrite(image_name, blurred_image)


def sharpen_image(image, kernel, image_name):
    sharpened_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imwrite(image_name, sharpened_image)


image = cv2.imread('lena.tif')

blur_filter1 = np.ones((5, 5)) / 25

sharpen_filter1 = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

blur_filter2 = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16

sharpen_filter2 = np.array([[0, -1,  0],
                            [-1,  5, -1],
                            [0, -1,  0]])

blur_image(image, blur_filter1, 'blurred_image1.tif')
blur_image(image, blur_filter2, 'blurred_image2.tif')

sharpen_image(image, sharpen_filter1, 'sharpened_image1.tif')
sharpen_image(image, sharpen_filter2, 'sharpened_image2.tif')
