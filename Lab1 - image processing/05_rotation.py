import cv2
import numpy as np


def rotate_image(image, angle, img_name):
    img_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    cv2.imwrite(img_name, result)


image = cv2.imread('lena.tif')

rotate_image(image, 40, 'rotated_image1.tif') #counterlockwise
rotate_image(image, -120, 'rotated_image2.tif') #clockwise