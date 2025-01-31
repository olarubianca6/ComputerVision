import numpy as np
import cv2

def colorize_grayscale(grayscale_image):
    height, width = grayscale_image.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            intensity = grayscale_image[i, j]

            if intensity < 85:
                color_image[i, j] = [intensity, 0, 255 - intensity]
            elif intensity < 170:
                color_image[i, j] = [255 - intensity, intensity, 0]
            else:
                color_image[i, j] = [255, 255 - intensity, 0]

    return color_image

grayscale_image = cv2.imread('xray.jpg', cv2.IMREAD_GRAYSCALE)

colorized_image = colorize_grayscale(grayscale_image)

#cv2.imshow('Colorized Image', colorized_image)
cv2.imwrite('xray_colorized.jpg', colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()