import cv2
import numpy as np

image = cv2.imread('lena.tif')

height, width = image.shape[:2]

print('Image size:')
print(f'Width: {width}')
print(f'Height: {height}')

cv2.imwrite('lena2.tif', image)

# blur & sharpen

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

# filter

image = cv2.imread('lena.tif')

if image is None:
    print('Could not read image')

filter1 = np.array([[0, -2, 0],
                    [-2, 8, -2],
                    [0, -2, 0]])

filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=filter1)

cv2.imwrite('filtered_image.tif', filtered_image)

#rotation

def rotate_image(image, angle, img_name):
    img_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    cv2.imwrite(img_name, result)


image = cv2.imread('lena.tif')

rotate_image(image, 40, 'rotated_image1.tif') #counterlockwise
rotate_image(image, -120, 'rotated_image2.tif') #clockwise

#rectangularCrop

def crop_rectangle(image, x, y, width, height):
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError('Dimensions out of bounds.')

    cropped_img = image[y:y + height, x:x + width]

    return cropped_img


image = cv2.imread('lena.tif')
cropped_image = crop_rectangle(image, 0, 0, 300, 150)

cv2.imwrite('cropped_image.tif', cropped_image)