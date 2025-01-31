import cv2


def crop_rectangle(image, x, y, width, height):
    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError('Dimensions out of bounds.')

    cropped_img = image[y:y + height, x:x + width]

    return cropped_img


image = cv2.imread('lena.tif')
cropped_image = crop_rectangle(image, 0, 0, 300, 150)

cv2.imwrite('cropped_image.tif', cropped_image)
