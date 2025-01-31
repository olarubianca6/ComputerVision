import cv2
import numpy as np


def load_image(image_path):
    return cv2.imread(image_path)

def simple_averaging(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def weighted_average(image):
    weights = [0.299, 0.587, 0.114]
    img = np.dot(image[..., :3], weights).astype(np.uint8)
    return img

def desaturation(image):
    min_val = np.min(image, axis=2)
    max_val = np.max(image, axis=2)
    img = (min_val + max_val) // 2
    return img

def decomposition(image):
    gray_max = np.max(image, axis=2)
    gray_min = np.min(image, axis=2)
    return gray_max, gray_min

def single_color_channel(image, channel):
    if channel == 'R':
        return image[:, :, 2]
    elif channel == 'G':
        return image[:, :, 1]
    elif channel == 'B':
        return image[:, :, 0]

def custom_gray_shades(image, p):
    img = weighted_average(image)
    bins = np.linspace(0, 256, p + 1)
    new_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(p):
        min_val = bins[i]
        max_val = bins[i + 1]
        mask = (img >= min_val) & (img < max_val)
        new_img[mask] = int((min_val + max_val) / 2)

    return new_img

def floyd_steinberg_dithering(image):
    img = weighted_average(image)
    height, width = img.shape
    dithered = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                img[y, x + 1] = np.clip(img[y, x + 1] + error * 7 / 16, 0, 255)
            if y + 1 < height:
                if x > 0:
                    img[y + 1, x - 1] = np.clip(img[y + 1, x - 1] + error * 3 / 16, 0, 255)
                img[y + 1, x] = np.clip(img[y + 1, x] + error * 5 / 16, 0, 255)
                if x + 1 < width:
                    img[y + 1, x + 1] = np.clip(img[y + 1, x + 1] + error * 1 / 16, 0, 255)

    return dithered

def stucki_dithering(image):
    img = weighted_average(image)
    height, width = img.shape
    dithered = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            dithered[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                img[y, x + 1] = np.clip(img[y, x + 1] + error * 8 / 42, 0, 255)
            if x + 2 < width:
                img[y, x + 2] = np.clip(img[y, x + 2] + error * 4 / 42, 0, 255)
            if y + 1 < height:
                if x > 0:
                    img[y + 1, x - 1] = np.clip(img[y + 1, x - 1] + error * 2 / 42, 0, 255)
                img[y + 1, x] = np.clip(img[y + 1, x] + error * 4 / 42, 0, 255)
                if x + 1 < width:
                    img[y + 1, x + 1] = np.clip(img[y + 1, x + 1] + error * 8 / 42, 0, 255)
            if y + 2 < height:
                if x > 1:
                    img[y + 2, x - 2] = np.clip(img[y + 2, x - 2] + error * 1 / 42, 0, 255)
                if x > 0:
                    img[y + 2, x - 1] = np.clip(img[y + 2, x - 1] + error * 2 / 42, 0, 255)
                img[y + 2, x] = np.clip(img[y + 2, x] + error * 4 / 42, 0, 255)
                if x + 1 < width:
                    img[y + 2, x + 1] = np.clip(img[y + 2, x + 1] + error * 2 / 42, 0, 255)
                if x + 2 < width:
                    img[y + 2, x + 2] = np.clip(img[y + 2, x + 2] + error * 1 / 42, 0, 255)

    return dithered


image = load_image("pencils.jpeg")

gray_avg = simple_averaging(image)
gray_weighted = weighted_average(image)
gray_desaturation = desaturation(image)
gray_max, gray_min = decomposition(image)
gray_single_r = single_color_channel(image, 'R')
gray_single_b = single_color_channel(image, 'B')
gray_custom = custom_gray_shades(image, 5)
gray_floyd = floyd_steinberg_dithering(image)
gray_stucki = stucki_dithering(image)

cv2.imshow('Original', image)
cv2.imshow('Simple Averaging', gray_avg)
cv2.imshow('Weighted Average', gray_weighted)
cv2.imshow('Desaturation', gray_desaturation)
cv2.imshow('Maximum Decomposition', gray_max)
cv2.imshow('Minimum Decomposition', gray_min)
cv2.imshow('Single Red Channel', gray_single_r)
cv2.imshow('Single Blue Channel', gray_single_b)
cv2.imshow('Custom Gray Shades', gray_custom)
cv2.imshow('Floyd-Steinberg Dithering', gray_floyd)
cv2.imshow('Stucki Dithering', gray_stucki)

cv2.waitKey(0)
cv2.destroyAllWindows()