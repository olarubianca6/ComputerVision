import cv2
import numpy as np
import os

skin_path = 'skin/'
output_rgb = 'output/rgb/'
output_hsv = 'output/hsv/'
output_ycbcr = 'output/ycbcr/'
output_hsv2 = 'output/hsv2/'


def skin_rgb(img):
    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    mask = (R > 95) & (G > 40) & (B > 20) & \
           ((np.max(img, axis=2) - np.min(img, axis=2)) > 15) & \
           (np.abs(R - G) > 15) & (R > G) & (R > B)
    return mask.astype(np.uint8) * 255


def skin_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    mask = (H >= 0) & (H <= 25) & (S >= 0.23) & (S <= 0.68) & (V >= 0.35) & (V <= 1.0)
    return mask.astype(np.uint8) * 255


def skin_ycbcr(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    mask = (Y > 80) & (Cb >= 85) & (Cb <= 135) & (Cr >= 135) & (Cr <= 180)
    return mask.astype(np.uint8) * 255


for filename in os.listdir(skin_path):
    image_path = os.path.join(skin_path, filename)
    image = cv2.imread(image_path)

    rgb_mask = skin_rgb(image)
    hsv_mask = skin_hsv(image)
    ycbcr_mask = skin_ycbcr(image)

    # cv2.imwrite(os.path.join(output_rgb, f"{filename}_rgb_mask.png"), rgb_mask)
    # cv2.imwrite(os.path.join(output_hsv, f"{filename}_hsv_mask.png"), hsv_mask)
    cv2.imwrite(os.path.join(output_hsv2, f"{filename}_hsv_mask.png"), hsv_mask)
    # cv2.imwrite(os.path.join(output_ycbcr, f"{filename}_ycbcr_mask.png"), ycbcr_mask)
