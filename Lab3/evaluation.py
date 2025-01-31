import cv2
import numpy as np
import os
from skin_detection import skin_rgb, skin_hsv, skin_ycbcr

pratheepan_path = 'dataset/Pratheepan_Dataset/FamilyPhoto'
ground_truth_path = 'dataset/Ground_Truth/GroundT_FamilyPhoto'


def calculate_confusion_accuracy(pred_mask, gt_mask):
    TP = np.sum((pred_mask == 255) & (gt_mask == 255))  # true positives
    TN = np.sum((pred_mask == 0) & (gt_mask == 0))      # true negatives
    FP = np.sum((pred_mask == 255) & (gt_mask == 0))    # false positives
    FN = np.sum((pred_mask == 0) & (gt_mask == 255))    # false negatives
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return TP, TN, FP, FN, accuracy


for filename in os.listdir(pratheepan_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(pratheepan_path, filename)

        ground_truth_filename = filename.replace('.jpg', '.png')
        ground_truth_path_img = os.path.join(ground_truth_path, ground_truth_filename)

        image = cv2.imread(image_path)
        ground_truth = cv2.imread(ground_truth_path_img, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error loading image '{filename}'")
            continue
        if ground_truth is None:
            print(f"Error loading ground truth for '{ground_truth_filename}'")
            continue

        binary_mask_rgb = skin_rgb(image)
        binary_mask_hsv = skin_hsv(image)
        binary_mask_ycbcr = skin_ycbcr(image)

        for method, mask in zip(['RGB', 'HSV', 'YCbCr'], [binary_mask_rgb, binary_mask_hsv, binary_mask_ycbcr]):
            TP, TN, FP, FN, accuracy = calculate_confusion_accuracy(mask, ground_truth)
            print(f"Image: {filename} - {method}")
            print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
            print(f"Accuracy = {accuracy:.4f}\n")