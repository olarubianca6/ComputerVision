import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix

IMAGE_SIZE = (128, 128)
MODEL_PATH = 'tumor_unet_model.h5'
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def load_images_and_masks(image_dir, mask_dir, image_size=IMAGE_SIZE):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            images.append(img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, image_size)
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)

    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    masks = (masks > 0.5).astype(np.float32)
    return images, masks


def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    cm = confusion_matrix(y_true_flat, y_pred_flat)

    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]

    pixel_accuracy = (TP + TN) / (TP + FP + TN + FN)

    jaccard_index = TP / (TP + FP + FN)

    dice_coefficient = 2 * TP / (2 * TP + FP + FN)

    return pixel_accuracy, jaccard_index, dice_coefficient


image_dir = 'evaluation/img'
mask_dir = 'evaluation/masks'

images, masks = load_images_and_masks(image_dir, mask_dir)

y_pred = model.predict(images)

y_pred_binary = (y_pred > 0.5).astype(np.float32)

pixel_accuracy, jaccard_index, dice_coefficient = compute_metrics(masks, y_pred_binary)

print(f"Mean Pixel Accuracy: {pixel_accuracy}")
print(f"Mean Jaccard Index (IoU): {jaccard_index}")
print(f"Mean Dice Coefficient: {dice_coefficient}")