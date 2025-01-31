import os
import numpy as np
import cv2
import tensorflow as tf

IMAGE_SIZE = (128, 128)
MODEL_PATH = "tumor_unet_model.h5"
DATASET_DIR = "Lab3_dataset"
OUTPUT_DIR = "test_output"

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask not found: {mask_path}")
    mask = cv2.resize(mask, IMAGE_SIZE)
    mask = mask / 255.0
    mask = (mask > 0.5).astype(np.float32)
    return mask


def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    pixel_accuracy = (TP + TN) / (TP + TN + FP + FN)

    jaccard_index = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    dice_coefficient = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return pixel_accuracy, jaccard_index, dice_coefficient


def evaluate_model_on_dataset(dataset_dir, ground_truth_dir, category):
    input_dir = os.path.join(dataset_dir, category)
    gt_dir = os.path.join(ground_truth_dir, f"GroundT_{category}")

    input_images = sorted(os.listdir(input_dir))
    gt_images = sorted(os.listdir(gt_dir))

    total_pixel_accuracy = 0
    total_jaccard_index = 0
    total_dice_coefficient = 0
    num_samples = len(input_images)

    category_output_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_output_dir, exist_ok=True)

    for i, (img_name, mask_name) in enumerate(zip(input_images, gt_images)):
        img_path = os.path.join(input_dir, img_name)
        mask_path = os.path.join(gt_dir, mask_name)

        input_img = preprocess_image(img_path)
        true_mask = preprocess_mask(mask_path)

        pred_mask = (model.predict(input_img) > 0.5).astype(np.float32).squeeze()

        pixel_accuracy, jaccard_index, dice_coefficient = calculate_metrics(true_mask, pred_mask)

        total_pixel_accuracy += pixel_accuracy
        total_jaccard_index += jaccard_index
        total_dice_coefficient += dice_coefficient

        pred_filename = os.path.join(category_output_dir, f"pred_{i + 1}.png")
        pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
        cv2.imwrite(pred_filename, pred_mask_uint8)

    mean_pixel_accuracy = total_pixel_accuracy / num_samples
    mean_jaccard_index = total_jaccard_index / num_samples
    mean_dice_coefficient = total_dice_coefficient / num_samples

    return mean_pixel_accuracy, mean_jaccard_index, mean_dice_coefficient


categories = ["FacePhoto", "FamilyPhoto"]
for category in categories:
    print(f"Evaluating on {category} dataset...")
    mean_pixel_accuracy, mean_jaccard_index, mean_dice_coefficient = evaluate_model_on_dataset(
        os.path.join(DATASET_DIR, "Pratheepan_Dataset"),
        os.path.join(DATASET_DIR, "Ground_Truth"),
        category
    )
    print(f"Results for {category}:")
    print(f"Mean Pixel Accuracy: {mean_pixel_accuracy:.4f}")
    print(f"Mean Jaccard Index (IoU): {mean_jaccard_index:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice_coefficient:.4f}")