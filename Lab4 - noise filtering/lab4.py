import pytesseract
import cv2
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

gt1 = 'Tesseract Will Fail With Noisy Backgrounds'

gt2 = ("PREREQUISITES In order to make the most of this, you will need to have a little bit of programming experience. "
       "All examples in this book are in the Python programming language. Familiarity with Python or other scripting languages is suggested, but not required. "
       "You'll also need to know some basic mathematics. This book is hands-on and example driven:"
       "lots of examples and lots of code, so even if your math skills are not up to par, do not worry!"
       "The examples are very detailed and heavily documented to help you follow along.")

img_path = 'images/'

def calculate_accuracy(gt_text, ocr_text):
    gt_chars = len(gt_text)
    matched_chars = len([c for c, g in zip(ocr_text, gt_text) if c == g])
    return matched_chars / gt_chars * 100

def add_gaussian_noise(image):
    return np.clip(image + np.random.normal(0, 25, image.shape).astype(np.uint8), 0, 255)


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def shear_image(image, shear_factor):
    rows, cols, _ = image.shape
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (cols + int(shear_factor * rows), rows))

def resize_image(image):
    return cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 1)

def sharpen_image(image):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, sharpen_kernel)

results = []
final_results = []

for filename in os.listdir(img_path):
    image_path = os.path.join(img_path, filename)
    image = cv2.imread(image_path)

    gaussian = add_gaussian_noise(image)
    salt_pepper = add_salt_and_pepper_noise(image)
    rotated = rotate_image(image, 30)
    sheared = shear_image(image, 0.2)
    resized = resize_image(image)
    blurred = blur_image(image)
    sharpened = sharpen_image(image)

    cv2.imwrite(f'processed_images/{filename}_gaussian_noise.jpg', gaussian)
    results.append(("Gaussian Noise", f'processed_images/{filename}_gaussian_noise.jpg', filename))

    cv2.imwrite(f'processed_images/{filename}_salt_pepper_noise.jpg', salt_pepper)
    results.append(("Salt-and-Pepper Noise", f'processed_images/{filename}_salt_pepper_noise.jpg', filename))

    cv2.imwrite(f'processed_images/{filename}_rotated.jpg', rotated)
    results.append(("Rotation", f'processed_images/{filename}_rotated.jpg', filename))

    cv2.imwrite(f'processed_images/{filename}_sheared.jpg', sheared)
    results.append(("Shear", f'processed_images/{filename}_sheared.jpg', filename))

    cv2.imwrite(f'processed_images/{filename}_resized.jpg', resized)
    results.append(("Resized", f'processed_images/{filename}_resized.jpg', filename))

    cv2.imwrite(f'processed_images/{filename}_blurred.jpg', blurred)
    results.append(("Gaussian Blur", f'processed_images/{filename}_blurred.jpg', filename))

    cv2.imwrite(f'processed_images/{filename}_sharpened.jpg', sharpened)
    results.append(("Sharpening", f'processed_images/{filename}_sharpened.jpg', filename))

for transformation, image_path, filename in results:
    processed_image = cv2.imread(image_path)
    ocr_text = pytesseract.image_to_string(processed_image).strip()
    ground_truth = gt1 if "example_02" in filename else gt2
    accuracy = calculate_accuracy(ground_truth, ocr_text)
    final_results.append((transformation, accuracy, ocr_text))
    print(f"Transformation: {transformation}, OCR Accuracy: {accuracy:.2f}%")

for transformation, accuracy, ocr_output in final_results:
    print(f"Transformation: {transformation}\nAccuracy: {accuracy:.2f}%\nOCR Output: {ocr_output}\n")