import cv2
import os
from skin_detection import skin_rgb

pratheepan_path = 'dataset/Pratheepan_Dataset/FacePhoto/'
output_path = 'output/faces/'


def detect_face(mask, image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    face_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if 0.5 < aspect_ratio < 1.5:
                face_contours.append(contour)

    if face_contours:
        largest_contour = max(face_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        side = max(w, h)
        x_center, y_center = x + w // 2, y + h // 2
        x = max(0, x_center - side // 2)
        y = max(0, y_center - side // 2)

        face_box = image.copy()
        cv2.rectangle(face_box, (x, y), (x + side, y + side), (0, 255, 0), 2)

        return face_box, (x, y, side, side)
    else:
        print("No face detected.")
        return image, None


for filename in os.listdir(pratheepan_path):
    image_path = os.path.join(pratheepan_path, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image '{filename}'")
        continue

    skin_mask = skin_rgb(image)
    face_image, bounding_box = detect_face(skin_mask, image)

    cv2.imwrite(os.path.join(output_path, filename), face_image)