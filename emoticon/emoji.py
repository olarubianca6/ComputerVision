import cv2
import numpy as np

image = np.ones((500, 500, 3), dtype="uint8") * 255

head_color = (160, 150, 160)

# head

cv2.circle(image, (250, 250), 200, (0, 0, 0), 5)
cv2.circle(image, (250, 250), 200, head_color, -1)

# ears

cv2.polylines(image, [np.array([[100, 100], [200, 50], [200, 150]], np.int32)],
              isClosed=True, color=(0, 0, 0), thickness=5)
cv2.fillPoly(image, [np.array([[100, 100], [200, 50], [200, 150]], np.int32)], head_color)

cv2.polylines(image, [np.array([[400, 100], [300, 50], [300, 150]], np.int32)],
              isClosed=True, color=(0, 0, 0), thickness=5)
cv2.fillPoly(image, [np.array([[400, 100], [300, 50], [300, 150]], np.int32)], head_color)

# eyes and eyebrows

cv2.circle(image, (180, 220), 30, (0, 0, 0), -1)
cv2.circle(image, (320, 220), 30, (0, 0, 0), -1)

cv2.line(image, (150, 170), (210, 190), (0, 0, 0), 8)
cv2.line(image, (290, 190), (350, 170), (0, 0, 0), 8)

# nose

nose_points = np.array([[240, 300], [260, 300], [250, 320]], np.int32)
cv2.polylines(image, [nose_points], isClosed=True, color=(0, 0, 0), thickness=5)
cv2.fillPoly(image, [nose_points], (0, 0, 0))

# mouth

frown_points = np.array([[210, 350], [250, 340], [290, 350]], np.int32)
cv2.polylines(image, [frown_points], isClosed=False, color=(0, 0, 0), thickness=5)

cv2.line(image, (250, 320), (250, 340), (0, 0, 0), 5)

cv2.imshow("Harry", image)
# cv2.imwrite("harry.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()