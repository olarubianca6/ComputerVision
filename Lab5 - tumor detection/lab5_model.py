import os
import keras
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, Model

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

def load_data(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load image: {img_path}")
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        images.append(img)

        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Unable to load mask: {mask_path}")
            continue

        mask = cv2.resize(mask, IMAGE_SIZE)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)

    if not images or not masks:
        raise ValueError("No valid images or masks found.")

    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    masks = (masks > 0.5).astype(np.float32)
    return images, masks

image_dir = "tumorDataset/images"
mask_dir = "tumorDataset/masks"
images, masks = load_data(image_dir, mask_dir)

X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.3, random_state=42)

def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # decoder
    u6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model(input_size=IMAGE_SIZE + (3,))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE, epochs=EPOCHS)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

mean_iou_metric = keras.metrics.MeanIoU(num_classes=2)
y_pred = (model.predict(X_test) > 0.5).astype(np.float32)
mean_iou_metric.update_state(y_test.flatten(), y_pred.flatten())
print(f"Mean IoU: {mean_iou_metric.result().numpy()}")

output_dir = "visualizations"
input_dir = os.path.join(output_dir, "inputs")
mask_dir = os.path.join(output_dir, "masks")
prediction_dir = os.path.join(output_dir, "predictions")

os.makedirs(input_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(prediction_dir, exist_ok=True)

num_samples = 5
for i in range(min(num_samples, len(X_test))):
    input_image = (X_test[i] * 255).astype(np.uint8)
    true_mask = (y_test[i].squeeze() * 255).astype(np.uint8)
    predicted_mask = (y_pred[i].squeeze() * 255).astype(np.uint8)

    input_filename = os.path.join(input_dir, f"input_{i + 1}.png")
    cv2.imwrite(input_filename, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

    mask_filename = os.path.join(mask_dir, f"mask_{i + 1}.png")
    cv2.imwrite(mask_filename, true_mask)

    prediction_filename = os.path.join(prediction_dir, f"prediction_{i + 1}.png")
    cv2.imwrite

model.save("tumor_unet_model.h5")