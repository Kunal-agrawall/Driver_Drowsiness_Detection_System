import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Project configuration
PROJECT_PATH = "D:\my desk\Coding\Projects\Driver_Drowsiness_detection(CNN multithreading)"
YAWN_DATA_PATH = os.path.join(PROJECT_PATH, "data", "yawn_data")

# Verify data existence
if not os.path.exists(YAWN_DATA_PATH):
    raise FileNotFoundError(f"Yawn data path {YAWN_DATA_PATH} does not exist")

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into train and validation
)

# Flow from directories
train_generator = train_datagen.flow_from_directory(
    YAWN_DATA_PATH,
    target_size=(64, 32),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    YAWN_DATA_PATH,
    target_size=(64, 32),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'
)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 32, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training with early stopping
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(os.path.join(PROJECT_PATH, 'models', 'best_yawn_model.h5'), 
                    save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)

# Evaluate on test set
test_generator = train_datagen.flow_from_directory(
    YAWN_DATA_PATH,
    target_size=(64, 32),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'
)

model.evaluate(test_generator)

# Generate classification report
Y_pred = model.predict(test_generator)
y_pred = np.where(Y_pred > 0.5, 1, 0).flatten()
print(classification_report(test_generator.classes, y_pred))

# Save the model in TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a file
tflite_model_path = os.path.join(PROJECT_PATH, 'models', 'yawn_model.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TensorFlow Lite model saved to {tflite_model_path}")
