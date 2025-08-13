import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Project configuration
PROJECT_PATH = "D:\my desk\Coding\Projects\Driver_Drowsiness_detection(CNN multithreading)"
EYE_DATA_PATH = os.path.join(PROJECT_PATH, "data", "eye_data")

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directories
train_generator = train_datagen.flow_from_directory(
    os.path.join(EYE_DATA_PATH, 'train'),
    target_size=(36, 26),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(EYE_DATA_PATH, 'validation'),
    target_size=(36, 26),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(EYE_DATA_PATH, 'test'),
    target_size=(36, 26),
    batch_size=32,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False
)

# Model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(36, 26, 1)),
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_eye_model.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=callbacks
)

# Evaluate on test set
model.evaluate(test_generator)

# Save the model
model.save(os.path.join(PROJECT_PATH, 'models', 'eye_model.h5'))

# Generate classification report
Y_pred = model.predict(test_generator)
y_pred = np.where(Y_pred > 0.5, 1, 0).flatten()
print(classification_report(test_generator.classes, y_pred))
