import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load CSV data
def load_csv(csv_path):
    data = pd.read_csv(csv_path)
    return data

# Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data loading and preprocessing
data_dir = 'F:/deforestation.v12i.tensorflow'  # Base directory
train_image_dir = os.path.join(data_dir, 'train')
val_image_dir = os.path.join(data_dir, 'valid')
train_annotation_path = os.path.join(train_image_dir, '_annotations.csv')
val_annotation_path = os.path.join(val_image_dir, '_annotations.csv')

# Load annotations CSV
train_annotations = load_csv(train_annotation_path)
val_annotations = load_csv(val_annotation_path)
train_image_paths = [os.path.join(train_image_dir, filename) for filename in train_annotations['filename']]
val_image_paths = [os.path.join(val_image_dir, filename) for filename in val_annotations['filename']]
train_labels = np.array(train_annotations['class'])  # Assuming 'class' contains the labels (0, 1, 2)
val_labels = np.array(val_annotations['class'])  # Assuming 'class' contains the labels (0, 1, 2)

# Data generators
batch_size = 32
input_shape = (224, 224, 3)

train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({'filename': train_image_paths, 'class': train_labels}),
    x_col='filename',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame({'filename': val_image_paths, 'class': val_labels}),
    x_col='filename',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Train the model
history = model.fit(train_generator,
                    epochs=40,
                    validation_data=val_generator)
model.save('F:/saved_model')
# Display accuracy and loss curves
# Display accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()