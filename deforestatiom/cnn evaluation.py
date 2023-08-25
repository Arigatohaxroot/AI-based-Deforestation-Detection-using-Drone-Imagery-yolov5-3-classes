import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Paths
saved_model_path = 'F:/saved_model'  # Path to your saved model
image_path_to_predict = 'E:/2.jpeg'  # Path to the image you want to predict

# Load the model
model = load_model(saved_model_path)  # Load your CNN model here

# Load and preprocess the image
image = tf.keras.preprocessing.image.load_img(
    image_path_to_predict,
    target_size=(224, 224)  # Resize to the input shape expected by the model (224, 224)
)
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension
image_array = tf.keras.applications.resnet50.preprocess_input(image_array)  # Preprocess image

# Make predictions for bounding boxes
predictions = model.predict(image_array)

# Visualization
plt.imshow(image)

for j, label in enumerate(['brown', 'green', 'trees']):
    class_bboxes = predictions[j]  # Get predicted bounding boxes for this class

    for bbox in class_bboxes:
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = bbox

        # Draw predicted bounding box
        plt.gca().add_patch(Rectangle((pred_xmin, pred_ymin),
                                      pred_xmax - pred_xmin,
                                      pred_ymax - pred_ymin,
                                      edgecolor='blue', fill=False))
        plt.text(pred_xmin, pred_ymin, label, color='blue')

plt.title(f"Predicted Bounding Boxes for Image: {image_path_to_predict}")
plt.show()
