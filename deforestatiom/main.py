import cv2
import torch
import numpy as np

# Load YOLOv5 model
path = 'C:/Users/user/Downloads/best (5).pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)


# Function to perform inference and display results
def perform_inference_and_display(image_path):
    # Read the input image
    img = cv2.imread(image_path)

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Display the results
    results.show()


# Input image path
input_image_path = 'E:/33.jpg'

# Call the function to perform inference and display results
perform_inference_and_display(input_image_path)
