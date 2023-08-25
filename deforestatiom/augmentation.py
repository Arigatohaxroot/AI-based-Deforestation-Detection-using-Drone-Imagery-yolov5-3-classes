import os
import cv2
import numpy as np
import imgaug.augmenters as iaa

# Function to load annotations from a file
def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = f.read().splitlines()
    return annotations

# Function to find images with corresponding annotation files
def find_images_with_labels(images_dir, labels_dir):
    image_files = os.listdir(images_dir)
    image_paths = [os.path.join(images_dir, img_file) for img_file in image_files]
    images_with_labels = []
    for img_path in image_paths:
        annotation_file = os.path.basename(img_path).replace('.jpg', '.txt')
        annotation_path = os.path.join(labels_dir, annotation_file)
        if os.path.exists(annotation_path):
            images_with_labels.append((img_path, annotation_path))
    return images_with_labels

# Function to apply augmentations to an image
def apply_augmentations(image):
    # Define augmentation pipeline
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # apply vertical flip
        iaa.Affine(rotate=(-10, 10)),  # apply rotation
        iaa.Multiply((0.8, 1.2), per_channel=0.2),  # adjust brightness
    ])
    augmented_image = seq(image=image)
    return augmented_image

# Function to augment images with corresponding annotations
def augment_images_with_labels(images_with_labels, output_dir, num_augmented_images_per_image=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, "images")
    annotations_output_dir = os.path.join(output_dir, "annotations")
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    if not os.path.exists(annotations_output_dir):
        os.makedirs(annotations_output_dir)

    total_augmented_images = 0

    for img_path, annotation_path in images_with_labels:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not read image '{img_path}'")
            continue

        # Load annotations
        annotations = load_annotations(annotation_path)

        # Apply augmentations and save the augmented images and annotations
        for i in range(num_augmented_images_per_image):
            augmented_image = apply_augmentations(image)

            # Save augmented image
            output_img_path = os.path.join(images_output_dir, f"{os.path.basename(img_path).split('.')[0]}_aug_{total_augmented_images}.jpg")
            cv2.imwrite(output_img_path, augmented_image)

            # Save updated annotations (just copy them)
            output_annotation_path = os.path.join(annotations_output_dir, f"{os.path.basename(annotation_path).split('.')[0]}_aug_{total_augmented_images}.txt")
            with open(annotation_path, 'r') as f:
                annotations_content = f.read()
            with open(output_annotation_path, 'w') as f:
                f.write(annotations_content)

            total_augmented_images += 1

    print(f"Total augmented images generated: {total_augmented_images}")

if __name__ == "__main__":
    input_train_images_dir = r"E:\deforestation.v2i.yolov5pytorch\train\images"
    input_train_labels_dir = r"E:\deforestation.v2i.yolov5pytorch\train\labels"
    input_valid_images_dir = r"E:\deforestation.v2i.yolov5pytorch\valid\images"
    input_valid_labels_dir = r"E:\deforestation.v2i.yolov5pytorch\valid\labels"
    input_test_images_dir = r"E:\deforestation.v2i.yolov5pytorch\test\images"
    input_test_labels_dir = r"E:\deforestation.v2i.yolov5pytorch\test\labels"

    output_dir = r"E:\output_augmented_dataset"

    print("Augmenting training images...")
    train_images_with_labels = find_images_with_labels(input_train_images_dir, input_train_labels_dir)
    augment_images_with_labels(train_images_with_labels, os.path.join(output_dir, 'train'), num_augmented_images_per_image=10)

    print("Augmenting validation images...")
    valid_images_with_labels = find_images_with_labels(input_valid_images_dir, input_valid_labels_dir)
    augment_images_with_labels(valid_images_with_labels, os.path.join(output_dir, 'valid'), num_augmented_images_per_image=5)

    print("Augmenting testing images...")
    test_images_with_labels = find_images_with_labels(input_test_images_dir, input_test_labels_dir)
    augment_images_with_labels(test_images_with_labels, os.path.join(output_dir, 'test'), num_augmented_images_per_image=5)

    print("Finished augmenting images.")
