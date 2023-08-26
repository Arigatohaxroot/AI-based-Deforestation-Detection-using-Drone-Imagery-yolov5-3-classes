# AI-Based Deforestation Detection using Drone Imagery

## Overview

Welcome to my AI-Based Deforestation Detection project! 🌿🚁

This repository showcases my innovative approach to detecting deforestation using drone-captured imagery. Leveraging advanced AI technologies, I've developed a solution that can identify different landscape features within images, ranging from flourishing forests to areas afflicted by deforestation.

## Dataset and Training

### Dataset

I've compiled a comprehensive dataset of over 250 high-quality images, each sourced from diverse geographical locations. To ensure the accuracy of my model, I've meticulously annotated these images using [Roboflow](https://universe.roboflow.com/haxroot/deforestation-uf6lu), providing precise ground truth labels for both training and testing.

### Training Process

My model relies on YOLOv5, a state-of-the-art object detection framework. After extensive training on my annotated dataset, I've created a custom model named `best (6).pt`. This model is proficient in classifying images into three distinct classes:

- Brown: Depicting areas affected by deforestation
- Green: Representing lush green regions or grassy landscapes
- Trees: Identifying dense forested areas

## Repository Structure

- `deforestation/`: This directory contains the trained YOLOv5 model (`best (6).pt`) alongside other relevant files.
- `main.py`: This script reads individual images, performs detection using the trained model, and generates annotated images with bounding boxes.
- `video.py`: Allows the model to detect deforested areas in video input.

## Visual Examples

<div align="center">

### Green Area with Trees
![Green Area with Trees](https://i.imgur.com/xv3kVJO.jpg)

### Deforested Area
![Deforested Area](https://i.imgur.com/e6qtFZc.jpg)

### Deforestation Detection in Action
![Deforestation Detection GIF](https://github.com/Arigatohaxroot/AI-based-Deforestation-Detection-using-Drone-Imagery-yolov5-3-classes/blob/d6259e97b7f2af0aaed94c2c223e674740d0cd5c/exp8.gif)

### Another Deforestation Detection GIF
![Another Deforestation Detection GIF](https://github.com/Arigatohaxroot/AI-based-Deforestation-Detection-using-Drone-Imagery-yolov5-3-classes/blob/7f41b4716084f3d2ddffd4a60975177ce4063230/exp9.gif)

### Green Area with Trees
![Green Area with Trees](https://github.com/Arigatohaxroot/AI-based-Deforestation-Detection-using-Drone-Imagery-yolov5-3-classes/blob/79c6ae9172dc81445552f46837378186d86b810d/exp4.gif)

</div>

## Project Goals and Beyond

My primary objective is to leverage AI and drone technology for comprehensive deforestation analysis. Beyond detection, I have a larger vision:

- **Reforestation:** I'm dedicated to addressing the aftermath of deforestation. My vision includes a mechanism to dispense seeds precisely in deforested areas, initiating reforestation efforts and nurturing sustainable ecosystems.

## Getting Involved

I value collaboration and the shared passion for environmental sustainability. To contribute, follow these steps:

1. Fork the repository.
2. Implement your enhancements or fixes.
3. Submit a pull request, and I'll review your contribution!

## Future Development Ideas

The potential for expansion and impact is immense. Here are some captivating ideas for the project's future:

- **Real-time Monitoring:** Enable real-time deforestation monitoring through live drone feeds, allowing instant responses to changes.
- **Ecosystem Analysis:** Extend the model to assess biodiversity, identifying specific species and gauging ecosystem health.
- **Community Engagement:** Develop a user-friendly interface to encourage public participation in reporting deforestation incidents.

## Acknowledgments

I extend my gratitude to the open-source community, the contributors, and the researchers who've made AI solutions like this possible.

Join me in my mission to make a positive environmental impact. 🌍✨

</div>
