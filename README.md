# Vehicle Detection using HOG and Sliding Window

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An implementation of a classic computer vision object detection pipeline. This project uses Histogram of Oriented Gradients (HOG) features, a Linear Support Vector Machine (SVM), and a Sliding Window approach to detect vehicles.

## Core Concepts
* **HOG (Histogram of Oriented Gradients):** A feature extraction technique that helps the computer "see" shapes. Instead of looking at raw pixel colors, HOG analyzes the edges and the direction of light-to-dark transitions (gradients) in an image. This makes it incredibly effective at recognizing the structural outline of a car, regardless of whether the car is red, blue, in bright sunlight, or in the shadows.
* **Sliding Window:** A spatial scanning technique used to locate objects. Imagine taking a small rectangular frame and sliding it step-by-step across a larger photograph. At every single stop, the system extracts the HOG features of that specific patch and asks the trained classifier, "Is there a car inside this exact frame?" 

## Dataset
This project utilizes the **UIUC Image Database for Car Detection**. 
* **Link:** [CarData.tar.gz](http://cogcomp.cs.illinois.edu/Data/Car/CarData.tar.gz)
* The dataset contains cleanly cropped side-profile images of cars (positive samples) and various background scenes (negative samples).

## Pipeline Architecture
1.  **Data Preparation:** Automated downloading, extraction, and sorting of positive and negative training images.
2.  **Feature Extraction:** Computing HOG descriptors to capture gradient structures.
3.  **Classifier Training:** Training a `LinearSVC` model to distinguish between "Car" and "Non-Car" spatial features.
4.  **Sliding Window Detection:** Scanning a test image at various locations to locate the target object.
5.  **Non-Maximum Suppression (NMS):** Filtering out overlapping bounding boxes to yield a single, highly confident detection per vehicle.

## Data Visualization
To understand how the SVM distinguishes vehicles, we visualize the HOG features. Notice how the HOG gradient map heavily highlights the structural edges of the cars, making the classifier highly robust.

![Dataset Samples and HOG Features](asset1.pngg)

## Detection Results
Applying the sliding window across a test image initially results in multiple overlapping detections around the target. By applying Non-Maximum Suppression (NMS) via OpenCV, we refine these into a single, accurate bounding box.

![Final Detection Result](asset2.png)

## How to Run
This project is formatted for execution in **Google Colab**. 
1. Open the provided Jupyter Notebook in Google Colab.
2. Go to **Runtime > Run all**.
3. The script will automatically fetch the dataset, train the model, and output the visualizations shown above.

## Future Improvements
While HOG + Sliding Window is a foundational computer vision technique, exhaustive spatial scanning is computationally expensive. Future iterations of this project could explore region proposal networks (like R-CNN) or single-shot detectors (like YOLO) to bypass the sliding window for real-time inference speeds.
