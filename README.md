# Car Detection — HOG + Sliding Window

A classic computer vision pipeline for detecting cars in images. The approach combines Histogram of Oriented Gradients (HOG) feature descriptors, a linear SVM classifier, and a multi-scale sliding window search. The dataset used is the UIUC Image Database for Car Detection.

---

## Dataset

Download the UIUC Car dataset and extract it into the project root:

```
wget http://cogcomp.cs.illinois.edu/Data/Car/CarData.tar.gz
tar -xzf CarData.tar.gz
```

After extraction the directory structure should look like:

```
CarData/
    TrainImages/
        pos-*.pgm   # 550 car images
        neg-*.pgm   # 500 non-car images
    TestImages/
        test-*.pgm
```

---

## Method

1. **HOG Feature Extraction** — extract gradient orientation histograms from each 100×40 training patch.
2. **Linear SVM** — train a binary classifier on the HOG feature vectors (car vs. non-car).
3. **Sliding Window** — scan test images at multiple scales; run the classifier inside each window.
4. **Non-Maximum Suppression** — merge overlapping detections into a single bounding box.

---

## Project Structure

```
main.ipynb      — full pipeline notebook (data prep → evaluation)
CarData/        — dataset (downloaded separately, see above)
```

---

## Requirements

```
pip install opencv-python scikit-image scikit-learn numpy matplotlib
```

Python 3.8+ is recommended.

---

## Usage

Open `main.ipynb` in Jupyter and run cells top-to-bottom. Make sure the `CarData/` directory is present in the same folder as the notebook before running.

---

## Results

Detection accuracy and example output images will be shown at the end of `main.ipynb` after the evaluation section is complete.
