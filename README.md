# CV Model Test Framework

A lightweight Computer Vision model testing framework for evaluating
YOLO-based object detection models on annotated datasets.

The project provides an automated evaluation pipeline that calculates
object detection metrics and visualizes model predictions together with
ground-truth annotations.

---

## 🚀 Project Overview

This project was developed to create a reusable testing framework for
Computer Vision object detection models.

The framework:

- Loads an annotated dataset
- Loads a YOLO object detection model
- Runs predictions on validation images
- Applies confidence filtering
- Matches predictions with ground-truth annotations
- Calculates IoU
- Calculates True Positive, False Positive and False Negative values
- Calculates Precision and Recall
- Visualizes ground-truth and prediction bounding boxes
- Provides an interactive Streamlit dashboard

---

## 🏗️ Project Architecture

```text
cv-model-test-framework/
│
├── app/
│   └── app.py
│
├── src/
│   ├── __init__.py
│   ├── dataset_manager.py
│   ├── model_runner.py
│   ├── evaluator.py
│   └── visualizer.py
│
├── scripts/
│   ├── download_dataset.py
│   └── run_evaluation.py
│
├── tests/
│   └── test_evaluator.py
│
├── data/
│
├── runs/
│
├── .gitignore
├── README.md
├── requirements.txt
└── bus.jpg