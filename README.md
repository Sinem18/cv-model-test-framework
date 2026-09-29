# 🤖 CV Model Test Framework

A lightweight and reusable Computer Vision testing framework for evaluating
YOLO-based object detection models on annotated datasets.

The framework runs object detection models on validation images, compares
predictions with ground-truth annotations, calculates evaluation metrics,
and visualizes the results through a Streamlit dashboard.

---

## 🚀 Project Overview

This project was developed to create a reusable testing pipeline for
Computer Vision object detection models.

Instead of evaluating a model manually image by image, the framework
automates the evaluation process:

- Loads an annotated dataset
- Loads a YOLO object detection model
- Runs predictions on validation images
- Filters predictions by confidence threshold
- Matches predictions with ground-truth bounding boxes using IoU
- Calculates True Positive, False Positive and False Negative values
- Calculates Precision and Recall
- Visualizes ground-truth and predicted bounding boxes
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
│   ├── dataset_manager.py
│   ├── evaluator.py
│   ├── model_runner.py
│   └── visualizer.py
│
├── scripts/
│   ├── download_dataset.py
│   └── run_evaluation.py
│
├── tests/
│   └── test_evaluator.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── bus.jpg
