# Breast Cacer detection using ML algorithms

This project uses machine learning techniques to detect breast cancer based on various features extracted from breast mass biopsies. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
Kaggle Dataset source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Project Overview

The goal of this project is to build and evaluate several classification models to predict whether a breast mass is malignant or benign. The models used in this project include:

- Decision Tree Classifier
- Support Vector Machine (SVM)
- Gaussian Naive Bayes (NB)
- K-Nearest Neighbors (KNN)

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. It contains 569 instances of various features computed from breast mass biopsies.

### Features

- radius_mean
- perimeter_mean
- concave_points_mean
- ... (additional features in the dataset)

### Target

- `diagnosis`: Binary variable indicating whether the tumor is malignant (`M`) or benign (`B`). This has been converted to 1 for malignant and 0 for benign for the purpose of modeling.

## Files

- `data.csv`: The dataset file.
- `newbreast-cancer-prediction-using-machine-learning.ipynb`: The main Python script that includes data preprocessing, model training, evaluation, and visualization.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the necessary dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
