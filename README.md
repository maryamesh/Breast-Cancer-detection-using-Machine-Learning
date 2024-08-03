# Breast Cacer detection using ML algorithms

This project uses machine learning techniques to detect breast cancer based on various features extracted from breast mass biopsies. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.


## Project Overview

The goal of this project is to build and evaluate several classification models to predict whether a breast mass is malignant or benign. The models used in this project include:

- Decision Tree Classifier (CART)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes (NB)
- K-Nearest Neighbors (KNN)

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. It contains 569 instances of various features computed from breast mass biopsies.

#### Kaggle Dataset source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

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


## Usage
Clone the Repository
git clone https://github.com/maryamesh/breast-cancer-detection.git
cd breast-cancer-detection

##Run the Script
Make sure you have the data.csv file in the same directory as breast_cancer_detection.py. Then, run the script:
python breast_cancer_detection.py




# Model Evaluation
The project includes cross-validation and performance evaluation for each model. Here are some key results:

Support Vector Machine (SVM)
Cross-Validation Accuracy: 0.9185
Test Set Accuracy: 0.8860
Confusion Matrix:
[[73  2]
 [11 28]]
Classification Report:
              precision    recall  f1-score   support

         0       0.87      0.97      0.92        75
         1       0.93      0.72      0.81        39

  accuracy                           0.89       114
 macro avg       0.90      0.85      0.86       114
