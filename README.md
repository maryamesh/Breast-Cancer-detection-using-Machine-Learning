[![image](https://github.com/user-attachments/assets/92e0044b-3a3c-4675-ae37-1b387636303b)](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.nyit.edu%2Fnews%2Ffeatures%2Frecognizing_breast_cancer_awareness_month&psig=AOvVaw0d2OqJRQOmaenI5FEwyN0r&ust=1722813382827000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCPCrn5D62YcDFQAAAAAdAAAAABAJ)


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
`pip install numpy pandas matplotlib seaborn scikit-learn`
```
# Data Preprocessing
Load the dataset and inspect the first few rows.
Convert the diagnosis column to binary values (1 for malignant, 0 for benign).
Drop unnecessary columns and set the index.

# Exploratory Data Analysis (EDA)
Visualize the correlation matrix using a heatmap to understand the relationships between features.
Model Training and Evaluation
The following steps outline the process of training and evaluating the models:
1. Split the dataset into training and test sets.
2. Train multiple models using cross-validation to find the best performing model.
3. Evaluate the models on the test set and compare their performance using accuracy, confusion matrix, and classification report.

# Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any feature requests, bug fixes, or improvements.

# License
This project is licensed under the MIT License.

# Acknowledgments
The dataset is publicly available at the UCI Machine Learning Repository.
Thanks to the open-source community for providing tools and libraries used in this project.

