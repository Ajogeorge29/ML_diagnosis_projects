
# Kyphosis Disease Classification

This repository contains a Python script that classifies the presence of kyphosis (a spinal condition) based on patient data, using machine learning techniques.

## Dataset
The dataset used for this project is `kyphosis.csv`, containing:
- **Kyphosis**: Whether kyphosis is present (1) or absent (0).
- **Age**: The age of the patient.
- **Number**: The number of vertebrae involved in the surgery.
- **Start**: The starting vertebra involved in the operation.

## Files
- `Machine Learning for Kyphosis Disease Classification.py`: Python script for data analysis and kyphosis classification.
- `kyphosis.csv`: Dataset with patient details.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Use
1. Install the required dependencies: `pip install -r requirements.txt`.
2. Run the Python script to load the dataset, perform exploratory data analysis, and train a RandomForest model to classify kyphosis.
3. View the classification performance using confusion matrix and classification report.
