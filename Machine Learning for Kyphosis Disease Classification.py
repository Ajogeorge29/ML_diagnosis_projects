
# TASK 1: UNDERSTAND THE PROBLEM STATEMENT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot

jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

# TASK 2: IMPORT LIBRARIES AND DATASETS
Kyphosis_df = pd.read_csv("kyphosis.csv")

# Explore the dataset
print(Kyphosis_df.head())
print(Kyphosis_df.tail(5))
Kyphosis_df.info()  # Corrected method call

# TASK 3: PERFORM DATA VISUALIZATION
from sklearn.preprocessing import LabelEncoder
LabelEncoder_y = LabelEncoder()
Kyphosis_df['Kyphosis'] = LabelEncoder_y.fit_transform(Kyphosis_df['Kyphosis'])

# Plot correlation heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(Kyphosis_df.corr(), annot=True)

# Pairplot visualization
sns.pairplot(Kyphosis_df, hue='Kyphosis')

# Plot count of samples in each class
sns.countplot(x=Kyphosis_df['Kyphosis'], label="Count")

# TASK 4: SPLIT DATA AND MODELING

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Define features and target
X = Kyphosis_df.drop('Kyphosis', axis=1)
y = Kyphosis_df['Kyphosis']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
RandomForest = RandomForestClassifier()
RandomForest.fit(X_train, y_train)

# Predict the test set results
y_predict_test = RandomForest.predict(X_test)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

# Print classification report
print(classification_report(y_test, y_predict_test))

# Show the plots
plt.show()
