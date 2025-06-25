# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:04:03 2025

@author: Morteza
"""

import pandas as pd
import seaborn as sns

# Load the famous Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows
print(iris.head())

# %%

import matplotlib.pyplot as plt

# Scatter plot of sepal length vs sepal width
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

#%%

import seaborn as sns

# Scatter plot using Seaborn (which also adds colors and legend automatically)
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.title('Sepal Length vs Sepal Width')
plt.show()

#%%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select features and target variable
X = iris[['sepal_length', 'sepal_width', 'petal_width']]
y = iris['petal_length']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

results = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred})
print(results.head())

#%%

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Prepare dataset
X = iris.drop(columns=['species'])
y = iris['species']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# Create DataFrame to show y_test and y_pred side by side
comparison_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# Display the dataframe
print(comparison_df)

#%%

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris  # Import to load iris dataset

# Load the iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target labels to the dataframe
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Select features and target variable
X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal width (cm)']]
y = iris_df['species']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical
y_cat = pd.Categorical(y)
y_encoded = to_categorical(y_cat.codes)

# Create neural network model
model = Sequential([
    Dense(10, activation='relu', input_shape=(X.shape[1],)),  # Input layer
    Dense(10, activation='relu'),  # Hidden layer
    Dense(y_encoded.shape[1], activation='softmax')  # Output layer
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(pd.Categorical(y_train).codes), epochs=200, verbose=1)

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, to_categorical(pd.Categorical(y_test).codes), verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_encoded = pd.Categorical(y_test).codes

# Display y_pred and y_test side by side
results = pd.DataFrame({'y_test': y_test_encoded, 'y_pred': y_pred})
print("\nPredictions vs True Labels:")
print(results.head(10))  # Show first 10 for readability
