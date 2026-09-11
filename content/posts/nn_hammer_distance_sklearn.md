---
date: '2024-05-28T00:00:00-05:00'
draft: false
title: 'Neural Network Analysis of Hammer Throw Distance using Scikit-Learn'
author: 'Jorgen Bergstrom'
tags: ['Hammer Throw', 'Neural Networks', 'Scikit-Learn', 'Python']
categories: ['Hammer Throw']
---

## Summary
The `MLPRegressor` in scikit-learn is a powerful tool for performing regression tasks using a multi-layer perceptron (MLP), which is a type of artificial neural network. It is a supervised learning algorithm that learns a function that maps input data to continuous output values. It can model complex relationships between features and the target variable.
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, mean_absolute_error 
X = np.genfromtxt('data_X.csv', delimiter=',')
Y = np.genfromtxt('data_Y.csv', delimiter=',')
# Split the data into training and testing sets. test_size=0.3 indicates that 30%
# of the data will be used for testing, and 70% for training.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# Transform the input X data by subtracting the mean and dividing with the standard dev.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# MLP = multi layer perceptron
# This sets up the whole Neural Network model
mlp = MLPRegressor(hidden_layer_sizes=(50, 25, 12, 6, 3), activation='relu', solver='adam', max_iter=10000, verbose=False)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(f"Mean absolute error={mean_absolute_error(y_test, predictions):.3f}")
plt.scatter(X_test[:,1], y_test, c='r')
plt.scatter(X_test[:,1], predictions, c='b')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Distance (m)')
```
The output from running this code is:
```text
Mean absolute error=0.727
```
![](/Hammer_Img14.png)
This example shows how simple it can be to set up and run a Neural Network (NN) model. The results show that the model predictions are decent, but not as good as a 4th order polynomial regression model.
