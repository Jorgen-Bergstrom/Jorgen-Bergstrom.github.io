---
date: '2024-06-01T00:00:00-05:00'
draft: false
title: 'Can ChatGPT Predict Hammer Throw Distance?'
author: 'Jorgen Bergstrom'
tags: ['Hammer Throw', 'LLM', 'ChatGPT', 'Python']
categories: ['Hammer Throw']
---

## Introduction
In my previous articles in this series I have shown how to calculate how far spherical object will fly based on its initial velocity and angle. I used both physics and machine learning methods to solve the problem. In this article I will examine how much of this can be completely automated using a large language model like ChatGPT 4o.


##### Prompt:
I am interested in predicting the distance as a function of the angle and velocity. What regression model do you recommend that I use? Can you provide python code for the analysis?

<br>


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (assuming the data is saved in a CSV file named 'data.csv')
data = pd.read_csv('data_XY.csv', header=None, names=['Angle', 'Velocity', 'Distance'])

# Check for missing values
print(data.isnull().sum())

# Split the data into features (X) and target (y)
X = data[['Angle', 'Velocity']]
y = data['Distance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Distance')
plt.show()

# Display the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
```
