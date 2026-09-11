---
date: '2024-05-29T00:00:00-05:00'
draft: false
title: 'Neural Network Analysis of Hammer Throw Distance using PyTorch'
author: 'Jorgen Bergstrom'
tags: ['Hammer Throw', 'Neural Networks', 'PyTorch', 'Python']
categories: ['Hammer Throw']
---

## Summary
In this example I have used a “Sequential” neural network model to solve the regression problem of how far a hammer will fly given an initial velocity and angle. A sequential model is linear stack of layers, and allows you to create a Neural Network (NN) by simply adding layers sequentially. Both PyTorch and Keras are popular frameworks for building these NN models. Compared to Kears, the PyTorch approach is exposes more details which makes it more flexible and suitable for complex and dynamic models.  Here is a PyTorch implementation of the hammer throw distance calculation:
```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, mean_absolute_error 
# Load the data
X = np.load('data_X.npy')
Y = np.load('data_Y.npy')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define the model
model = nn.Sequential(
    nn.Linear(2, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)
# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
# Training parameters
n_epochs = 10000
batch_size = 100
batch_start = torch.arange(0, len(X_train), batch_size)
# Hold the best model
best_mse = np.inf
best_weights = None
# Training loop
for epoch in range(n_epochs):
    model.train()
    for start in batch_start:
        # Take a batch
        X_batch = X_train[start:start+batch_size]
        y_batch = y_train[start:start+batch_size]
        
        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    # Evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
 
# Restore model and return best accuracy
model.load_state_dict(best_weights)
predictions = model(X_test).detach().numpy()
print(f"Mean absolute error={mean_absolute_error(y_test, predictions):.3f}")
plt.scatter(X_test[:,1], y_test, c='r')
plt.scatter(X_test[:,1], predictions, c='b')
```
The forward pass, backward pass, and optimization step is shown for each batch and epoch loop. Running this code produces the following results. Note that the accuracy of the predictions are similar to the other NN predictions.
```text
Mean absolute error=0.921
```
![](/Hammer_Img19.png)
