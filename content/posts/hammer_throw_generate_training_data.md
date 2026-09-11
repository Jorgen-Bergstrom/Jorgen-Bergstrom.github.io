---
date: '2024-05-26T00:00:00-05:00'
draft: false
title: 'Hammer Throw: Generate Training Data'
author: 'Jorgen Bergstrom'
tags: ['Hammer Throw', 'Data', 'Python']
categories: ['Hammer Throw']
---

## Training Data
All machine learning algorithms require data for training the model. In this example, we can use physics calculations to generate a dataset with two input variables: velocity and angle, and one output variable: flight distance. The following Python code creates an input file named ‘data_X.csv’ containing the input variables, and a results file with the flight distance. These files will be used in subsequent machine learning demonstrations.
```python
import math
import numpy as np
import csv
def calc_distance(angle, velocity):
    # fixed input parameters
    g = 9.81          # gravity [m/s^2]
    rho_air = 1.204   # air density kg/m^3
    CD = 0.45         # drag coefficient for a sphere
    mass = 7.26       # ball mass [kg]
    rho_steel = 7500  # ball density [kg/m^3]
    dt = 0.001        # time increment size
    # calculations
    ball_radius = (3 * mass / (4 * math.pi * rho_steel))**(1/3)
    A_ball  = math.pi * ball_radius**2
    x = [0.0]
    y = [0.0]
    vx = math.cos(angle * math.pi/180) * velocity
    vy = math.sin(angle * math.pi/180) * velocity
    while True:
        v = math.sqrt(vx**2 + vy**2)
        F_drag = 0.5 * rho_air * v**2 * CD * A_ball
        dvx = -(F_drag/mass) * (vx/v) * dt
        dvy = -((F_drag/mass) * (vy/v) + g) * dt
        vx = vx + dvx
        vy = vy + dvy
        if (y[-1] + vy * dt < 0):
            break
        x.append( x[-1] + vx * dt )
        y.append( y[-1] + vy * dt )
    return x[-1]
N = 500 # number of data points to generate
rng = np.random.RandomState(1234)
X_ang = rng.uniform(0, 45, N)
X_vel = rng.uniform(0, 50, N)
X = np.zeros((N,2))
Y = np.zeros(N)
for i in range(N):
    X[i,0] = X_ang[i]
    X[i,1] = X_vel[i]
    Y[i] = calc_distance(X_ang[i], X_vel[i])
# save to csv-files
with open('data_X.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(X)
with open('data_Y.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(Y)
```
