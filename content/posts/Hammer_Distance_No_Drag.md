---
date: '2024-05-25T00:00:00-05:00'
draft: False
title: 'Hammer Throw Distance Calculation when No Air Drag'
author: 'Jorgen Bergstrom'
tags: ['Hammer Throw', 'Python']
categories: ['Hammer Throw']
---

## Introduction

Welcome to the first article in my series on calculating the distance a projectile will travel based on its initial velocity and angle. Using hammer throwing, an Olympic track and field sport, as an example, I will demonstrate these calculations. However, the principles apply equally well to baseball and other sports.

For simplicity, I will focus solely on predicting the distance the hammer will fly, leaving the complete flight path for later discussion. In this series, we will explore both Newtonian mechanics and various machine learning algorithms. Our goal is to determine the most effective method for making these predictions.

## Calculation when no Air Drag

It is easy to calculate the distance a projectile will fly when there is no air drag. In this case the horizontal and vertical motions can be decoupled, and the distance the ball will fly is given by $v_0 t \cos(\alpha)$, where the total flight time is given by $2 v_0 \sin(\alpha) / g$. In summary, the total distance will be given by the following equation and Python code:

$$d = \frac{v^2}{g} \sin(2\alpha)$$

```python
import math
v = 19.0                     # velocity [m/s]
angle = math.radians(40.0)
g = 9.81                     # gravity [m/s^2]
v_vert = math.sin(angle) * v # vertical velocity
# v = a * t, so t = v / a
# v0 - g*t = 0  => t = v0 / g
t = 2 * v_vert / g
print(f"time = {t:.3f}")
d = math.cos(angle) * v * t
print(f"solution 1: distance={d:.3f} m")
d = math.sin(2*angle) * v**2 / g
print(f"solution 2: distance={d:.3f} m")
```

The max distance occurs when the angle is 45°.

Note that GPT-4o also gives this equation when asked the question "How far does a projectile fly if its initial velocity is V and initial angle is alpha? Ignore the air drag."
