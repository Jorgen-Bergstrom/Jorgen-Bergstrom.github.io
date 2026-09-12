---
date: '2024-05-25T00:00:00-05:00'
draft: false
title: 'Physics Calculation of Hammer Throw Distance with Air Drag'
author: 'Jorgen Bergstrom'
tags: ['Hammer Throw', 'Physics', 'Python']
categories: ['Hammer Throw']
---

## Physics-Based Theory and Numerical Implementation
The weight of the hammer ball is 7.26 kg, the material is steel and therefore the ball radius is given by $r = (3m/(4\pi \rho_s))^{1/3}$. The drag force from the air resistance is $F_d = \rho_a v^2 C_d A$, where $A$ is the cross-sectional area. In summary, Newton’s equation in the horizontal and vertical directions can be written in incremental form: $$ \displaystyle \Delta v_x = – \frac{F_d(v)}{m} \frac{v_x}{v} \Delta t$$ $$ \displaystyle \Delta v_y = – \left[ \frac{F_d(v)}{m} \frac{v_y}{v} + g \right] \Delta t$$

<br>
These equations can be solved numerically using the following Python code.

```python
import math
import matplotlib.pyplot as plt
angle = 44        # angle in deg
v0 = 28.0         # initial velocity [m/s]
g = 9.81          # gravity [m/s^2]
rho_air = 1.204   # air density kg/m^3
CD = 0.45         # drag coefficient for a sphere
mass = 7.26       # ball mass [kg]
rho_steel = 7500  # ball density [kg/m^3]
dt = 0.001        # time increment size
ball_radius = (3 * mass / (4 * math.pi * rho_steel))**(1/3)
print(f"Ball diameter={2*ball_radius}")
A_ball  = math.pi * ball_radius**2
x = [0.0]
y = [0.0]
vx = math.cos(angle * math.pi/180) * v0
vy = math.sin(angle * math.pi/180) * v0
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
print(f"distance thrown = {x[-1]:.2f}")
plt.plot(x,y)
plt.axis('equal');
plt.ylim(bottom=1);
```

<br>
The output from running that code is:

```
Ball diameter=0.12273231093802761
distance thrown = 77.72
```

<img src="/Hammer_Img03.png">



This Python code implementation is very simple and is based on the forward Euler integration and therefore requires small time steps. It is easy to implement but perhaps not the most elegant solutions.


## ChatGPT-4 Solution
I used the following prompt in ChatGPT-4: `"Can you write Python code that calculate how how far a ball will fly if its initial velocity is V and the initial angle is A. Use Newton's equation to perform the calculation and include the influence of air drag."` 

The following code was generated:
```python
import math
import numpy as np
from scipy.integrate import solve_ivp
def equations_of_motion(t, y, params):
    Vx, Vy = y[2], y[3]
    v = math.sqrt(Vx**2 + Vy**2)
    g, rho, Cd, A, m = params
    
    # Drag forces
    Fd_x = -0.5 * Cd * rho * A * Vx * v
    Fd_y = -0.5 * Cd * rho * A * Vy * v
    
    # Equations of motion
    dVx_dt = Fd_x / m
    dVy_dt = (Fd_y / m) - g
    
    return [Vx, Vy, dVx_dt, dVy_dt]

def calculate_trajectory(V, A, D, m):
    g = 9.81  # Gravity in m/s^2
    rho = 1.225  # Air density in kg/m^3
    Cd = 0.47  # Drag coefficient for a sphere
    A = math.pi * (D / 2)**2  # Cross-sectional area in m^2
    
    # Initial conditions
    A_rad = math.radians(A)
    Vx0 = V * math.cos(A_rad)
    Vy0 = V * math.sin(A_rad)
    y0 = [0, 0, Vx0, Vy0]  # Initial positions and velocities
    # Parameters tuple
    params = (g, rho, Cd, A, m)
    # Time span for the simulation
    t_span = (0, 10)  # Assume the ball will land within 10 seconds
    t_eval = np.linspace(0, 10, 1000)  # Evaluation times
    # Solve the differential equations
    sol = solve_ivp(equations_of_motion, t_span, y0, args=(params,), t_eval=t_eval, rtol=1e-8, atol=1e-8)
    # Extract positions
    x = sol.y[0]
    y = sol.y[1]
    # Find where the ball hits the ground (y=0)
    ground_hits = np.where(y <= 0)[0]
    if len(ground_hits) > 0:
        range_distance = x[ground_hits[0]]
    else:
        range_distance = x[-1]  # If it never hits the ground in the interval, take the last position
    return range_distance

# Example usage
initial_velocity = 50  # in meters per second
launch_angle = 45      # in degrees
diameter = 0.1         # in meters
mass = 0.145           # in kilograms (typical mass of a baseball)
range_distance = calculate_trajectory(initial_velocity, launch_angle, diameter, mass)
print(f"The ball will fly approximately {range_distance:.2f} meters.")
```

I really like this code because it uses the `solve_ivp()` function from scipy.integrate. The problem here is that it does not work. It always returns a distance of 0, which is clearly not right. I reviewed the code and found 2 errors. On line 20, the variable `A` should be called something else since `A` is already a local variable. And on lines 43-44 the first row should be removed to make sure that the test for `ground_hits` works.


## GPT-4o Solution

I used the following prompt in GPT-4o: 
```"Can you write Python code that calculate how how far a ball will fly if its initial velocity is V and the initial angle is A. Use Newton's equation to perform the calculation and include the influence of air drag."```

The following code was generated (note that I changed some of the variable values to be similar to what I had selected):

```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
rho = 1.204  # Density of air (kg/m^3)
Cd = 0.45  # Drag coefficient of a sphere
r = 0.061366  # Radius of the ball (m)
A = np.pi * r**2  # Cross-sectional area (m^2)
m = 7.26  # Mass of the ball (kg)
# Function to compute the drag force

def drag_force(v):
    return 0.5 * Cd * A * rho * v**2

# Function to perform the projectile motion calculation with air drag
def calculate_trajectory(V, angle, dt=0.01):
    # Initial conditions
    theta = np.radians(angle)
    vx = V * np.cos(theta)
    vy = V * np.sin(theta)
    x, y = 0.0, 0.0
    
    # Lists to store trajectory points
    x_points = [x]
    y_points = [y]
    
    # Time loop
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        Fd = drag_force(v)
        ax = -Fd * vx / (m * v)
        ay = -g - (Fd * vy / (m * v))
        
        # Update velocities
        vx += ax * dt
        vy += ay * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        
        # Store trajectory points
        x_points.append(x)
        y_points.append(y)
    
    return x_points, y_points

# Initial conditions
initial_velocity = 28.0  # Initial velocity in m/s
initial_angle = 44.0  # Initial angle in degrees
# Calculate the trajectory
x_points, y_points = calculate_trajectory(initial_velocity, initial_angle)
# Plot the trajectory
plt.figure(figsize=(10, 5))
plt.plot(x_points, y_points)
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Projectile Motion with Air Drag')
plt.grid(True)
plt.show()

# Print the range of the projectile
print(f'The ball will fly approximately {x_points[-1]:.2f} meters.')
```


Here are the results:

```The ball will fly approximately 77.61 meters.```

<img src="/Hammer_Img04.png">

This solutions works and is very similar to the simple solution that I created above.


## One GPT-4o Calculation
Finally, I tried the to ask the specific question:
```I am throwing a ball with a diameter of 0.1227 m, a mass of 7.26 kg, with an initial velocity of 28 m/s, and an initial angle of 44 degrees. What distance will the ball fly before hitting the ground?```

The returned answer was:
```With a mass of 7.26 kg, a diameter of 0.1227 m, an initial velocity of 28 m/s, and an initial angle of 44 degrees, the ball will fly approximately 77.52 meters before hitting the ground.```

This is correct. I’m impressed. By exploring the answer I can see that GPT used Python code to come up with this answer. Note, however, that if I repeat that question but don’t mention anything about the mass then GPT gives an answer of 30 m, which is incorrect and bad. If the mass is not specified then the problem cannot be solved, and I was not informed about that. That is certainly a important limitation of this LMM.

<br>
{{< admonition type=note >}}
In summary: It is fun to solve this problem by using fundamental equations and then coding them up in Python. That works, but is time consuming. The GPT-4o solution is very similar and much quicker to perform!
{{< /admonition >}}
