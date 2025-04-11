# Import Libaries
import time 
import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt

# Define constants 
Year = 365.25 * 24 * 60 ** 2 # s
AU = c.au.value # m 
G = c.G.value # m^3 kg^-1 s^-2
M_sun = c.M_sun.value # kg
r_aphelion = 5.2e12 # m - Furtherest distance away from the sun
v_aphelion = 880 # m/s -  Speed at aphelion (will be only in y-direction)

'''
Main Program below for simple Runge-Kutta fixed time-step method
'''
# Gravitational acceleration due to the Sun
def acceleration(f):
    x, y, vx, vy = f # State vector in SI units 
    r2 = x * x + y * y
    r = np.sqrt(r2)
    # Avoid division by Zero in event it is at the origin 
    if r > 1e-12:
        ax = -G * M_sun * x / (r2 * r)
        ay = -G * M_sun * y / (r2 * r)
    else:
        ax = ay = 0.0
    return np.array([vx, vy, ax, ay], dtype=np.float64)
    
# One Runge-Kutta step
# f is state vector, dt is time step
def rk4_step(f, dt):
    k1 = dt * acceleration(f)
    k2 = dt * acceleration(f + 0.5 * k1)
    k3 = dt * acceleration(f + 0.5 * k2)
    k4 = dt * acceleration(f + k3)
    return f + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

# Integration of the orbit
def run_simulation(r_aphelion, v_aphelion, end_years):
    x, y = [], []
    current_t, steps_taken = 0.0, 0.0
    start_time = time.time()
    # Initial state: [x, y, vx, vy]
    f = np.array([r_aphelion, 0, 0, v_aphelion], dtype=np.float64)

    step = Year / 50          # Fixed step size 
    end_time = end_years * Year  # Total time to simulate

    while current_t < end_time:
        f = rk4_step(f, step)  # Update state vector

        # Store results
        x.append(f[0])
        y.append(f[1])

        current_t += step
        steps_taken += 1

    runtime = time.time() - start_time
    return x, y, steps_taken, end_time, runtime

x, y, steps_taken, end_time, runtime = run_simulation(r_aphelion,
                                                   v_aphelion, 
                                                   end_years = 76)

print(f"Total steps taken were {steps_taken}, with a total time of {end_time / Year} years. Runtime was {runtime:.2f} seconds.")

# Plot orbit
plt.figure(figsize=(14, 10))
plt.plot(np.array(x) / AU, np.array(y) / AU, '.', label="Halley's Comet", markersize=2, color='blue')
plt.plot(0, 0, '*', color='darkorange', markersize=15, label="Sun")
plt.xlabel('x (AU)'), plt.ylabel('y (AU)')
plt.axis('equal')
plt.legend()
plt.show()