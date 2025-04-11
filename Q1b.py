# Import Libaries
import time 
import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt

#Define constants 
Year = 365.25 * 24 * 60 ** 2 # s
AU = c.au.value # m 
G = c.G.value # m^3 kg^-1 s^-2
M_sun = c.M_sun.value # kg
r_aphelion = 5.2e12 # m - Furtherest distance away from the sun
v_aphelion = 880 # m/s -  Speed at aphelion (will be only in y-direction)

'''
Universal Comet Class
'''
# Setting up the Comet
class Comet:
    # Initialize the Comet with properties
    # initial_pos and initial_vel are the initial position and velocity vectors [x,y] and [vx,vy]
    def __init__(self, name, initial_pos, initial_vel, mass):
        self.name = name
        self.f_vec = np.array(initial_pos + initial_vel, dtype=np.float64)  # [x, y, vx, vy]
        self.history = [self.f_vec.copy()]
        self.energies = []  # Store KE, PE, and Total Energy
        self.dt_values = []  # Store time-steps
        self.time_values = []  
        self.mass = mass

    # Compute the acceleration of the comet
    def acceleration(self, f):
        x, y, vx, vy = f
        r2 = x**2 + y**2
        r = np.sqrt(r2)
        if r > 1e-12:
            ax = -G * M_sun * x / (r2 * r)
            ay = -G * M_sun * y / (r2 * r)
        else:
            ax = ay = 0.0
        return np.array([vx, vy, ax, ay], dtype=np.float64)
    
    # Compute the energy of the comet
    def compute_energy(self):
        x, y, vx, vy = self.f_vec
        r = np.sqrt(x**2 + y**2)
        v2 = vx**2 + vy**2

        KE = 0.5 * self.mass * v2                            # Kinetic Energy
        PE = -G * self.mass * M_sun / r if r > 1e-12 else 0  # Potential Energy - prevent division by 0
        E = KE + PE                                          # Total Energy

        self.energies.append([KE, PE, E])  # Store energy values as a list

'''
Universal Simulation Class 
Extensions to this are below to use other methods
Options to plot all graphs together or individually
'''
# The main class to run the simulation
class Runsimulation:
    def __init__(self, comet, dt_init, end_time, error_tolerance, step_method=None):
        self.comet = comet
        self.dt = dt_init
        self.end_time = end_time
        self.error_tolerance = error_tolerance
        self.step_method = step_method if step_method else self.rk4_step  # Default to RK4

    def rk4_step(self, f, dt):
        error = 0.0 
        k1 = dt * self.comet.acceleration(f)
        k2 = dt * self.comet.acceleration(f + 0.5 * k1)
        k3 = dt * self.comet.acceleration(f + 0.5 * k2)
        k4 = dt * self.comet.acceleration(f + k3)
        f_next = f + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return f_next, f_next, error # Need to ensure 3 values outputted for later use
    
    def adaptive_step(self, body, second_input=None):
        # Adaptive step method can take second input (useful later on and optional)
        while True:
            f_orig = body.f_vec.copy()
            # Use the step method (RK4 or RKF45). Only need one value
            f_full, _, _ = self.step_method(f_orig, self.dt)
            f_half1, _, _ = self.step_method(f_orig, self.dt / 2.0)
            f_half2, _, _ = self.step_method(f_half1, self.dt / 2.0) # Two half steps 
            # Calculate the difference between full and half 
            err_vec = f_full - f_half2
            err = np.linalg.norm(err_vec) / 30.0  # RK4 correction factor

            if err > self.error_tolerance:
                self.dt *= 0.5  # Reduce step size if error is too large
                continue
            else:
                scale = (self.error_tolerance / (err + 1e-10)) ** 0.2
                # Change capped at: 0.5 < dt < 2
                dt_new = self.dt * np.clip(scale, 0.5, 2.0)

                # Accept new state
                body.f_vec, _, _ = self.step_method(f_orig, dt_new)  # Unpacking
                body.history.append(body.f_vec.copy())  # Append new state
                body.compute_energy()  # Record energy at each step

                # Modify step based on second_input (if provided)
                if second_input is not None:
                    dt_new *= second_input  # Example use case

                return dt_new  # Return updated dt

    def run_simulation(self): # Runs simulation with chosen method 
        current_t, steps_taken = 0.0, 0
        start_time = time.time()
        # Run for specified amount of time 
        while current_t < self.end_time:
            self.dt = self.adaptive_step(self.comet)
            current_t += self.dt
            self.comet.time_values.append(current_t)
            self.comet.dt_values.append(self.dt)
            steps_taken += 1

        runtime = time.time() - start_time
        print(f"Simulation finished in {runtime:.2f} s with {steps_taken} steps.") # Used for comparisons 

    #Code for plotting all the graphs together
    def plot_all(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        # Trajectory with Speed Color Map
        data = np.array(self.comet.history)
        x, y, vx, vy = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        speed = np.sqrt(vx**2 + vy**2) / 1000.0  # Convert to km/s
        scatter = axes[0].scatter(x / AU, y / AU, c=speed, cmap='plasma', s=10, label="Halley's Comet")
        axes[0].plot(0, 0, '*', color='darkorange', label='Sun', markersize=15)
        cbar = fig.colorbar(scatter, ax=axes[0])
        cbar.set_label("Speed (km/s)")
        axes[0].set_xlabel("x (AU)")
        axes[0].set_ylabel("y (AU)")
        axes[0].legend(loc='upper right')
        axes[0].set_title("Trajectory and Speed")

        # Energy Plot 
        if self.comet.energies and self.comet.time_values:
            energies = np.array(self.comet.energies)
            time_values = np.array(self.comet.time_values) / Year
            min_length = min(len(time_values), len(energies))
            energies, time_values = energies[:min_length], time_values[:min_length]

            KE, PE, E_total = (energies[:, 0] / Year) / 1e16, (energies[:, 1] / Year) / 1e16, (energies[:, 2] / Year) / 1e16
            axes[1].plot(time_values, KE, label="Kinetic Energy", color="blue")
            axes[1].plot(time_values, PE, label="Potential Energy", color="green")
            axes[1].plot(time_values, E_total, label="Total Energy", color="black", linestyle="dashed")
            axes[1].set_xlabel("Time (Years)")
            axes[1].set_ylabel("Energy (x $10^{16}$ J)")
            axes[1].legend(loc='upper right')
            axes[1].set_title("Energy Over Time")

        # Time Step Plot
        time_years = np.array(self.comet.time_values) / Year
        dt_years = np.array(self.comet.dt_values) / Year
        axes[2].plot(time_years, dt_years, label="Adaptive Time Step", color="purple")
        axes[2].set_xlabel("Time (Years)")
        axes[2].set_ylabel("Time Step (Years)")
        axes[2].legend(loc='upper right')
        axes[2].set_title("Time Step Adaptation")

        # Finalize Layout
        plt.tight_layout()
        plt.show()

'''
Runge-Kutta Fehlberg 45 Method
'''
# Comet class remains the same - only changing the method used 
class RunsimulationRKF45(Runsimulation): # Inherit parameters from above 
    def __init__(self, comet, dt_init, end_time, error_tolerance):
        super().__init__(comet, dt_init, end_time, error_tolerance, step_method = self.rkf45_step)

    #Define the new method - RKF45
    def rkf45_step(self, f, dt):
        # Computes extra terms and uses Fehlberg coefficients 
        k1 = dt * self.comet.acceleration(f)
        k2 = dt * self.comet.acceleration(f + k1 * 1/4)
        k3 = dt * self.comet.acceleration(f + k1 * 3/32 + k2 * 9/32)
        k4 = dt * self.comet.acceleration(f + k1 * 1932/2197 - k2 * 7200/2197 + k3 * 7296/2197)
        k5 = dt * self.comet.acceleration(f + k1 * 439/216 - k2 * 8 + k3 * 3680/513 - k4 * 845/4104)
        k6 = dt * self.comet.acceleration(f - k1 * 8/27 + k2 * 2 - k3 * 3544/2565 + k4 * 1859/4104 - k5 * 11/40)
        # Compute 4th and 5th-order estimates
        f4 = f + k1 * 25/216 + k3 * 1408/2565 + k4 * 2197/4104 - k5 * 1/5
        f5 = f + k1 * 16/135 + k3 * 6656/12825 + k4 * 28561/56430 - k5 * 9/50 + k6 * 2/55
        # Compute error estimate
        error = np.linalg.norm(f5 - f4)

        return f4, f5, error

'''
Velocity-Verlet Method
'''
# Comet class remains the same - only changing the method used 
class RunVerlet(Runsimulation): # Inherit parameters from adaptive RK4 method 
    def __init__(self, comet, dt_init, end_time):
        # Verlet method does not need a tolerance so can be set to None 
        super().__init__(comet, dt_init, end_time, error_tolerance = None, step_method=self.verlet_step)

    # Define verlet method 
    def verlet_step(self, f, dt):
        x, y, vx, vy = f
        # Compute acceleration at current position
        a_current = self.comet.acceleration(f)[2:]  # [ax, ay]
        # Update positions and temporaily store them
        x_new = x + vx * dt + 0.5 * a_current[0] * dt**2
        y_new = y + vy * dt + 0.5 * a_current[1] * dt**2
        f_temp = np.array([x_new, y_new, vx, vy], dtype=np.float64) 
        # Acceleration at the new position
        a_next = self.comet.acceleration(f_temp)[2:]  
        # Update velocities
        vx_new = vx + 0.5 * (a_current[0] + a_next[0]) * dt
        vy_new = vy + 0.5 * (a_current[1] + a_next[1]) * dt

        return np.array([x_new, y_new, vx_new, vy_new], dtype=np.float64), None, None  

    # Need to adapt the adaptive_step() to not use error estimate 
    def adaptive_step(self, body):
        f_new, _, _ = self.step_method(self.comet.f_vec, self.dt)
        self.comet.f_vec = f_new
        self.comet.history.append(f_new.copy())
        self.comet.compute_energy()
        # Fixed step size, so dt remains unchanged
        return self.dt  


#Running Adaptive RK4 Method 
comet = Comet(name = "Halley", 
              initial_pos = [r_aphelion, 0], 
              initial_vel = [0, v_aphelion], 
              mass=2.2e14)
sim_rk4 = Runsimulation(comet, 
                        dt_init = Year / 100,   # Initial time interval 
                        end_time = Year * 76,   # Total time to run simulation for 
                        error_tolerance = 1000)         # In m/s not km/s! Lower = more regular intervals 

sim_rk4.run_simulation()  # Uses RK4 by default
sim_rk4.plot_all()

# Running Adaptive RKF45 Method
'''
comet = Comet(name = "Halley", 
              initial_pos = [r_aphelion, 0], 
              initial_vel = [0, v_aphelion], 
              mass=2.2e14)
sim_rkf45 = RunsimulationRKF45(comet, 
                        dt_init = Year / 100,   # Initial time interval 
                        end_time = Year * 76,   # Total time to run simulation for 
                        error_tolerance = 1000)         # In m/s not km/s! Lower = more regular intervals 

sim_rkf45.run_simulation()  # Uses RKF45 by default
sim_rkf45.plot_all()
'''

# Running Velocty - Verlet Method
'''
comet = Comet("Halley", 
              [r_aphelion, 0], 
              [0, v_aphelion], 
              mass = 2.2e14)
simverlet = RunVerlet(comet, 
                dt_init=Year / 100, 
                end_time=Year * 80)

simverlet.run_simulation()      # Uses Verlet integration
simverlet.plot_all()
'''