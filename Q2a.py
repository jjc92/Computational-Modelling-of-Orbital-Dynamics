# Import Libaries 
import time 
import numpy as np
import astropy.constants as c
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

# Define constants
Year = 365.25 * 24 * 60 ** 2 # s
AU = c.au.value # m 
G = c.G.value # m^3 kg^-1 s^-2
M_sun = c.M_sun.value # kg
r_aphelion = 5.2e12 # m - Furtherest distance away from the sun
v_aphelion = 880 # m/s -  Speed at aphelion (will be only in y-direction)

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


'''

This is now new code from below - code before was from previous question but is inherited here.

'''

class Planet(Comet):
    def __init__(self, name, a, mass):
        # Inherit from Comet and add semi-major axis
        super().__init__(name, [a, 0.0], [0.0, np.sqrt(G * M_sun / a)], mass)

class SolarSystemSimulation(Runsimulation):
    def __init__(self, planets, dt_init, end_time, error_tolerance, step_method=None):
        # Inherti from Runsimulation and add multiple planets
        self.planets = planets
        self.step_method = step_method or self.rk4_step  # Default to RK4
        super().__init__(planets[0], dt_init, end_time, error_tolerance)  # Use first planet for setup

    # Redefine adaptive step to handle multiple planets
    def run_simulation(self):
        current_t, steps_taken = 0.0, 0
        start_time = time.time()
        while current_t < self.end_time:
            dt_list = []  # Store time steps for each planet
            for planet in self.planets:
                dt_list.append(self.adaptive_step(planet))  # Compute dt for each planet
            self.dt = min(dt_list)  # Use smallest dt to maintain accuracy
            current_t += self.dt
            steps_taken += 1
        runtime = time.time() - start_time
        print(f"Simulation finished in {runtime:.2f} s with {steps_taken} steps.")

    # Plot orbits of planets coloured by speed
    def plot_orbits_with_speed(self, planets_1_2, planets_jupiter_saturn):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        # Collect all speeds so can compare 
        all_speeds = []
        for planet in planets_1_2 + planets_jupiter_saturn:
            data = np.array(planet.history)
            vx, vy = data[:, 2], data[:, 3]
            speeds = np.sqrt(vx**2 + vy**2) / 1000  # Convert m/s to km/s
            all_speeds.extend(speeds)

        vmin, vmax = min(all_speeds), max(all_speeds)  # Used for the shared colour scale

        norm = colors.Normalize(vmin=vmin, vmax=vmax) # Normalize the colour scale
        sm = cm.ScalarMappable(cmap = cm.plasma, norm=norm)
        sm.set_array([])  # Required for colourbar

        # Will have two subplots side-by-side with Star / Sun at the centre 
        axs[0].plot(0, 0, '*', color='darkorange', label='Star', markersize=15)
        axs[1].plot(0, 0, '*', color='darkorange', label='Sun', markersize=15)
        # First plot
        for planet in planets_1_2:
            data = np.array(planet.history)
            x, y = data[:, 0] / AU, data[:, 1] / AU
            vx, vy = data[:, 2], data[:, 3]
            speeds = np.sqrt(vx**2 + vy**2) / 1000  # Convert to km/s
            axs[0].scatter(x, y, c=speeds, cmap=cm.plasma, s=10, norm=norm, label=planet.name)
        axs[0].set_xlabel("x (AU)")
        axs[0].set_ylabel("y (AU)")
        axs[0].legend(loc="upper right")
        axs[0].set_ylim(-11.5, 11.5)
        axs[0].set_xlim(-10, 10)
        axs[0].grid(True)

        # Second subplot
        for planet in planets_jupiter_saturn:
            data = np.array(planet.history)
            x, y = data[:, 0] / AU, data[:, 1] / AU
            vx, vy = data[:, 2], data[:, 3]
            speeds = np.sqrt(vx**2 + vy**2) / 1000  # Convert to km/s
            axs[1].scatter(x, y, c=speeds, cmap=cm.plasma, s=10, norm=norm, label=planet.name)
        axs[1].set_xlabel("x (AU)")
        axs[1].legend(loc="upper right")
        axs[1].axis("equal")
        axs[1].grid(True)

        # Shared colourbar
        cbar = fig.colorbar(sm, ax=axs, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label("Speed relative to COM (km/s)", rotation=90)
        plt.show()

# Define and create planets
planet_data = {
    'Planet 1': (2.52 * AU, 1e-3 * M_sun),
    'Planet 2': (5.24 * AU, 4e-2 * M_sun),
    'Jupiter': (5.2 * AU, 1.898e27),
    'Saturn': (9.58 * AU, 5.683e26)
}

planet_objects = [Planet(name, *data) for name, data in planet_data.items()]

# Run simulations for each planet system 
sim_planet_1_2 = SolarSystemSimulation(planet_objects[:2], 
                                       dt_init=Year / 100, 
                                       end_time=Year * 100, 
                                       error_tolerance=1000)

sim_jupiter_saturn = SolarSystemSimulation(planet_objects[2:], 
                                           dt_init=Year / 100, 
                                           end_time=Year * 100, 
                                           error_tolerance =1000)

# Run and plot
sim_planet_1_2.run_simulation(), sim_jupiter_saturn.run_simulation()
sim_planet_1_2.plot_orbits_with_speed(planet_objects[:2], planet_objects[2:])