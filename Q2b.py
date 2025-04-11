#Import Libaries
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


''''
Class for the celestial bodies - similar to Comet class in 1b 
'''
# Class for Celestial Bodies with properties and state vector in SI units
class CelestialBody:
    def __init__(self, name, mass, initial_pos, initial_vel):
        self.name = name
        self.mass = mass
        self.state_vec = np.array(initial_pos + initial_vel, dtype=np.float64)
        self.state_history = [self.state_vec.copy()]  # Keep track of all states
        self.energy_history = []  # Store kinetic, potential, and total energy
        self.dt_history = []      # Store time-step sizes
        self.time_history = []    # Store simulation time values

    # Compute net gravitational acceleration from other bodies. 
    def acceleration(self, all_bodies):
        x, y, vx, vy = self.state_vec
        ax, ay = 0.0, 0.0

        for other_body in all_bodies:
            if other_body is self:
                continue
            xj, yj, vxj, vyj = other_body.state_vec
            m_j = other_body.mass

            dx = xj - x
            dy = yj - y
            r2 = dx**2 + dy**2
            if r2 > 1e-12:
                r = np.sqrt(r2)
                accel_factor = G * m_j / (r2 * r)  # G*m_j / r^3
                ax += accel_factor * dx
                ay += accel_factor * dy

        return np.array([vx, vy, ax, ay], dtype=np.float64)
   
    # Calculating energy per body 
    def compute_energy(self, all_bodies):
        x, y, vx, vy = self.state_vec
        speed_sq = vx**2 + vy**2
        KE = 0.5 * self.mass * speed_sq  # Kinetic Energy
        PE = 0.0 # Initially

        for other_body in all_bodies:
            if other_body is self:
                continue
            xj, yj, _, _ = other_body.state_vec
            m_j = other_body.mass
            dx, dy = xj - x, yj - y
            r = np.sqrt(dx**2 + dy**2)
            if r > 1e-12:
                PE += -G * self.mass * m_j / r

        E = KE + PE # Total Energy
        self.energy_history.append([KE, PE, E])

'''
Main Class for the multi-body simulation
Similar to the Runsimulation class in 1b
'''
# The main class for the multi-body simulation
class MultiBodySimulation:
    def __init__(self, bodies, dt_init, end_time, error_tolerance):
        self.bodies = bodies
        self.dt = dt_init
        self.end_time = end_time
        self.error_tolerance = error_tolerance
        self.shift_to_centre_of_mass()

    # Compute where the centre of mass is for the system 
    def compute_centre_of_mass(self):
        total_mass = sum(body.mass for body in self.bodies)
        pos_cm = np.zeros(2)
        vel_cm = np.zeros(2)

        for body in self.bodies:
            pos_cm += body.mass * body.state_vec[:2]
            vel_cm += body.mass * body.state_vec[2:]

        return pos_cm / total_mass, vel_cm / total_mass
    
    # Compute the velocity of the center of mass at a given index in history.
    def compute_com_velocity(self, index):
        total_mass = sum(body.mass for body in self.bodies)
        # X and Y components of the velocity of the center of mass
        vx_cm = sum(body.mass * body.state_history[index][2] for body in self.bodies) / total_mass
        vy_cm = sum(body.mass * body.state_history[index][3] for body in self.bodies) / total_mass
        return np.array([vx_cm, vy_cm])

    # Shift bodies so COM is at the Origin and has zero net velocity
    def shift_to_centre_of_mass(self):
        r_cm, v_cm = self.compute_centre_of_mass()
        for body in self.bodies:
            body.state_vec[:2] -= r_cm
            body.state_vec[2:] -= v_cm
            # Reset the history to reflect the shift
            body.state_history = [body.state_vec.copy()]

    # Get the system state vectors as a flattened array
    def get_system_state(self):
        return np.concatenate([body.state_vec for body in self.bodies])

    # Update each body's state vector from a single flattened array
    def set_system_state(self, flat_state):
        for i, body in enumerate(self.bodies):
            body.state_vec = flat_state[4*i : 4*i + 4]

    # Compute velocity of each body relative to the center of mass at a given history index.
    def compute_relative_speeds(self, index):
        com_vel = self.compute_com_velocity(index)
        rel_speed_dict = {}
        for body in self.bodies:
            vx, vy = body.state_history[index][2], body.state_history[index][3]
            v_rel = np.sqrt((vx - com_vel[0])**2 + (vy - com_vel[1])**2)
            rel_speed_dict[body.name] = v_rel
        return rel_speed_dict

    # Compute derivatives for all bodies at once [vx, vy, ax, ay]
    def compute_derivatives(self, flat_state):
        self.set_system_state(flat_state)
        derivatives = []
        for body in self.bodies:
            derivatives.append(body.acceleration(self.bodies))
        return np.concatenate(derivatives)

    # Standard RK4 step just for whole system now 
    def rk4_step(self, state_array, dt):
        k1 = dt * self.compute_derivatives(state_array)
        k2 = dt * self.compute_derivatives(state_array + 0.5 * k1)
        k3 = dt * self.compute_derivatives(state_array + 0.5 * k2)
        k4 = dt * self.compute_derivatives(state_array + k3)

        return state_array + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # Adaptive RK4 step for the whole system
    def adaptive_rk4_step(self):
        while True:
            original_state = self.get_system_state()
            # Hard code rk4 method for this 
            full_step_state = self.rk4_step(original_state, self.dt)
            half_step_1 = self.rk4_step(original_state, self.dt / 2.0)
            half_step_2 = self.rk4_step(half_step_1, self.dt / 2.0)
            # Calculate the difference between full and half 
            error_vec = full_step_state - half_step_2
            local_error = np.linalg.norm(error_vec) / 30.0 # 30 = RK4 correction factor

            if local_error > self.error_tolerance:
                self.dt *= 0.5 # Reduce step size if error is too large
                continue
            else:
                scaling_factor = (self.error_tolerance / (local_error + 1e-10)) ** 0.2
                # Change capped at: 0.5 < dt < 2
                self.dt = self.dt * np.clip(scaling_factor, 0.5, 2.0)
                # Final update using the adjusted dt
                new_state = self.rk4_step(original_state, self.dt)
                self.set_system_state(new_state)
                # Record new state and compute energy for each body
                for body in self.bodies:
                    body.state_history.append(body.state_vec.copy())
                    body.compute_energy(self.bodies)

                return self.dt

    # Run the simulation for the specified time
    def run_simulation(self):
        current_time, steps_taken = 0.0, 0
        start_time = time.time()
        # Run for specified amount of time
        while current_time < self.end_time:
            # Perform one adaptive step
            updated_dt = self.adaptive_rk4_step()
            current_time += updated_dt
            # Store time and dt for each body
            for body in self.bodies:
                body.time_history.append(current_time)
                body.dt_history.append(updated_dt)

            steps_taken += 1

        runtime = time.time() - start_time
        print(f"Simulation finished in {runtime:.2f} s with {steps_taken} steps.")

    # Plotting the trajectories of all bodies with colour-coded speeds
    def plot_trajectory_speed(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        # Dont want the same markers 
        custom_markers = {"Star": "*", "Planet 1": "^", "Planet 2": "x", "Jupiter": "D", "Saturn": "v",}

        # Collect speeds for a shared colour scale
        data = []
        for idx in range(len(self.bodies[0].state_history)):
            com_vel = self.compute_com_velocity(idx)
            for body in self.bodies:
                vx, vy = body.state_history[idx][2], body.state_history[idx][3] # Extract velocities
                # Compute relative speed to the center of mass
                v_rel = np.sqrt((vx - com_vel[0])**2 + (vy - com_vel[1])**2)
                data.append(v_rel) # Store all speeds

        # Convert to numpy array and get min/max for colour scale
        data = np.array(data)
        vmin, vmax = data.min() / 1000.0, data.max() / 1000.0

        # Plot orbits - loop over all bodies
        for body in self.bodies:
            data = np.array(body.state_history)
            # Extract x and y positions - convert to AU
            x_vals, y_vals = data[:, 0] / AU, data[:, 1] / AU
            rel_speeds = []
            for idx in range(len(body.state_history)):
                com_vel = self.compute_com_velocity(idx)
                vx, vy = body.state_history[idx][2], body.state_history[idx][3]
                # Compute relative speed to the center of mass
                v_rel = np.sqrt((vx - com_vel[0])**2 + (vy - com_vel[1])**2)
                # Convert to km/s
                rel_speeds.append(v_rel / 1000.0)
            # Use custom markers for each body
            marker_style = custom_markers.get(body.name, "o")
            sc = ax.scatter(x_vals, y_vals, c=rel_speeds, s=10, cmap='plasma',
                            vmin=vmin, vmax=vmax, marker=marker_style, label=body.name)
        # Add the colourbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Speed relative to COM (km/s)", rotation=90)
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.axis('equal')
        ax.grid(True)
        ax.legend(loc = 'upper right')

        plt.show()

'''
Define the bodies for the simulation 
'''
star = CelestialBody(name = "Star", 
                     mass = M_sun, 
                     initial_pos = [0.0, 0.0], 
                     initial_vel = [0.0, 0.0])

Jupiter = CelestialBody(name = "Jupiter", 
                        mass = 1.898e27, 
                        initial_pos = [5.2 * AU, 0.0],
                        initial_vel = [0.0, np.sqrt(G * M_sun / (5.2 * AU))])

Saturn = CelestialBody(name = "Saturn", 
                       mass = 5.683e26, 
                       initial_pos = [9.58 * AU, 0.0],
                       initial_vel = [0.0, np.sqrt(G * M_sun / (9.58 * AU))])

planet1 = CelestialBody(name = "Planet 1",
                        mass = 1e-3 * M_sun, 
                        initial_pos = [2.52 * AU, 0.0],
                        initial_vel = [0.0, np.sqrt(G * M_sun / (2.52 * AU))])

planet2 = CelestialBody(name = "Planet 2", 
                        mass = 4e-2 * M_sun, 
                        initial_pos = [5.24 * AU, 0.0],
                        initial_vel = [0.0, np.sqrt(G * M_sun / (5.24 * AU))])

# List of all bodies
body_list = [star, planet1, planet2]

# Create and run the simulation
simulation = MultiBodySimulation(body_list, dt_init=Year / 100, 
                                 end_time = 50 * Year, 
                                 error_tolerance=100)
simulation.run_simulation()
simulation.plot_trajectory_speed()