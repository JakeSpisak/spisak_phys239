import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants in cgs units
c = 3*10**10 #speed of light
ab = 5.29*10**-9 #bohr radius
e = 4.8*10**-10 #electron charge (statcoulomb)
z = 1 # Ion charge
me = 9.1 * 10**-28 # electron mass

# Initial conditions
p_o = np.array([-500, 1000])*ab
v_o = np.array([1/3000., 0.])*c
a_o = -1*z*e**2*p_o/(me* np.linalg.norm(p_o)**3)

#Simulation parameters
num_steps = 50000
delta_t = 50 * ab/c 
times = np.arange(num_steps+1)*delta_t

#The solver.
def elec_sim(num_steps, delta_t, p_o, v_o, a_o):
    """Given a certain number of steps, simulation parameters, and some initial conditions, computes the 
    position, velocity and acceleration of an electrion."""
    positions = [p_o]
    velocities = [v_o]
    accelerations = [a_o]
    for i in range(num_steps):
        a_temp = -1*z*e**2*positions[i]/(me* np.linalg.norm(positions[i])**3)
        v_temp = velocities[i] + delta_t * a_temp
        p_temp = positions[i] + delta_t * v_temp
        accelerations.append(a_temp)
        velocities.append(v_temp)
        positions.append(p_temp)
        
    return np.array(positions), np.array(velocities), np.array(accelerations)

#Run the solver and plot the results
print(f"Running the solver with {num_steps} steps")
positions, velocities, accelerations = elec_sim(num_steps, delta_t, p_o, v_o, a_o)

print(f"Plotting output to /output")
plt.figure(figsize=(6,6))
plt.plot(positions[:,0]/ab, positions[:,1]/ab)
plt.title("Position")
plt.xlabel(r"x ($a_{bohr}$)")
plt.ylabel(r"y ($a_{bohr}$)")
plt.savefig("outputs/position")

plt.figure(figsize=(6,6))
plt.plot(times, velocities[:,0])
plt.title("Velocities")
plt.xlabel(r"time (s)")
plt.ylabel(r"Vx (cm/s)")
plt.savefig("outputs/vx")

plt.figure(figsize=(6,6))
plt.plot(times, velocities[:,1])
plt.title("Velocities")
plt.xlabel(r"time (s)")
plt.ylabel(r"Vy (cm/s)")
plt.savefig("outputs/vy")

plt.figure(figsize=(6,6))
plt.plot(times, accelerations[:,0])
plt.title("Acceleration")
plt.xlabel(r"time (s)")
plt.ylabel(r"Ax (cm/s^2)")
plt.savefig("outputs/ax")

plt.figure(figsize=(6,6))
plt.plot(times, accelerations[:,1])
plt.title("Acceleration")
plt.xlabel(r"time (s)")
plt.ylabel(r"Ay (cm/s^2)")
plt.savefig("outputs/ay")
