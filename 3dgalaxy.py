# RigbySpace: Discrete 3D Spiral Orbit + Stepped Rotation Curve (True RS + Empirical Comparison)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- RS Recursive Sequence ---
def generate_Tn(n_max, delta=1/11):
    T = [22/7, 7/19]
    for _ in range(2, n_max):
        T.append(T[-1] + T[-2] + delta)
    return np.array(T)

def compute_Phi(T):
    n_vals = np.arange(len(T))
    T_bar = n_vals * (T[1] - T[0]) + T[0]
    return T - T_bar

# --- RS Metric Field (3D) ---
def generate_recursive_metric_3d(size, T, modulus=20):
    metric = np.zeros((size, size, size, 3, 3))
    center = size // 2
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r2 = (i - center)**2 + (j - center)**2 + (k - center)**2
                index = min(r2, len(T) - 1)
                g_tt = int(T[index] % modulus + 1)
                metric[i, j, k, 0, 0] = g_tt
                metric[i, j, k, 1, 1] = 1
                metric[i, j, k, 2, 2] = 1
    return metric

# --- Empirical Mass-Based Metric Field ---
def generate_empirical_metric_3d(size, radii, velocities, G=6.67430e-11):
    metric = np.zeros((size, size, size, 3, 3))
    center = size // 2
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                if r <= radii[0]:
                    v = velocities[0]
                elif r >= radii[-1]:
                    v = velocities[-1]
                else:
                    for idx in range(1, len(radii)):
                        if r <= radii[idx]:
                            r1, r2 = radii[idx - 1], radii[idx]
                            v1, v2 = velocities[idx - 1], velocities[idx]
                            v = v1 + (v2 - v1) * ((r - r1) / (r2 - r1))
                            break
                M = (v**2 * r) / G
                g_tt = int(np.floor(M % 20 + 1))
                metric[i, j, k, 0, 0] = g_tt
                metric[i, j, k, 1, 1] = 1
                metric[i, j, k, 2, 2] = 1
    return metric

# --- Spiral Orbit (RS Lattice) ---
def spiral_orbit_steps_3d(radius, height, turns):
    steps = []
    theta = 0
    dtheta = 2 * np.pi / turns
    dz = height / turns
    for _ in range(turns):
        dx = int(np.round(radius * np.cos(theta)))
        dy = int(np.round(radius * np.sin(theta)))
        dz_step = int(np.round(dz))
        steps.append([dx, dy, dz_step])
        theta += dtheta
    return np.array(steps)

# --- RS Action ---
def compute_action_rational_3d(metric, steps):
    S = 0
    pos = np.array([0, 0, 0])
    for dx in steps:
        g = metric[pos[0] % metric.shape[0], pos[1] % metric.shape[1], pos[2] % metric.shape[2]]
        dx_vec = np.array(dx).reshape((3, 1))
        term = int(dx_vec.T @ g @ dx_vec)
        S += term
        pos += dx
    return S / len(steps)

# --- Rolling Average ---
def rolling_average(data, window=2):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# === RUN ===
n_max = 1000
size = 32
T = generate_Tn(n_max)
metric3d = generate_recursive_metric_3d(size, T)

radii = list(range(2, 14, 2))
actions = []

for r in radii:
    steps = spiral_orbit_steps_3d(radius=r, height=10, turns=100)
    S = compute_action_rational_3d(metric3d, steps)
    actions.append(S)
    print(f"Radius={r}, RS Action per Step: {S:.3f}")

# --- Compute Rolling Average ---
avg_roll = rolling_average(actions, window=2)
avg_radii = radii[1:]

# === PLOT: RS Action vs Radius (Stepped) ===
plt.figure(figsize=(8, 5))
plt.step(radii, actions, where='mid', label="RS Spiral Metric", color='blue', marker='o')
plt.plot(avg_radii, avg_roll, color='green', linestyle='--', label=f"Rolling Mean")
plt.title("Galactic Rotation in RS (Stepped)")
plt.xlabel("Spiral Radius (RSU)")
plt.ylabel("Avg. Action / Step (RSU)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rs_rotation_stepped_with_overlay.png", dpi=300)
plt.show()

# === PLOT: 3D Spiral Orbit ===
last_steps = spiral_orbit_steps_3d(radius=8, height=10, turns=100)
pos = np.array([0, 0, 0])
positions = [pos.copy()]
for step in last_steps:
    pos += step
    positions.append(pos.copy())
positions = np.array(positions)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='blue', marker='o', label="RS Spiral Path")
ax.set_title("Discrete 3D Spiral Orbit in RS Curvature Field")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.savefig("rs_spiral3d_discrete.png", dpi=300)
plt.show()

# === Empirical Mass Simulation (Optional Use) ===
empirical_radii = [2, 4, 6, 8, 10, 12]
empirical_si_velocities = [160000, 185000, 200000, 210000, 215000, 220000]
empirical_metric3d = generate_empirical_metric_3d(size, empirical_radii, empirical_si_velocities)
actions_empirical = []
for r in radii:
    steps = spiral_orbit_steps_3d(radius=r, height=10, turns=100)
    S_emp = compute_action_rational_3d(empirical_metric3d, steps)
    actions_empirical.append(S_emp)
    print(f"[Empirical] Radius={r}, Action per Step: {S_emp:.3f}")

# === PLOT: Empirical-Based vs Pure RS ===
plt.figure(figsize=(8, 5))
plt.step(radii, actions, where='mid', label="RS Metric (Recursive)", color='blue', marker='o')
plt.step(radii, actions_empirical, where='mid', label="Empirical Mass Field", color='red', marker='x')
plt.title("RS vs Empirical-Based Curvature Response")
plt.xlabel("Spiral Radius (RSU)")
plt.ylabel("Avg. Action / Step (RSU)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rs_vs_empirical_action.png", dpi=300)
plt.show()

