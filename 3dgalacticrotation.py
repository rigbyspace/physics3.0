# RigbySpace 3D Spiral Orbit Action + Imbalance Field Scan
import numpy as np
import matplotlib.pyplot as plt

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

# --- RS Discrete Metric Tensor (3D) ---
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

# --- Spiral Orbit Path (3D) ---
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

# --- RS Geodesic Action ---
def compute_action_rational_3d(metric, steps):
    S = 0
    pos = np.array([0, 0, 0])
    for dx in steps:
        g = metric[pos[0] % metric.shape[0], pos[1] % metric.shape[1], pos[2] % metric.shape[2]]
        dx_vec = dx.reshape((3, 1))
        term = int(dx_vec.T @ g @ dx_vec)
        S += term
        pos += dx
    return S / len(steps)

# === RUN SIMULATION ===
print("\n[RS 3D SPIRAL ORBIT ACTION FROM RECURSION]")
n_max = 1000
size = 32
T = generate_Tn(n_max)
metric3d = generate_recursive_metric_3d(size, T)

radii = list(range(2, 14, 2))
actions = []

for r in radii:
    steps = spiral_orbit_steps_3d(radius=r, height=10, turns=50)
    S = compute_action_rational_3d(metric3d, steps)
    actions.append(S)
    print(f"Radius={r}, RS Action per Step: {S:.3f}")

# === Generate Spiral Path for 3D Plot ===
from mpl_toolkits.mplot3d import Axes3D

last_steps = spiral_orbit_steps_3d(radius=radii[-1], height=10, turns=50)
pos = np.array([0, 0, 0])
positions = [pos.copy()]

for step in last_steps:
    pos += step
    positions.append(pos.copy())

positions = np.array(positions)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='blue', label="RS Spiral Path")
ax.set_title("3D Spiral Orbit Path in RS Curvature Field")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.savefig("rs_spiral3d_plot_fixed.png", dpi=300)
plt.show()

# === Plot RS Action vs Radius + SI Overlay ===
plt.figure(figsize=(8, 5))
plt.plot(radii, actions, marker='o', label="RS Spiral Metric", color='blue')

# Proper SI → RSU conversion for velocity comparison
# Use derived RSU velocity unit: 1 RSU velocity = approx 1.9e5 m/s (based on RSU mass + energy)
# See RS section 9.2 for scaling logic

empirical_radii = [2, 4, 6, 8, 10, 12]  # RSU-compatible scale
empirical_si_velocities = [160000, 185000, 200000, 210000, 215000, 220000]  # in m/s

# Convert SI velocity to RS action scale (anchor to RSU ~ 190,000 m/s per unit)
rsu_per_mps = 1 / 190000  # 1 RSU velocity ≈ 190,000 m/s
empirical_rsu_velocities = [v * rsu_per_mps for v in empirical_si_velocities]

plt.plot(empirical_radii, empirical_rsu_velocities, 'ko', label="Observed Velocities (converted to RSU)")
plt.title("RS 3D Spiral Orbit Action vs Radius\nwith Empirical Rotation Curve Overlay")
plt.xlabel("Spiral Radius (RSU)")
plt.ylabel("Avg. Action / Step (RSU)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("galactic_rotation_with_overlay.png", dpi=300)
plt.show()

