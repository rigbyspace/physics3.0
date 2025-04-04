# Simulate RigbySpace geodesic action with and without irrational values
import numpy as np
import matplotlib.pyplot as plt
import fractions

# Recursive excitation sequence (Tn) and imbalance field (Phi)
def generate_Tn(n_max, delta=1/11):
    T = [22/7, 7/19]  # RS initialization: avoids sqrt, uses rational primes
    for _ in range(2, n_max):
        T.append(T[-1] + T[-2] + delta)
    return np.array(T)

def compute_Phi(T):
    n_vals = np.arange(len(T))
    T_bar = n_vals * (T[1] - T[0]) + T[0]
    return T - T_bar

# Galactic metric generated from recursive imbalance

def generate_recursive_metric(size, T):
    metric = np.zeros((size, size, 2, 2))
    center = size // 2
    for i in range(size):
        for j in range(size):
            r2 = (i - center)**2 + (j - center)**2
            index = min(r2, len(T) - 1)
            g_tt = int(T[index] % 20 + 1)  # time curvature from Tn
            metric[i, j, 0, 0] = g_tt
            metric[i, j, 1, 1] = 1  # flat space
    return metric

def circular_orbit_steps(radius, revolutions):
    steps = []
    theta = 0
    dtheta = 2 * np.pi / revolutions
    for _ in range(revolutions):
        dx = int(np.round(radius * np.cos(theta)))
        dy = int(np.round(radius * np.sin(theta)))
        steps.append([dx, dy])
        theta += dtheta
    return np.array(steps)

def compute_action_rational(metric, steps):
    S = 0
    pos = np.array([0, 0])
    for dx in steps:
        g = metric[pos[0] % metric.shape[0], pos[1] % metric.shape[1]]
        dx_vec = dx.reshape((2, 1))
        term = int(dx_vec.T @ g @ dx_vec)
        S += term
        pos += dx
    return S / len(steps)

# Run simulation
print("\n[RS GALACTIC ROTATION PROFILE FROM RECURSION]")
n_max = 1000
size = 64
T = generate_Tn(n_max)
metric = generate_recursive_metric(size, T)

radii = list(range(2, 20, 2))
actions = []

for r in radii:
    steps = circular_orbit_steps(radius=r, revolutions=50)
    S = compute_action_rational(metric, steps)
    actions.append(S)
    print(f"Radius={r}, RS Orbit Action per Step: {S:.3f}")

plt.figure(figsize=(8, 5))
plt.plot(radii, actions, marker='o', label="RS Recursion Metric")
plt.title("RigbySpace Orbital Action vs Radius (from imbalance)")
plt.xlabel("Orbital Radius (lattice units)")
plt.ylabel("Average Action per Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot imbalance field
print("\n[RS MASS GAP SCAN UP TO n = 750]")
Phi = compute_Phi(T)

for i in range(700, 750):
    print(f"n={i}, Tn={T[i]:.3e}, Φ={Phi[i]:.3e}")

plt.figure(figsize=(10, 5))
plt.plot(np.arange(n_max), Phi, label='Φ(n) — Imbalance Field')
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(240, color='red', linestyle=':', label='~20 TeV (n≈240)')
plt.axvline(720, color='purple', linestyle=':', label='Rigby Collapse Claim')
plt.title("Rigby Φ(n) — Time Structure Imbalance")
plt.xlabel("n")
plt.ylabel("Φ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

