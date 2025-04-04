# Simulate RigbySpace geodesic action with and without irrational values
import matplotlib.pyplot as plt
from fractions import Fraction
import numpy as np

# Recursive excitation sequence (Tn) and imbalance field (Phi)
def generate_Tn(n_max, delta=Fraction(1, 11)):
    T = [Fraction(22, 7), Fraction(7, 19)]
    for _ in range(2, n_max):
        T.append(T[-1] + T[-2] + delta)
    return T

def compute_Phi(T):
    n_vals = list(map(Fraction, range(len(T))))
    slope = T[1] - T[0]
    T_bar = [n * slope + T[0] for n in n_vals]
    return [t - tb for t, tb in zip(T, T_bar)]

def compute_spectral_field(T, a=1, b=1, r=Fraction(7, 8), s=Fraction(4, 3)):
    U = []
    for n in range(len(T)):
        phi = T[n] - (T[1] - T[0]) * n - T[0]
        weight = (a + b * n)**2
        value = phi * weight * (r**n) / (s**n)
        U.append(float(value))
    return U

# 3D Recursive metric from Tn

def generate_recursive_metric_3d(size, T):
    metric = np.zeros((size, size, size, 3, 3), dtype=int)
    center = size // 2
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r2 = (i - center)**2 + (j - center)**2 + (k - center)**2
                index = min(r2, len(T) - 1)
                g_tt = int(T[index] % 20 + 1)
                metric[i, j, k, 0, 0] = g_tt
                metric[i, j, k, 1, 1] = 1
                metric[i, j, k, 2, 2] = 1
    return metric

# Simulate a 2D elliptical orbit and track perihelion rotation

def elliptical_orbit_steps_2d(a, b, revolutions):
    steps = []
    positions = []
    angle = 0
    angle_step = Fraction(1, 20)
    for rev in range(revolutions):
        for _ in range(40):
            x = int(round(a * np.cos(float(angle))))
            y = int(round(b * np.sin(float(angle))))
            if positions and [x, y] != positions[-1]:
                dx = x - positions[-1][0]
                dy = y - positions[-1][1]
                steps.append([dx, dy, 0])
            elif not positions:
                steps.append([x, y, 0])
            positions.append([x, y])
            angle += angle_step
    return np.array(steps), positions

def track_perihelion(positions):
    perihelion_angles = []
    for i in range(1, len(positions)-1):
        r0 = np.linalg.norm(positions[i-1])
        r1 = np.linalg.norm(positions[i])
        r2 = np.linalg.norm(positions[i+1])
        if r1 < r0 and r1 < r2:
            x, y = positions[i]
            angle = np.arctan2(y, x)
            perihelion_angles.append(angle)
    return perihelion_angles

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

# Run Mercury precession test
print("\n[RS MERCURY ORBIT — FREE EVOLUTION + PERIHELION TRACKING]")
n_max = 1000
size = 64
T = generate_Tn(n_max)
metric3d = generate_recursive_metric_3d(size, T)

steps, positions = elliptical_orbit_steps_2d(a=10, b=8, revolutions=30)
S = compute_action_rational_3d(metric3d, steps)
print(f"Mercury-like orbit average action: {S:.3f}")

xy_path = np.array([[x, y] for x, y, _ in steps])
plt.figure(figsize=(6, 6))
plt.plot(xy_path[:, 0], xy_path[:, 1], lw=1)
plt.title("RigbySpace Mercury Orbit (Free Evolution)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Perihelion tracking
peri_angles = track_perihelion(positions)
plt.figure(figsize=(8, 4))
plt.plot(peri_angles, marker='o')
plt.title("RigbySpace Mercury Perihelion Angles over Time")
plt.xlabel("Orbit Count")
plt.ylabel("Angle (radians)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot imbalance field
print("\n[RS MASS GAP SCAN UP TO n = 750]")
Phi = compute_Phi(T)

for i in range(700, 750):
    print(f"n={i}, Tn={float(T[i]):.3e}, Φ={float(Phi[i]):.3e}")

plt.figure(figsize=(10, 5))
plt.plot(range(n_max), list(map(float, Phi)), label='Φ(n) — Imbalance Field')
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

# Plot RS energy spectrum from Φ field
U = compute_spectral_field(T)
plt.figure(figsize=(10, 4))
plt.plot(U)
plt.title("RigbySpace Recursive Energy Spectrum U(ν)")
plt.xlabel("n")
plt.ylabel("Energy Contribution")
plt.grid(True)
plt.tight_layout()
plt.show()

