from fractions import Fraction
from math import floor
from collections import namedtuple

# Set up RS recurrence
def generate_Tn(N, T0=Fraction(22, 7), T1=Fraction(7, 19), delta=Fraction(1, 11)):
    T = [T0, T1]
    for _ in range(2, N):
        T.append(T[-1] + T[-2] + delta)
    return T

# Define the RS imbalance field
def compute_Phi(T):
    T0, T1 = T[0], T[1]
    Phi = []
    for n, val in enumerate(T):
        Tbar = T0 + n * (T1 - T0)
        Phi.append(val - Tbar)
    return Phi

# Generate RS metric field g_tt(r) = floor(T_{r^2} mod M + 1)
def generate_metric_field(T, N, modulus=20):
    metric = []
    for n in range(N):
        r2 = n ** 2
        if r2 < len(T):
            raw = T[r2]
            mod_val = raw.numerator % modulus  # Avoid float %; operate on numerator
            g_tt = floor(mod_val + 1)
            metric.append(g_tt)
        else:
            metric.append(1)  # Fallback for out-of-range r^2
    return metric

# Discrete RS geodesic path tracer (2D lattice)
Step = namedtuple('Step', ['x', 'y', 'g_tt'])

def trace_geodesic(metric, max_radius):
    path = []
    for r in range(1, max_radius + 1):
        if r < len(metric):
            g_tt = metric[r]
            path.append(Step(r, r, g_tt))  # Sample spiral: (r, r) diagonal motion
    return path

# Compute total RS action along geodesic
def compute_action(path):
    return sum(step.g_tt for step in path)

# --- Execution ---

N = 200
Tn = generate_Tn(N)
Phi = compute_Phi(Tn)
metric_field = generate_metric_field(Tn, N)

path = trace_geodesic(metric_field, max_radius=50)
action = compute_action(path)

# Output results
print("RS Geodesic Simulation (Compliant)")
print(f"Total action along geodesic: {action}")
print(f"Steps traced: {len(path)}")

# Optional: log path to file
with open("rs_path_log.txt", "w") as f:
    for step in path:
        f.write(f"{step.x},{step.y},{step.g_tt}\n")

