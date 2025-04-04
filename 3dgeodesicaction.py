from fractions import Fraction
from math import floor
from collections import namedtuple

# RS recurrence for T_n
def generate_Tn(N, T0=Fraction(22, 7), T1=Fraction(7, 19), delta=Fraction(1, 11)):
    T = [T0, T1]
    for _ in range(2, N):
        T.append(T[-1] + T[-2] + delta)
    return T

# Metric field: g_tt(r) from T_{r^2}
def generate_metric_field(T, R, modulus=20):
    metric = {}
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            for z in range(-R, R + 1):
                r2 = x*x + y*y + z*z
                if r2 < len(T):
                    raw = T[r2]
                    g_tt = floor(raw.numerator % modulus + 1)
                    metric[(x, y, z)] = g_tt
    return metric

# 3D spiral path (rising with each full turn)
Step3D = namedtuple('Step3D', ['x', 'y', 'z', 'g_tt'])

def trace_3d_spiral(metric, radius=5, height=10):
    path = []
    for layer in range(height):
        for x in range(-radius, radius + 1):
            y = -x
            z = layer
            key = (x, y, z)
            g_tt = metric.get(key, 1)
            path.append(Step3D(x, y, z, g_tt))
    return path

# RS action
def compute_action(path):
    return sum(step.g_tt for step in path)

# --- Run it ---

N = 500
Tn = generate_Tn(N)
metric_field = generate_metric_field(Tn, R=6)

path3d = trace_3d_spiral(metric_field, radius=5, height=10)
action3d = compute_action(path3d)

print("3D RS Spiral Geodesic")
print(f"Steps: {len(path3d)}")
print(f"Total action: {action3d}")

# Optional log
with open("rs_spiral3d_log.txt", "w") as f:
    for step in path3d:
        f.write(f"{step.x},{step.y},{step.z},{step.g_tt}\n")
        
# Visualization (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    xs = [step.x for step in path3d]
    ys = [step.y for step in path3d]
    zs = [step.z for step in path3d]
    colors = [step.g_tt for step in path3d]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', marker='o')
    fig.colorbar(scatter, ax=ax, label='g_tt value')
    ax.set_title("RS 3D Spiral Geodesic in Discrete Curvature Field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig("rs_spiral3d_plot.png", dpi=300)
    plt.show()
except ImportError:
    print("matplotlib not installed: skipping plot")


