import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
from decimal import Decimal, getcontext

getcontext().prec = 50  # For symbolic safety where needed

# ---- 1. Φ(n) ----
def generate_phi_sequence(n_max):
    phi = [Fraction(1, 1)]
    for n in range(1, n_max):
        phi.append(phi[-1] + Fraction(1, n + 1))
    return phi

def plot_phi():
    n_max = 50
    phi = generate_phi_sequence(n_max)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(range(n_max), [float(p) for p in phi], label=r'$\Phi(n)$', color='darkred')
    plt.xlabel(r'$n$'); plt.ylabel(r'$\Phi(n)$')
    plt.title('RS Curvature Kernel')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig('fig_phi_sequence.png'); plt.close()

# ---- 2. τ(n) ----
def plot_tau():
    n = np.arange(1, 100)
    tau = np.cumsum(1 / (n + 1))  # placeholder: integrated growth
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(n, tau, label=r'$\tau(n)$', color='blue')
    plt.xlabel(r'$n$'); plt.ylabel(r'$\tau(n)$')
    plt.title('Recursive Spectral Growth')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig('fig_tau_sequence.png'); plt.close()

# ---- 3. g_tt(r) ----
def plot_gtt():
    r = np.linspace(1, 20, 200)
    gtt = 1 - 0.5 * np.exp(-r / 5)  # placeholder RS metric curve
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(r, gtt, label=r'$g_{tt}^{RS}(r)$', color='black')
    plt.xlabel(r'$r$'); plt.ylabel(r'$g_{tt}$')
    plt.title('RS Time Metric Component')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig('fig_gtt_metric.png'); plt.close()

# ---- 4. Tn vs T̄n ----
def plot_tn_vs_tnbar():
    n = np.arange(1, 50)
    Tn = np.log(n + 1) + 0.15 * np.sin(n / 3.0)
    Tn_bar = np.log(n + 1)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(n, Tn, label=r'$T_n$', color='darkorange')
    plt.plot(n, Tn_bar, '--', label=r'$\overline{T}_n$', color='gray')
    plt.xlabel(r'$n$'); plt.ylabel(r'$T$')
    plt.title('RS Step Oscillation')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(); plt.tight_layout()
    plt.savefig('fig_tn_vs_tnbar.png'); plt.close()

# ---- 5. Recursive Jet ----
def plot_recursive_jet():
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X**2 + Y**2) / (1 + X**2 + Y**2)  # placeholder RS jet field
    plt.figure(figsize=(6, 5), dpi=300)
    plt.contourf(X, Y, Z, 50, cmap='inferno')
    plt.colorbar(label='Curvature Density')
    plt.title('Recursive Jet Field')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('fig_recursive_jet.png'); plt.close()

# ---- 6. Spectral Field (CMB6 style) ----
def plot_spectral_field():
    x = np.linspace(0, 10, 300)
    y = np.linspace(0, 10, 300)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - 5)**2 + (Y - 5)**2)
    field = np.cos(R * 2 * np.pi / 3) * np.exp(-R / 4)  # damped wave
    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(field, extent=[0, 10, 0, 10], origin='lower', cmap='coolwarm')
    plt.colorbar(label='Spectral Amplitude')
    plt.title('RS Spectral Field')
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('fig_spectral_field.png'); plt.close()

# Run all
if __name__ == "__main__":
    plot_phi()
    plot_tau()
    plot_gtt()
    plot_tn_vs_tnbar()
    plot_recursive_jet()
    plot_spectral_field()

