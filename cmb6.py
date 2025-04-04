import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, c

VEN = 22
LUC = 19
Delta = 1/11
T0 = 22/7
T1 = 7/19

# Recursive mass-energy lattice (structure formation field)
def generate_recursive_volume(size=32, steps=20, noise=0.01):
    np.random.seed(42)
    volume = np.zeros((steps, size, size))
    T = generate_Tn(size**2)
    Phi = compute_phi(T)
    base_field = Phi.reshape((size, size)) - np.mean(Phi)
    volume[0] = base_field
    for t in range(1, steps):
        grad = np.gradient(volume[t-1])
        flow = -0.01 * (grad[0] + grad[1])
        volume[t] = volume[t-1] + flow + noise * np.random.randn(size, size)
    return volume

# Collapse + compute matter power spectrum
def compute_matter_power_spectrum(volume):
    final_density = volume[-1]
    final_density = np.nan_to_num(final_density)
    fft = np.fft.fft2(final_density)
    fft = np.nan_to_num(fft, nan=0.0, posinf=0.0, neginf=0.0)
    power = np.abs(fft)**2
    power = np.clip(power, 0, 1e12)
    power_shifted = np.fft.fftshift(power)
    return radial_average(power_shifted)


# Growth function analog: track evolution across RS time steps
def compute_growth_over_time(volume):
    spectra = []
    for frame in volume:
        fft = np.fft.fft2(frame)
        power = np.abs(fft)**2
        power_shifted = np.fft.fftshift(power)
        Cl = radial_average(power_shifted)
        spectra.append(Cl)
    return np.array(spectra)

# Map RS time steps to redshift values
def redshift_mapping(steps, z0=10, alpha=0.25):
    return z0 * np.exp(-alpha * np.arange(steps))

# --- Preexisting definitions kept as is ---
def generate_Tn(n_max):
    T = [T0, T1]
    for n in range(2, n_max):
        T_next = T[-1] + T[-2] + Delta
        T.append(T_next)
    return np.array(T)

def T_bar_n(n):
    slope = (T1 - T0)
    return n * slope + T0

def compute_phi(T):
    n_vals = np.arange(len(T))
    T_bar = T_bar_n(n_vals)
    return T - T_bar

def suppression(n, scale=0.15):
    return 1 / (1 + np.exp(scale * (n - 35)))

def map_n_to_freq(n):
    return 5 + n * 10

def u_RS(T, Phi):
    n_vals = np.arange(len(T))
    freqs = map_n_to_freq(n_vals)
    spec = Phi * freqs**2 * suppression(n_vals)
    return freqs, spec / np.max(spec)

def planck(freqs_GHz, T):
    freqs = freqs_GHz * 1e9
    return (2 * h * freqs**3 / c**2) / (np.exp(h * freqs / (k * T)) - 1)

def generate_fluctuation_field(size=64):
    np.random.seed(42)
    T = generate_Tn(size**2)
    Phi = compute_phi(T)
    field = Phi.reshape((size, size))
    field = np.nan_to_num(field)  # remove any NaNs/Infs
    return field - np.mean(field)


def compute_power_spectrum_2D(field):
    fft = np.fft.fft2(field)
    fft = np.nan_to_num(fft, nan=0.0, posinf=0.0, neginf=0.0)
    power = np.abs(fft)**2
    power = np.clip(power, 0, 1e12)  # avoid crazy spikes
    power_shifted = np.fft.fftshift(power)
    return power_shifted

def radial_average(power):
    y, x = np.indices(power.shape)
    center = np.array(power.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / np.maximum(nr, 1)
    return radial_prof

# Planck C_ell simulated benchmark
planck_ell = np.arange(2, 102)
planck_Cl = 1e-10 * (1/planck_ell) * np.exp(-(planck_ell - 200)**2 / (2 * 80**2))
planck_Cl /= np.max(planck_Cl)

# Run everything
field = generate_fluctuation_field()
power = compute_power_spectrum_2D(field)
Cl_RS = radial_average(power)
Cl_RS /= np.max(Cl_RS[:100])

vol = generate_recursive_volume(steps=20)
Pk_RS = compute_matter_power_spectrum(vol)
Cl_evolution = compute_growth_over_time(vol)
z_vals = redshift_mapping(len(Cl_evolution))

# Blackbody Spectrum
n_max = 100
T_vals = generate_Tn(n_max)
Phi_vals = compute_phi(T_vals)
freqs_RS, uRS = u_RS(T_vals, Phi_vals)
uP = planck(freqs_RS, 2.725)
uP /= np.max(uP)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(freqs_RS, uRS, label="RigbySpace Spectrum", lw=2)
plt.plot(freqs_RS, uP, label="Planck Spectrum (2.725K)", lw=2, ls='--')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Normalized Spectral Intensity")
plt.title("RigbySpace vs. Planck Blackbody Spectrum")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(Cl_RS[:100], label="RS Angular Power Spectrum", color='firebrick')
plt.plot(planck_ell, planck_Cl, label="Planck ΛCDM (simulated)", color='black', linestyle='--')
plt.xlabel("Multipole Moment (ℓ)")
plt.ylabel("Power")
plt.title("RigbySpace vs. Planck CMB Power Spectrum")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
for i in range(len(Cl_evolution)):
    label = f"z={z_vals[i]:.2f}" if i in [0, len(z_vals) - 1] else None
    plt.plot(Cl_evolution[i][:100], alpha=0.3 + 0.7 * (i / len(Cl_evolution)), label=label)
plt.plot(Pk_RS[:100], label="RS P(k) Final", color='navy', lw=2)
plt.xlabel("Wavenumber (k)")
plt.ylabel("Power")
plt.title("RigbySpace P(k) Evolution vs. Redshift")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
