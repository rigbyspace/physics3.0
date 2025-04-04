from fractions import Fraction
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt

getcontext().prec = 60

# RS Seed Constants
VEN = Fraction(22, 7)
LUC = Fraction(7, 19)
DELTA = Fraction(1, 11)

# Theta_RS reference ratio for curvature symmetry
THETA_RS = Decimal(22) / Decimal(19)

# Seed shells
seeds = [VEN, LUC]
# Alternating delta: [1/11, 11/1, 1/11, ...]
deltas = [Fraction(1, 11), Fraction(11, 1)]

def generate_rs_shells(n):
    shells = seeds.copy()
    for i in range(2, n):
        delta = deltas[i % 2]
        next_val = shells[-1] + shells[-2] + delta
        shells.append(next_val)
    return shells

def extended_wobble_ratios(shells):
    ratios = []
    theta_hits = []
    for i in range(2, len(shells)):
        num = shells[i] - 2 * shells[i - 1] + shells[i - 2]
        denom = shells[i - 1]
        ratio = Decimal(num.numerator) / Decimal(num.denominator) / (
            Decimal(denom.numerator) / Decimal(denom.denominator))
        ratios.append(float(ratio))

        # Track when curvature approaches Theta_RS
        shell_ratio = Decimal(shells[i].numerator) / Decimal(shells[i].denominator)
        if abs(shell_ratio - THETA_RS) < Decimal("0.0005"):
            theta_hits.append((i, float(shell_ratio)))

    return ratios, theta_hits

def plot_wobble(wobble_seq, theta_hits):
    steps = list(range(2, len(wobble_seq)+2))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, wobble_seq, marker='o', label='RS 2nd-Deriv Wobble Ratio')
    plt.axhline(y=0, color='gray', linestyle='--', label='Zero Crossing')
    for idx, theta_val in theta_hits:
        plt.axvline(x=idx, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Recursive Step')
    plt.ylabel('α_n = (Tn - 2Tn-1 + Tn-2)/Tn-1')
    plt.title('Rigby Space Recursive Compression Slope with Θ_RS Anchors')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_steps = 40
    shells = generate_rs_shells(n_steps)
    wobble_seq, theta_hits = extended_wobble_ratios(shells)
    plot_wobble(wobble_seq, theta_hits)

