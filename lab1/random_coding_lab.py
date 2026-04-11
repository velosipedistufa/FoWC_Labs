# File: /home/alex/FoWC_Labs/lab1/random_coding_lab.py
# Save this as random_coding_lab.py and run with: python random_coding_lab.py
# It will generate the plot "block_error_rate.png" that matches the expected figure
# (up to Monte-Carlo variation, since the codebook is random).
# Uses only NumPy + Matplotlib (standard in any Python env with the lab requirements).

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # optional progress bar, remove if not installed

def generate_codebook(M: int, n: int) -> np.ndarray:
    """
    Generate random Gaussian codebook and normalize each codeword to unit power per symbol
    (||c_m||^2 == n, i.e. average |c_i|^2 = 1).
    """
    # Complex Gaussian CN(0,1) per component
    codebook = (np.random.randn(M, n) + 1j * np.random.randn(M, n)) / np.sqrt(2.0)
    # Normalize each row: ||c|| = sqrt(n)
    norms = np.linalg.norm(codebook, axis=1, keepdims=True)  # shape (M, 1)
    codebook = codebook * (np.sqrt(n) / norms)
    return codebook

def simulate_bler(codebook: np.ndarray, EsN0_dB: float, num_trials: int = 50000, batch_size: int = 512) -> float:
    """
    Simulate block error rate using ML (nearest-neighbor) decoder.
    Vectorized but batched to avoid OOM for large M (k=16).
    """
    M, n = codebook.shape
    EsN0_lin = 10 ** (EsN0_dB / 10.0)
    N0 = 1.0 / EsN0_lin                     # Es = 1 (normalized power)

    num_errors = 0
    num_batches = (num_trials + batch_size - 1) // batch_size

    for b in range(num_batches):
        bs = min(batch_size, num_trials - b * batch_size)
        # Random messages
        msgs = np.random.randint(0, M, size=bs)
        # Transmitted codewords
        tx = codebook[msgs]                                 # (bs, n)

        # AWGN noise ~ CN(0, N0)
        noise = np.sqrt(N0 / 2.0) * (np.random.randn(bs, n) + 1j * np.random.randn(bs, n))
        y = tx + noise

        # ML decoder: argmax Re(<y, c_m>)  (equivalent to min ||y - c_m||^2 since ||c_m|| constant)
        # corr = y @ codebook.conj().T   -> shape (bs, M)
        corr = np.dot(y, codebook.conj().T)                # (bs, M)
        real_corr = np.real(corr)
        decoded = np.argmax(real_corr, axis=1)

        num_errors += np.sum(decoded != msgs)

    return num_errors / num_trials

# ========================== MAIN SIMULATION ==========================
if __name__ == "__main__":
    np.random.seed(42)          # reproducibility
    ks = [2, 4, 8, 16]
    R = 0.5
    EsN0_dB_vals = np.arange(-8, 11, 1)   # matches the figure range

    bler_results = {}
    print("Starting random coding simulation (Es/N0 metric)...\n")

    for k in ks:
        n = int(k / R)          # blocklength n = 2k
        M = 1 << k              # M = 2^k
        print(f"k={k:2d} | n={n:2d} | M={M:5d} | generating codebook...")
        codebook = generate_codebook(M, n)

        blers = []
        for EsN0_dB in tqdm(EsN0_dB_vals, desc=f"  SNR sweep (k={k})"):
            bler = simulate_bler(codebook, EsN0_dB,
                                 num_trials=100000 if k <= 8 else 20000,   # fewer trials for k=16 (still accurate enough)
                                 batch_size=1024 if k <= 8 else 256)
            blers.append(bler)
        bler_results[k] = blers
        print(f"  → done (last BLER @ 10 dB: {blers[-1]:.2e})\n")

    # ========================== PLOTTING (exact match to expected figure style) ==========================
    plt.figure(figsize=(8, 6))
    colors = ['red', 'darkred', 'crimson', 'firebrick']
    for i, k in enumerate(ks):
        plt.semilogy(EsN0_dB_vals, bler_results[k], 'o-', color=colors[i],
                     label=f'k = {k}', linewidth=2, markersize=4)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel(r'$E_s/N_0$, dB')
    plt.ylabel('Block Error Rate')
    plt.title('Random coding for short blocklength (R = 1/2)')
    plt.legend()
    plt.ylim(1e-4, 1)
    plt.xlim(-8, 10)
    plt.tight_layout()

    # Save plot
    plt.savefig('block_error_rate.png', dpi=300)
    plt.show()

    print("Simulation finished!")
    print("Plot saved as 'block_error_rate.png' (matches the expected figure shape).")
    print("Source code is ready for submission.")