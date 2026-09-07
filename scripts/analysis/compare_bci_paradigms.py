import sys
import os
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
_curr_dir = os.path.dirname(__file__)
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_itr(n_classes, accuracy, trial_duration_s=4.0):
    """
    Calculate Information Transfer Rate (ITR) in bits/minute (Wolpaw et al., 1998).
    """
    P = max(0.0001, min(0.9999, accuracy))
    N = n_classes
    if P <= 1.0 / N:
        return 0.0
    
    bits_per_trial = np.log2(N) + P * np.log2(P) + (1.0 - P) * np.log2((1.0 - P) / (N - 1.0))
    trials_per_min = 60.0 / trial_duration_s
    return max(0.0, bits_per_trial * trials_per_min)


def main():
    out_dir = os.path.abspath("analysis_results")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(" BCI PARADIGM DECODING COMPARISON BENCHMARK STUDIO ".center(80, "="))
    print("=" * 80)

    # Benchmark Data Summary
    paradigms = [
        "2-Class Motor Imagery\n(Left vs Right Limb)",
        "4-Class Motor Imagery\n(L/R/Feet/Tongue)",
        "2-Class Music Recall\n(Binary Music Pair)",
        "6-Track Music Recall\n(Real Performances)",
        "6-Track Music Recall\n(Synthesized Beats)"
    ]
    
    n_classes = [2, 4, 2, 6, 6]
    acc_scores = [85.0, 53.3, 65.1, 25.6, 25.6]  # % accuracy
    chance_levels = [100.0 / n for n in n_classes]

    itrs = [calculate_itr(n, acc / 100.0, trial_duration_s=4.0) for n, acc in zip(n_classes, acc_scores)]
    above_chance = [acc - chance for acc, chance in zip(acc_scores, chance_levels)]

    # Plot 1: Decoding Accuracy vs Chance Level
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(paradigms))
    width = 0.35

    bars1 = ax1.bar(x - width/2, acc_scores, width, label='BCI Decoding Accuracy (%)', color='#00E676', edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, chance_levels, width, label='Random Chance Level (%)', color='#A0A5B5', linestyle='--', edgecolor='black', alpha=0.6, linewidth=1.2)

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('BCI Decoding Performance: Motor Imagery vs. Music Memory Recall', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(paradigms, fontsize=10, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=11)

    for bar, acc in zip(bars1, acc_scores):
        ax1.text(bar.get_x() + bar.get_width()/2.0, acc + 1.5, f"{acc:.1f}%", ha='center', va='bottom', fontweight='bold', color='#00E676')

    for bar, ch in zip(bars2, chance_levels):
        ax1.text(bar.get_x() + bar.get_width()/2.0, ch + 1.5, f"{ch:.1f}%", ha='center', va='bottom', fontweight='bold', color='#636E72')

    plot1_path = os.path.join(out_dir, "bci_paradigm_decoding_comparison.png")
    fig.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: Information Transfer Rate (ITR bits/min)
    fig, ax2 = plt.subplots(figsize=(9, 5))
    bars_itr = ax2.bar(paradigms, itrs, color=['#74B9FF', '#E040FB', '#4DEEEA', '#FF7675'], width=0.55, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Information Transfer Rate (bits/min)', fontsize=12, fontweight='bold')
    ax2.set_title('BCI Information Transfer Rate (ITR in bits/min)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, itr_val in zip(bars_itr, itrs):
        ax2.text(bar.get_x() + bar.get_width()/2.0, itr_val + 0.5, f"{itr_val:.2f} bpm", ha='center', va='bottom', fontweight='bold')

    plot2_path = os.path.join(out_dir, "bci_itr_comparison.png")
    fig.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Executive Output
    print(f"\n[+] Saved Paradigm Accuracy Comparison plot: {plot1_path}")
    print(f"[+] Saved Information Transfer Rate plot   : {plot2_path}\n")

    print("=" * 80)
    print(" BCI PARADIGM COMPARISON EXECUTIVE SUMMARY ".center(80, "="))
    print("=" * 80)
    print(f"{'Paradigm':<35} | {'Classes':<8} | {'Accuracy':<10} | {'Chance':<8} | {'ITR (bits/min)':<14}")
    print("-" * 80)
    for p, n, acc, ch, itr in zip(paradigms, n_classes, acc_scores, chance_levels, itrs):
        p_clean = p.replace('\n', ' ')
        print(f"{p_clean:<35} | {n:<8} | {acc:<9.1f}% | {ch:<7.1f}% | {itr:<14.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
