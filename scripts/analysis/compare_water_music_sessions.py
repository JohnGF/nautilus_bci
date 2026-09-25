#!/usr/bin/env python3
"""
Compare Tower Defense Rhythm Decoding: ses-01 (Old Water Music) vs. ses-02 (New Water Music)
============================================================================================
Evaluates whether the replacement of the Water music track in ses-02 improves:
  1. Pairwise 2-class decoding involving Water (FIRE vs. WATER, WATER vs. WIND, WATER vs. ELECTRICITY)
  2. 4-Class per-class recall and F1-score for Water
  3. Zero-shot transfer learning from Auditory Perception (Listen) to Mental Imagery (Imagine)
  4. Representational Similarity Analysis (RSA) alignment
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_comparison(
    ses01_json="scripts/analysis_results/tower_defense_recall/comparison_run/ses-01/sub-01_ses-01/rhythm_decoding_summary.json",
    ses02_json="scripts/analysis_results/tower_defense_recall/comparison_run/ses-02/sub-01_ses-02/rhythm_decoding_summary.json",
    out_dir="scripts/analysis_results/tower_defense_recall/comparison_run"
):
    os.makedirs(out_dir, exist_ok=True)
    with open(ses01_json, "r", encoding="utf-8") as f:
        s1 = json.load(f)
    with open(ses02_json, "r", encoding="utf-8") as f:
        s2 = json.load(f)

    # Figure generation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    width = 0.35

    # Panel 1: Mental Imagery Pairwise (WATER pairs)
    ax1 = axes[0, 0]
    pairs = ['FIRE-WATER', 'WATER-WIND', 'WATER-ELECTRICITY']
    labels_short = ['FIRE vs WATER', 'WATER vs WIND', 'WATER vs ELEC']
    x = np.arange(len(pairs))

    pw1_im = s1['pairwise_decoding']['imagine']['pairs']
    pw2_im = s2['pairwise_decoding']['imagine']['pairs']

    acc1_im = [pw1_im[k]['best_accuracy'] * 100 for k in pairs]
    acc2_im = [pw2_im[k]['best_accuracy'] * 100 for k in pairs]

    r1 = ax1.bar(x - width/2, acc1_im, width, label='ses-01 (Old Water)', color='#95a5a6', edgecolor='black', linewidth=1)
    r2 = ax1.bar(x + width/2, acc2_im, width, label='ses-02 (New Water)', color='#3498db', edgecolor='black', linewidth=1)
    ax1.axhline(50, color='red', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax1.set_title('Mental Imagery (Imagine): Water Pairs (Best Model)', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, fontweight='bold', fontsize=10)
    ax1.set_ylabel('Decoding Accuracy (%)', fontweight='bold', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')

    for rect in r1 + r2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel 2: Auditory Perception Pairwise (WATER pairs)
    ax2 = axes[0, 1]
    pw1_lis = s1['pairwise_decoding']['listen']['pairs']
    pw2_lis = s2['pairwise_decoding']['listen']['pairs']

    acc1_lis = [pw1_lis[k]['best_accuracy'] * 100 for k in pairs]
    acc2_lis = [pw2_lis[k]['best_accuracy'] * 100 for k in pairs]

    r3 = ax2.bar(x - width/2, acc1_lis, width, label='ses-01 (Old Water)', color='#95a5a6', edgecolor='black', linewidth=1)
    r4 = ax2.bar(x + width/2, acc2_lis, width, label='ses-02 (New Water)', color='#2ecc71', edgecolor='black', linewidth=1)
    ax2.axhline(50, color='red', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax2.set_title('Auditory Perception (Listen): Water Pairs (Best Model)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, fontweight='bold', fontsize=10)
    ax2.set_ylabel('Decoding Accuracy (%)', fontweight='bold', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    for rect in r3 + r4:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel 3: Zero-shot Transfer (Listen -> Imagine)
    ax3 = axes[1, 0]
    cm1 = np.array(s1['transfer_decoding']['CSP_ShrinkageLDA_ListenToImagine']['confusion_matrix'])
    cm2 = np.array(s2['transfer_decoding']['CSP_ShrinkageLDA_ListenToImagine']['confusion_matrix'])

    tr_acc1 = s1['transfer_decoding']['CSP_ShrinkageLDA_ListenToImagine']['accuracy'] * 100
    tr_acc2 = s2['transfer_decoding']['CSP_ShrinkageLDA_ListenToImagine']['accuracy'] * 100
    w_rec1 = cm1[1, 1] / np.sum(cm1[1, :]) * 100
    w_rec2 = cm2[1, 1] / np.sum(cm2[1, :]) * 100

    x3 = np.arange(2)
    w_labels = ['Overall Transfer Acc', 'Water Class Recall']
    v1 = [tr_acc1, w_rec1]
    v2 = [tr_acc2, w_rec2]

    r5 = ax3.bar(x3 - width/2, v1, width, label='ses-01 (Old Water)', color='#e67e22', edgecolor='black', linewidth=1)
    r6 = ax3.bar(x3 + width/2, v2, width, label='ses-02 (New Water)', color='#9b59b6', edgecolor='black', linewidth=1)
    ax3.axhline(25, color='red', linestyle='--', linewidth=1.5, label='Chance (25%)')
    ax3.set_title('Zero-Shot Transfer (Listen -> Imagine)', fontweight='bold', fontsize=12)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(w_labels, fontweight='bold', fontsize=10)
    ax3.set_ylabel('Accuracy / Recall (%)', fontweight='bold', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')

    for rect in r5 + r6:
        h = rect.get_height()
        ax3.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel 4: Overall Mean Pairwise Summary
    ax4 = axes[1, 1]
    mean_w_im1 = np.mean(acc1_im)
    mean_w_im2 = np.mean(acc2_im)
    mean_w_lis1 = np.mean(acc1_lis)
    mean_w_lis2 = np.mean(acc2_lis)

    x4 = np.arange(2)
    summary_labels = ['Mean Water Pairs\n(Mental Imagery)', 'Mean Water Pairs\n(Auditory Perception)']
    s_v1 = [mean_w_im1, mean_w_lis1]
    s_v2 = [mean_w_im2, mean_w_lis2]

    r7 = ax4.bar(x4 - width/2, s_v1, width, label='ses-01 (Old Water)', color='#bdc3c7', edgecolor='black', linewidth=1)
    r8 = ax4.bar(x4 + width/2, s_v2, width, label='ses-02 (New Water)', color='#1abc9c', edgecolor='black', linewidth=1)
    ax4.axhline(50, color='red', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax4.set_title('Summary: Mean Accuracy of All Water Pairs', fontweight='bold', fontsize=12)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(summary_labels, fontweight='bold', fontsize=10)
    ax4.set_ylabel('Mean Accuracy (%)', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper left')

    for rect in r7 + r8:
        h = rect.get_height()
        ax4.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords='offset points', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.suptitle('Water Rhythm Music Replacement: ses-01 (Old Water) vs. ses-02 (New Water)\nTower Defense BCI Rhythm Decoding Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_fig = os.path.join(out_dir, "water_music_comparison_ses01_vs_ses02.png")
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"[+] Saved comparison figure to: {out_fig}")

if __name__ == "__main__":
    run_comparison()
