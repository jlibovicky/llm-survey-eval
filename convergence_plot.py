#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cot_course", type=str, help="Path to course of experiments with CoT prompting.")
    parser.add_argument("cot_greedy", type=str, help="Path to greedy experiment with CoT prompting.")
    parser.add_argument("score_course", type=str, help="Path to course of experiments with score prompting.")
    parser.add_argument("score_greedy", type=str, help="Path to greedy experiment with score prompting.")
    parser.add_argument("output", type=str, help="Path to the output PDF plot.")
    args = parser.parse_args()

    with open(args.cot_course) as f:
        data_cot = np.genfromtxt(f, delimiter=",", skip_header=0)

    with open(args.score_course) as f:
        data_score_only = np.genfromtxt(f, delimiter=",", skip_header=0)

    with open(args.cot_greedy) as f:
        greedy_cot = np.genfromtxt(f, delimiter=",", skip_header=0)

    with open(args.score_greedy) as f:
        greedy_score_only = np.genfromtxt(f, delimiter=",", skip_header=0)

    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    color1 = "tab:blue"
    ax1.plot(data_cot[:,0], color=color1)
    ax1.plot(data_score_only[:,0], color=color1, linestyle="--")
    # Y label in bold
    ax1.set_ylabel("Mean squared difference", color=color1, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlabel("Number of samples")
    ax1.set_xlim((0, 400))
    ax1.set_ylim((0.05, 0.12))

    ax2 = ax1.twinx()

    color2 = "tab:red"
    ax2.plot(data_cot[:,3], color=color2)
    ax2.plot(data_score_only[:,3], color=color2, linestyle="--")
    ax2.scatter([380], [greedy_cot[3]], color=color2, marker="$-$", s=500)

    ax2.annotate("Greedy, CoT ⟶", (380, greedy_cot[3]), textcoords="offset points", xytext=(-15, -3), ha='right', color=color2)
    ax2.scatter([380], [greedy_score_only[3]], color=color2, marker="$---$", s=500, linewidths=2)
    ax2.annotate("Greedy, scores only ⟶", (380, greedy_score_only[3]), textcoords="offset points", xytext=(-15, -3), ha='right', color=color2)
    ax2.set_ylabel("KL divergence", color=color2, fontweight="bold")

    ax1.scatter([21], [greedy_cot[0]], color=color1, marker="$-$", s=500, linewidths=2)
    ax1.annotate("⟵ Greedy, CoT", (21, greedy_cot[0]), textcoords="offset points", xytext=(15, -3), ha='left', color=color1)
    ax1.scatter([21], [greedy_score_only[0]], color=color1, marker="$---$", s=500, linewidths=2)
    ax1.annotate("⟵ Greedy, scores only", (21, greedy_score_only[0]), textcoords="offset points", xytext=(15, -3), ha='left', color=color1)

    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([1.3, 3.2])

    legend_lines = [plt.Line2D([0], [0], color='gray', linestyle='--', label='Score only'),
                    plt.Line2D([0], [0], color='gray', label='Chain of thought')]

    plt.legend(legend_lines, ["Score only", "Chain of thought"], loc="center right", bbox_to_anchor=(1, .6))

    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()