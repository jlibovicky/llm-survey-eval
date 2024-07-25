#!/usr/bin/env python3

import argparse

import numpy as np
import matplotlib.pyplot as plt

from compare_survey_and_model import (
    load_questions_ranges_and_survey_results, load_model_outputs_to_dataframe,
    COUNTRIES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("country", choices=COUNTRIES, help="Country to plot the self-correlation for.")
    parser.add_argument("output_file", type=str, help="PDF file to save the plot to.")
    parser.add_argument("model_outputs", nargs="+", type=str, help="List of model output files to compare.")
    args = parser.parse_args()

    questions, ranges, survey_results = load_questions_ranges_and_survey_results()
    country_results = survey_results[survey_results["B_COUNTRY_ALPHA"] == args.country]
    corr_matrix = country_results[questions].corr()

    model_answers = load_model_outputs_to_dataframe(args.model_outputs)[questions]
    model_corr = model_answers.corr().fillna(0)

    combined = np.tril(corr_matrix, -1) + np.triu(model_corr.fillna(0), 1)
    fig, ax = plt.subplots(figsize=(3.03, 3.03))
    ax.matshow(combined, vmin=-1, vmax=1, cmap=plt.cm.seismic)
    ax.set_xlabel("◺ Survey answer self-correlation")

    # Hide all ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax2 = ax.twinx()
    ax2.set_yticks([])

    # Move x-axis to top
    ax3 = ax2.twiny()
    ax3.set_xticks([])
    ax3.xaxis.set_ticks_position('top')
    ax3.xaxis.set_label_position('top')
    ax3.set_xlabel("◹ Model answer self-correlation")

    plt.savefig(args.output_file, bbox_inches="tight")


if __name__ == "__main__":
    main()