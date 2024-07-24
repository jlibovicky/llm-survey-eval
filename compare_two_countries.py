#!/usr/bin/env python3

import argparse
import logging

import pandas as pd

from compare_survey_and_model import compare_survey_and_model, load_questions_ranges_and_survey_results, COUNTRIES


# Logging INFO with the format: "2021-07-01 12:00:00 - INFO - Message"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("country_1", help="Country 1", choices=COUNTRIES)
    parser.add_argument("country_2", help="Country 2", choices=COUNTRIES)
    args = parser.parse_args()

    included_questions, max_ranges, survey_results = load_questions_ranges_and_survey_results()

    country_1 = survey_results[survey_results["B_COUNTRY_ALPHA"] == args.country_1]
    country_2 = survey_results[survey_results["B_COUNTRY_ALPHA"] == args.country_2]

    logging.info("Compare two countries.")
    mse, std, std_diff, kl_divergence, p_val = compare_survey_and_model(
        included_questions, max_ranges, country_1, country_2, compare_is_country=True)

    print(f"{mse:.4f}, {std:.4f}, {std_diff:.4f}, {kl_divergence:.4f}, {p_val:.4f}")
    logging.info("Done.")


if __name__ == "__main__":
    main()