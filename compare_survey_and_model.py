#!/usr/bin/env python3

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# Logging INFO with the format: "2021-07-01 12:00:00 - INFO - Message"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO)


COUNTRIES = [
    'AND', 'ARG', 'AUS', 'BGD', 'ARM', 'BOL', 'BRA', 'MMR', 'CAN', 'CHL', 'CHN',
    'TWN', 'COL', 'CYP', 'CZE', 'ECU', 'ETH', 'DEU', 'GRC', 'GTM', 'HKG', 'IDN',
    'IRN', 'IRQ', 'JPN', 'KAZ', 'JOR', 'KEN', 'KOR', 'KGZ', 'LBN', 'LBY', 'MAC',
    'MYS', 'MDV', 'MEX', 'MNG', 'MAR', 'NLD', 'NZL', 'NIC', 'NGA', 'PAK', 'PER',
    'PHL', 'PRI', 'ROU', 'RUS', 'SRB', 'SGP', 'SVK', 'VNM', 'ZWE', 'TJK',
    'THA', 'TUN', 'TUR', 'UKR', 'EGY', 'GBR', 'USA', 'URY', 'VEN', 'NIR']


def get_smoothed_distribution(
        value_counts: dict[int, int], max_range: int, smoothing=0.01) -> dict[int, float]:
    smoothed_counts = {k: smoothing for k in range(1, max_range + 1)}
    for k, v in value_counts.items():
        smoothed_counts[k] += v
    return {k: v / sum(smoothed_counts.values()) for k, v in smoothed_counts.items()}


def standardize_survey_answers(question_id: str, answers: pd.Series) -> pd.Series:
    survey_answers = answers[answers >= 0]

    # The original questionaire allowed a custom answer for Q111 and we ignore that.
    if question_id == "Q111":
        survey_answers = survey_answers[survey_answers < 3]
    # Q122-Q129 start from zero in the questionaire, but we wanted to have them
    # consistent with the other questions.
    if question_id in [f"Q{i}" for i in range(122, 130)]:
        survey_answers = survey_answers.apply(lambda x: x + 1)
    survey_answers = survey_answers[survey_answers >= 1]

    return survey_answers


def compare_survey_and_model(
        included_questions: list[str], max_ranges: list[int],
        country_data: pd.DataFrame,
        compare_data: pd.DataFrame,
        compare_is_country: bool = False) -> tuple[float, float, float, float, float]:
    all_mses = []
    all_kl_divergences = []
    stds = []
    std_diffs = []
    p_values = []

    # Compare frames
    for question_id, max_range in zip(included_questions, max_ranges):
        survey_answers = country_data[question_id]
        survey_answers = standardize_survey_answers(question_id, survey_answers)

        if len(survey_answers) == 0:
            logging.warning("No survey answers for question %s.", question_id)
            continue

        survey_answer_counts = survey_answers.value_counts(normalize=False).to_dict()
        survey_answer_distribution = get_smoothed_distribution(
            survey_answer_counts, max_range, smoothing=0.01)

        compare_answers = compare_data[question_id]
        compare_answers = compare_answers.dropna()

        if compare_is_country:
            compare_answers = standardize_survey_answers(question_id, compare_answers)
            if len(compare_answers) == 0:
                logging.warning("No survey answers for question %s.", question_id)
                continue

        compare_answer_counts = compare_answers.value_counts(normalize=False).to_dict()
        compare_answer_distribution = get_smoothed_distribution(
            compare_answer_counts, max_range, smoothing=0.01)

        # Compute normalized MSE
        min_val = min(survey_answers.min(), compare_answers.min())
        max_val = max(survey_answers.max(), compare_answers.max())
        mse = (
            ((survey_answers - min_val) / (max_val - min_val)).mean() -
            ((compare_answers - min_val) / (max_val - min_val)).mean()) ** 2
        all_mses.append(mse)

        # Compute KL-divergence between the survey and the LLM
        kl_divergence = 0
        for key in range(1, max_range + 1):
            if survey_answer_distribution[key] == 0:
                continue
            kl_divergence += (
                survey_answer_distribution[key] *
                np.log(survey_answer_distribution[key] /
                    compare_answer_distribution[key]))
        all_kl_divergences.append(kl_divergence)

        # Compute the p-value of the Chi-squared test
        contingency_table = []
        for key in range(1, max_range + 1):
            new_line = [
                survey_answer_counts.get(key, 0),
                compare_answer_counts.get(key, 0)]
            if sum(new_line) > 0:
                contingency_table.append(new_line)
        contingency_table = np.array(contingency_table)
        if contingency_table[:, 1].sum() == 0:
            continue
        p_values.append(chi2_contingency(contingency_table)[1])

        # Stdev difference
        compare_std = (compare_answers / max_range).std()
        stds.append(compare_std)
        survey_std = (survey_answers / max_range).std()
        std_diffs.append(compare_std - survey_std)

    mse = float(np.nanmean(all_mses))
    kl_divergence = float(np.nanmean(all_kl_divergences))
    std = float(np.mean(stds))
    std_diff = float(np.mean(std_diffs))
    mean_p_value = float(np.mean(p_values))

    return mse, std, std_diff, kl_divergence, mean_p_value


def load_questions_ranges_and_survey_results() -> tuple[list[str], list[int], pd.DataFrame]:
    logging.info("Read the survey configuration.")
    with open('prompts/cot.en.txt', 'r') as f:
        included_questions = [l.strip().split(";")[0] for l in f.readlines()]
    with open('validators.txt', 'r') as f:
        max_ranges = []
        for line in f:
            if line.strip() == "1or2":
                max_ranges.append(2)
            elif "to" in line:
                max_ranges.append(int(line.split("to")[1]))
            else:
                raise ValueError("Invalid line in validators.txt: " + line)

    logging.info("Read the survey results.")
    survey_results = pd.read_csv(
        "./WVS_Cross-National_Wave_7_csv_v5_0.csv", low_memory=False)
    return included_questions, max_ranges, survey_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "country", type=str, choices=COUNTRIES,
        help="The country code from the survey."
    )
    parser.add_argument(
        "model_outputs", nargs="+",
        help="The JSON files with the LLM outputs.")
    parser.add_argument(
        "--all-only", action="store_true", default=False,
        help="Use all the LLM outputs.")
    args = parser.parse_args()

    included_questions, max_ranges, survey_results = load_questions_ranges_and_survey_results()

    logging.info("Read the LLM from %d JSON files.", len(args.model_outputs))
    llm_results = []
    for path in args.model_outputs:
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
                llm_results.append(data["results"])
        except:
            logging.error("Failed to read the LLM results from %s.", path)
    df_llm_results = pd.DataFrame(llm_results)

    country = survey_results[survey_results["B_COUNTRY_ALPHA"] == args.country]

    logging.info("Compare the survey and the LLM for different result counts.")
    start_index = len(df_llm_results) if args.all_only else 1
    for i in range(start_index, len(df_llm_results) + 1):
        mse, std, std_diff, kl_divergence, p_val = compare_survey_and_model(
            included_questions, max_ranges, country, df_llm_results.head(i))
        print(f"{mse:.4f},{std:.4f},{std_diff:.4f},{kl_divergence:.4f},{p_val:.4f}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
