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


def get_smoothed_distribution(value_counts, max_range, smoothing=0.01):
    smoothed_counts = {k: smoothing for k in range(1, max_range + 1)}
    for k, v in value_counts.items():
        smoothed_counts[k] += v
    return {k: v / sum(smoothed_counts.values()) for k, v in smoothed_counts.items()}


def compare_survey_and_model(included_questions, max_ranges, country, df_llm_results):
    all_mses = []
    all_kl_divergences = []
    p_values = []

    # Compare frames
    for question_id, max_range in zip(included_questions, max_ranges):
        survey_answers = country[question_id]
        survey_answers = survey_answers[survey_answers >= 0]

        # The original questionaire allowed a custom answer for Q111 and we ignore that.
        if question_id == "Q111":
            survey_answers = survey_answers[survey_answers < 3]
        # Q122-Q129 start from zero in the questionaire, but we wanted to have them
        # consistent with the other questions.
        if question_id in [f"Q{i}" for i in range(122, 130)]:
            survey_answers = survey_answers.apply(lambda x: x + 1)

        survey_answer_counts = survey_answers.value_counts(normalize=False).to_dict()
        survey_answer_distribution = get_smoothed_distribution(
            survey_answer_counts, max_range, smoothing=0.0)

        llm_answers = df_llm_results[question_id]
        llm_answers = llm_answers.dropna()
        llm_answer_counts = llm_answers.value_counts(normalize=False).to_dict()
        llm_answer_distribution = get_smoothed_distribution(
            llm_answer_counts, max_range, smoothing=0.01)

        # Compute normalized MSE
        min_val = min(survey_answers.min(), llm_answers.min())
        max_val = max(survey_answers.max(), llm_answers.max())
        mse = (
            ((survey_answers - min_val) / (max_val - min_val)).mean() -
            ((llm_answers - min_val) / (max_val - min_val)).mean()) ** 2
        all_mses.append(mse)

        # Compute KL-divergence between the survey and the LLM
        kl_diveregence = 0
        for key in range(1, max_range + 1):
            if survey_answer_distribution[key] == 0:
                continue
            kl_diveregence += (
                survey_answer_distribution[key] *
                np.log(survey_answer_distribution[key] /
                    llm_answer_distribution[key]))
        all_kl_divergences.append(kl_diveregence)

        # Compute the p-value of the Chi-squared test
        llm_count = len(llm_answers)
        contingency_table = []
        for key in range(1, max_range + 1):
            new_line = [
                survey_answer_counts.get(key, 0),
                llm_answer_counts.get(key, 0)]
            if sum(new_line) > 0:
                contingency_table.append(new_line)
        contingency_table = np.array(contingency_table)
        if contingency_table[:, 1].sum() == 0:
            continue
        p_values.append(chi2_contingency(contingency_table)[1])

    mse = np.mean(all_mses)
    kl_divergence = np.mean(all_kl_divergences)

    # TODO use permutation test for the p-value of mse and kl_divergence

    return mse, kl_divergence, np.mean(p_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_outputs", nargs="+",
        help="The JSON files with the LLM outputs.")
    args = parser.parse_args()

    logging.info("Read the survey configuration.")
    with open('questions.en.txt', 'r') as f:
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

    country = survey_results[survey_results["B_COUNTRY_ALPHA"] == "USA"]

    logging.info("Compare the survey and the LLM for different result counts.")
    for i in range(1,len(df_llm_results) + 1):
        mse, kl_divergence, p_val = compare_survey_and_model(
            included_questions, max_ranges, country, df_llm_results.head(i))
        print(f"{i:03d}: {mse:.4f}, {kl_divergence:.4f}, {p_val:.4f}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
