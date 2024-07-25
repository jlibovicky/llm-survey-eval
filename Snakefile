IDS = [f"{i:03d}" for i in range(1, 401)]

MODELS = ["llama3"] #, "mistral"]#, "gemma"]

MODEL_IDS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "gemma": "google/gemma-7b"
}

LANGUAGES = ["en", "cs", "de"]
SELECTED_COUNTRIES = ["USA", "GBR", "CZE", "DEU", "IRN", "CHN"]
PROMPTS = ["score", "cot"]


rule all:
    input:
        expand("model_outputs/{model}/{lng}.{id}.json",
               model=MODELS, lng=["en"], id=IDS)


wildcard_constraints:
    id="|".join(IDS),
    model="|".join(MODEL_IDS),
    prompt_type="|".join(PROMPTS),
    lng="|".join(LANGUAGES),


rule run_survey_sample:
    output:
        "model_outputs/{prompt_type}/{model}/{lng}.{id}.json"
    params:
        model_name=lambda wildcards: MODEL_IDS[wildcards.model],
    resources:
        mem="20G",
        cpus_per_task=2,
        slurm_partition="gpu-troja,gpu-ms",
        constraint="'gpuram48G'",
        slurm_extra="'--gres=gpu:1'",
    shell:
        """
        # Some seeds are unlucky and the models does not generate results in the
        # correct format, so we try multiple seeds.

        for I in {{1..10}}; do
            SEED=$( echo 1000 \\* $I + {wildcards.id} | bc )
            if `timeout 2h python3 value_survey.py {params.model_name} {wildcards.lng} {wildcards.prompt_type} --seed $SEED > {output}`; then
                break
            fi
        done

        # If output is still empty, we failed
        if [ ! -s {output} ]; then
            exit 1
        fi
        """

rule run_survey_greedy:
    output:
        "model_outputs/{prompt_type}/{model}/{lng}.greedy.json"
    params:
        model_name=lambda wildcards: MODEL_IDS[wildcards.model],
    resources:
        mem="20G",
        cpus_per_task=2,
        slurm_partition="gpu-troja,gpu-ms",
        constraint="'gpuram48G'",
        slurm_extra="'--gres=gpu:1'",
    shell:
        """
            python3 value_survey.py {params.model_name} {wildcards.lng} {wildcards.prompt_type} --greedy > {output}
        """


rule measure_convergence:
    input:
        expand("model_outputs/{{prompt_type}}/llama3/en.{id}.json",
        id=IDS)
    output:
        "results/llama3_en_USA_{prompt_type}_course.csv"
    shell:
        "python3 compare_survey_and_model.py USA {input} > {output}"


rule measure_greedy:
    input:
        model_outputs="survey_results/{prompt_type}/llama3/en.greedy.json",
    output:
        "results/llama3_en_USA_{prompt_type}_greedy.csv"
    shell:
        "python3 compare_survey_and_model.py USA {input.model_outputs} > {output}"


rule plot_convergence:
    input:
        cot_course="results/llama3_en_USA_cot_course.csv",
        cot_greedy="results/llama3_en_USA_cot_greedy.csv",
        score_course="results/llama3_en_USA_score_course.csv",
        score_greedy="results/llama3_en_USA_score_greedy.csv",
    output:
        "results/llama3_en_USA_convergence.pdf"
    shell:
        "python3 convergence_plot.py {input.cot_course} {input.cot_greedy} {input.score_course} {input.score_greedy} {output}"


rule compare_model_to_survey_sample:
    input:
        model_outputs=expand(
            "model_outputs/{{prompt_type}}/{{model}}/{{lng}}.{id}.json", id=IDS[:200]),
    output:
        "results/{model}_{lng}_{country}_{prompt_type}_sample_final.csv"
    shell:
        "python3 compare_survey_and_model.py {wildcards.country} {input.model_outputs} --all-only > {output}"


rule compare_model_to_survey_greedy:
    input:
        model_output="model_outputs/{prompt_type}/{model}/{lng}.greedy.json",
    output:
        "results/{model}_{lng}_{country}_{prompt_type}_greedy_final.csv"
    shell:
        "python3 compare_survey_and_model.py {wildcards.country} {input.model_output} --all-only > {output}"


rule survey_model_table:
    input:
        expand("results/{{model}}_{lng}_{country}_{prompt_type}_{decoding}_final.csv",
               lng=LANGUAGES, country=SELECTED_COUNTRIES, prompt_type=PROMPTS,
               decoding=["greedy", "sample"]),
    output:
        "results/{model}_compare_table.csv"
    run:
        import numpy as np

        output = open(f"{output}", "w")
        print("Language,Prompt,Decoding," +
              ",".join([f"{c}_mse" for c in SELECTED_COUNTRIES]) + "," + 
              ",".join([f"{c}_stddiff" for c in SELECTED_COUNTRIES]) + "," +
              ",".join([f"{c}_kldiv" for c in SELECTED_COUNTRIES]), file=output)

        for prompt_type in PROMPTS:
            for lang in LANGUAGES:
                for decoding in ["greedy", "sample"]:
                    mses = []
                    stdiffs = []
                    kl_divs = []
                    for country in SELECTED_COUNTRIES:
                        with open(f"results/{wildcards.model}_{lang}_{country}_{prompt_type}_{decoding}_final.csv") as f:
                            scores = np.genfromtxt(f, delimiter=",", skip_header=0)
                            mses.append(scores[0])
                            stdiffs.append(scores[2])
                            kl_divs.append(scores[3])
                    print(f"{lang},{prompt_type},{decoding}," +
                          ",".join([str(x) for x in mses]) + "," +
                          ",".join([str(x) for x in stdiffs]) + "," +
                          ",".join([str(x) for x in kl_divs]), file=output)
        output.close()