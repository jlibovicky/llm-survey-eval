IDS = [f"{i:03d}" for i in range(1, 401)]

MODELS = ["llama3"] #, "mistral"]#, "gemma"]

MODEL_IDS = {
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "gemma": "google/gemma-7b"
}


rule all:
    input:
        expand("survey_results/{model}/{lng}.{id}.json",
               model=MODELS, lng=["en"], id=IDS)


rule run_survey_sample:
    output:
        "survey_results/{prompt_type}/{model}/{lng}.{id}.json"
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
        "survey_results/{prompt_type}/{model}/{lng}.greedy.json"
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
            python3 value_survey.py {params.model_name} {wildcards.lng} {wildcards.prompt_type} --seed $SEED --greedy > {output}
        """

rule measure_convergence:
    input:
        expand("survey_results/{{prompt_type}}/llama3/en.{id}.json", id=range(1, 401))
    output:
        "results/llama3_en_USA_{prompt_type}_course.csv"
    shell:
        "python3 python3 compare_survey_and_model.py USA {input} > {output}"

rule measure_greedy:
    input:
        survey_results="survey_results/{prompt_type}/llama3/en.greedy.json",
    output:
        "results/llama3_en_USA_{prompt_type}_greedy.csv"
    shell:
        "python3 compare_survey_and_model.py USA {input.survey_results} > {output}"