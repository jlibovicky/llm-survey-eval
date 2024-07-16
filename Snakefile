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


rule run_survey:
    output:
        "survey_results/{model}/{lng}.{id}.json"
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
            if `timeout 1h python3 value_survey.py {params.model_name} {wildcards.lng} --seed $SEED > {output}`; then
                break
            fi
        done
        """
