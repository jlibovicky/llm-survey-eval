# On the Credibility of Evaluating LLMs using Survey Questions

To replicate the results in the paper:

1. Download file `WVS_Cross-National_Wave_7_csv_v5_0.csv` from the [official webpage of the World Value Survey](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp).

2. Install requirements from `requirements.txt` and setup Snakemake to work with your environment (i.e., set up scheduling on your cluster or just run it locally on a machine with sufficiently big GPU).

3. Run all targets in the `Snakefile`.

After running the targets, outputs of the models will be in directory `model_outupts`, the quantitative results will be in `results`.
