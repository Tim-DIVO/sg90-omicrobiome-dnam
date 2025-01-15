This repository contains code for ML models applied to oral microbiome data.

It should be used in the following sequence:

1. If using a binary classification problem, re-use code on assumption testing to evaluate whether sequencing depth differs between groups.
2. Models folder contains all models (main pipeline, confounder model, random baseline model)
3. HPC scripts contains variations of scripts to submit model training jobs to HPC servers
4. Model results contains results from HPC pipeline
5. Shapley feature importances were obtained using scripts in Shapley folder


For details on model selection, please refer to the manuscript.
