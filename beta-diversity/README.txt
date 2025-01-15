This folder contains code to run Beta diversity analysis.

RPCA_ASV -> Robust Aitchison's distance. Use SG90_environment.txt in Python.
Phylogeny_ASV_based -> Phylogenetic robust Aitchison's distance. Use qiime2_environment.txt to run MAFFT and Fasttree (for details on parameters refer to manuscript). Use SG90_environment.txt to obtain beta diversity. Refer to Gemelli documentation on correct command-line code.
BrayCurtis -> Bray Curtis distance obtained using 1000-fold rarefaction. Also includes SIMPER code!
PERMANOVA -> Test associations between metadata and beta diversities, and adjust for potential confounders and time interaction effects
SIMPER -> Actual SIMPER code is in Bray Curtis file, however, this is an implementation that is possible to run on HPC servers - barring memory issues.
Plotting -> Code to plot all results.
