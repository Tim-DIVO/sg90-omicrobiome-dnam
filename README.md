# sg90-omicrobiome-dnam
All scripts from the analysis of my MSc thesis - Salivary Signatures of Longevity: Oral Short-Chain Fatty Acid-Producing Bacteria Associated with Biological Age and Cognition Among the Oldest Old

Repository is organised by analysis type:
- Data preparation (preparing, plotting, cleaning, and merging microbiome data, epigenetic data, and metadata of SG90 cohort)
- Time interaction tests (evaluating time interaction effects between SG90 metadata and biological data (saliva and methylation data) using 3-step pipeline)
- Alpha diversity (Shannon, Simpson, Chao1, Richness, and Pielou Evenness plus rarefaction code)
- Beta diversity (robust Aitchison's distance, phylogenetic robust Aitchison's distance, and Bray Curtis)
- Conservation scores (experimental, just in case someone would like to use this data later on)
- Differential abundance analysis (using ALDEx2)
- Machine learning (pipeline tuning feature selection/model type/model parameters to test various models (incl scripts to run it on HPC), get baseline comparison models, and evaluate results)
- SPIEC-EASI (code for both microbiome-only networks and microbiome-epigenome networks and their interpretation)
- Cognition model (GLM model on MMSE scores)

There are dedicated README's in every folder with further details about the methods that were used.
Where needed, environment.txt or environment.yml files are found in subfolders as well for convenient later use.

