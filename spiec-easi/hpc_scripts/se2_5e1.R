library(SpiecEasi)
library(parallel)
# Set the number of cores
num_cores <- 4  

microbiome <- read.csv("asvs.csv")
cpgs <- read.csv("cpgs_imputed.csv")
cpgs_pca <- read.csv("pca_df_100.csv")

# only retain i.e. asvs that are present in at least 10% of samples
threshold <- nrow(microbiome) * 0.15
# Identify columns (ASVs) where the number of non-zero values is >= threshold
columns_to_keep <- colSums(microbiome != 0) >= threshold
# Filter the dataframe to keep only these columns
microbiome_filtered <- microbiome[, columns_to_keep]

MB_matrix <- as.matrix(microbiome_filtered)
Cpg_matrix <- as.matrix(cpgs)
Cpg_filtered_matrix <- as.matrix(cpgs_pca)

# Run SpiecEasi with multicore support for mb method
se.mb <- spiec.easi(list(MB_matrix, Cpg_filtered_matrix), method='mb', lambda.min.ratio=(2.5)e-1, nlambda=20, pulsar.params=list(rep.num=50, ncores=num_cores))
saveRDS(se.mb, file = "se_2_5e1_mb.rds")    
rm(se.mb)  # Remove object from memory
gc()  # Run garbage collection to free up memory

# Run SpiecEasi with multicore support for glasso method
se.gl <- spiec.easi(list(MB_matrix, Cpg_filtered_matrix), method='glasso', lambda.min.ratio=(2.5)1e-1, nlambda=20, pulsar.params=list(rep.num=50, ncores=num_cores))
saveRDS(se.gl, file = "se_2_5e1_gl.rds")
rm(se.gl)  # Remove object from memory
gc()  # Run garbage collection to free up memory