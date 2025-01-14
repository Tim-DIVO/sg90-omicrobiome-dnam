library(SpiecEasi)
library(parallel)

# Set the number of cores
num_cores <- 4  # Adjust as needed based on your server capacity

# Species level with both methods
df <- read.csv("ML_Data/FINAL_SPECIES_RAW.csv")
subset_df <- df[, 16:ncol(df)]
X <- as.matrix(subset_df)

# Run SpiecEasi with multicore support for mb method
se_species.mb <- spiec.easi(X, method='mb', lambda.min.ratio=1e-3, nlambda=20, 
                            pulsar.params=list(rep.num=50, ncores=num_cores))
saveRDS(se_species.mb, file = "se_species_mb.rds")    
rm(se_species.mb)  # Remove object from memory
gc()  # Run garbage collection to free up memory

# Run SpiecEasi with multicore support for glasso method
se_species.gl <- spiec.easi(X, method='glasso', lambda.min.ratio=1e-3, nlambda=20, 
                            pulsar.params=list(rep.num=50, ncores=num_cores))
saveRDS(se_species.gl, file = "se_species_gl.rds")
rm(se_species.gl)  # Remove object from memory
gc()  # Run garbage collection to free up memory

# Genus level with both methods
df <- read.csv("ML_Data/FINAL_GENUS_TAXA_RAW.csv")
selection <- colnames(df)[2:(ncol(df) - 13)]
X <- as.matrix(df[, selection])

# Run SpiecEasi with multicore support for mb method
se_genus.mb <- spiec.easi(X, method='mb', lambda.min.ratio=1e-3, nlambda=20, 
                          pulsar.params=list(rep.num=50, ncores=num_cores))
saveRDS(se_genus.mb, file = "se_genus_mb.rds")
rm(se_genus.mb)  # Remove object from memory
gc()  # Run garbage collection to free up memory

# Run SpiecEasi with multicore support for glasso method
se_genus.gl <- spiec.easi(X, method='glasso', lambda.min.ratio=1e-3, nlambda=20, 
                          pulsar.params=list(rep.num=50, ncores=num_cores))
saveRDS(se_genus.gl, file = "se_genus_gl.rds")
rm(se_genus.gl)  # Remove object from memory
gc()  # Run garbage collection to free up memory

# Final cleanup
rm(df, subset_df, X)
gc()  # Ensure all memory is freed






