library(parallel)
library(vegan)
library(dplyr)

metadata_clean <- read.csv("metadata_rarefaction.csv")
asv_df <- read.csv("asv_rarefaction.csv")

set.seed(123)
rarefaction_depth <- 20651
num_iterations <- 1000
permutations <- 999

rarefy_and_simper <- function(asv_table, group_metadata, depth, permutations) {
  rarefied_table <- as.data.frame(rrarefy(asv_table, depth))
  rarefied_table <- rarefied_table[rowSums(rarefied_table) > 0, ]
  
  rarefied_metadata <- group_metadata[1:nrow(rarefied_table), , drop=FALSE]
  
  simper_results <- simper(rarefied_table, rarefied_metadata$Group, permutations = permutations)
  summary(simper_results)
}

calculate_average_simper <- function(asv_table, group_metadata, depth, iterations, permutations) {
  # Just use the number of cores on this single node
  ncores <- 5#as.numeric(Sys.getenv("OMP_NUM_THREADS", "20"))
  
  # If OMP_NUM_THREADS isn't set, fallback to np from PBS_NODEFILE count
  if (is.na(ncores) || ncores == 0) {
    nodefile <- Sys.getenv("PBS_NODEFILE")
    if (nodefile != "") {
      ncores <- length(readLines(nodefile))
    } else {
      ncores <- detectCores()
    }
  }
  
  cl <- makeCluster(ncores, type = "PSOCK")
  
  clusterEvalQ(cl, {
    library(vegan)
    library(dplyr)
  })
  
  clusterExport(cl, c("asv_table", "group_metadata", "depth", "permutations", "rarefy_and_simper"))
  
  simper_list <- parLapply(cl, 1:iterations, function(i) {
    rarefy_and_simper(asv_table, group_metadata, depth, permutations)
  })
  
  stopCluster(cl)
  
  combined_contributions <- do.call(rbind, lapply(simper_list, function(x) {
    do.call(rbind, lapply(x, as.data.frame))
  }))
  
  averaged_contributions <- combined_contributions %>%
    group_by(feature) %>%
    summarise(
      average_contribution = mean(average, na.rm = TRUE),
      sd_contribution = sd(average, na.rm = TRUE),
      ratio_contribution = mean(ratio, na.rm = TRUE)
    )
  
  averaged_contributions
}

averaged_simper_results <- calculate_average_simper(asv_df, metadata_clean, rarefaction_depth, num_iterations, permutations)
write.csv(averaged_simper_results, "averaged_simper_results.csv", row.names = FALSE)
