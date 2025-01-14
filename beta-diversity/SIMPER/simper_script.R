# Load required libraries
library(vegan)  # For SIMPER and rarefaction
library(dplyr)  # For data manipulation

# Load metadata and ASV table
metadata_clean <- read.csv("metadata_rarefaction.csv")  # Metadata (pre-aligned)
asv_df <- read.csv("asv_rarefaction.csv")  # ASV table

# Parameters
set.seed(123)  # For reproducibility
rarefaction_depth <- 20651
num_iterations <- 1000

# Function to rarefy data
rarefy_and_simper <- function(asv_table, group_metadata, depth, ...) {
  # Rarefy the ASV table
  rarefied_table <- as.data.frame(rrarefy(asv_table, depth))
  
  # Remove samples with zero counts after rarefaction
  rarefied_table <- rarefied_table[rowSums(rarefied_table) > 0, ]
  
  # Ensure metadata matches the rarefied samples
  rarefied_metadata <- group_metadata[1:nrow(rarefied_table), , drop = FALSE]
  
  # Run SIMPER on the rarefied table
  simper_results <- simper(rarefied_table, rarefied_metadata$Group, ...)
  return(simper_results)
}

# Main script to calculate average SIMPER results
calculate_average_simper <- function(asv_table, group_metadata, depth, iterations, permutations = 999) {
  simper_list <- vector("list", iterations)  # To store SIMPER results
  
  for (i in 1:iterations) {
    # Perform rarefaction and SIMPER
    simper_list[[i]] <- rarefy_and_simper(asv_table, group_metadata, depth, permutations = permutations)
  }
  
  # Average SIMPER results across iterations
  # Extract contributions for each ASV
  contribution_list <- lapply(simper_list, function(x) summary(x))
  
  # Combine contributions into a single data frame
  combined_contributions <- do.call(rbind, lapply(contribution_list, function(x) {
    do.call(rbind, lapply(x, as.data.frame))
  }))
  
  # Group by taxa (assuming "feature" is the ASV ID) and calculate the mean contribution
  averaged_contributions <- combined_contributions %>%
    group_by(feature) %>%
    summarise(
      average_contribution = mean(average, na.rm = TRUE),
      sd_contribution = sd(average, na.rm = TRUE),
      ratio_contribution = mean(ratio, na.rm = TRUE)
    )
  
  return(averaged_contributions)
}

# Run the analysis
averaged_simper_results <- calculate_average_simper(asv_df, metadata_clean, rarefaction_depth, num_iterations)

# Save results to a CSV file
write.csv(averaged_simper_results, "averaged_simper_results.csv", row.names = FALSE)

