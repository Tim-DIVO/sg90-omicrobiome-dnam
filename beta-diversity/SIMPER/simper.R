# Load required libraries
library(vegan)  # For SIMPER and rarefaction
library(dplyr)  # For data manipulation

# Load metadata and ASV table
metadata_clean <- read.csv("metadata_rarefaction.csv")  # Metadata (pre-aligned)
asv_df <- read.csv("asv_rarefaction.csv")  # ASV table with row names

# Parameters
set.seed(123)  # For reproducibility
rarefaction_depth <- 20651
num_iterations <- 1000

# Function to rarefy data and perform SIMPER
rarefy_and_simper <- function(asv_table, group_metadata, depth, ...) {

  rarefied_table <- as.data.frame(rrarefy(asv_table, depth))
  
  # Match metadata
  rarefied_metadata <- group_metadata
  
  # Perform SIMPER
  simper_results <- simper(rarefied_table, rarefied_metadata$Group, ...)

  
  return(simper_results)
}

# Main script to calculate average SIMPER results
calculate_average_simper <- function(asv_table, group_metadata, depth, iterations, permutations = 999) {
  print("Entering calculate_average_simper function...")  # Debugging print
  
  simper_list <- vector("list", iterations)  # Store SIMPER results
  
  for (i in 1:iterations) {
    print(paste("Starting iteration:", i))  # Debugging print
    
    simper_list[[i]] <- rarefy_and_simper(asv_table, group_metadata, depth, permutations = permutations)
    
    # Print progress every 50 iterations
    if (i %% 50 == 0) {
      message(paste("Iteration", i, "completed"))
    }
  }
  
  # Extract and combine SIMPER contributions
  #print("Extracting SIMPER results summary...")
  contribution_list <- lapply(simper_list, function(x) summary(x))
  
  combined_contributions <- do.call(rbind, lapply(contribution_list, function(x) {
    do.call(rbind, lapply(x, as.data.frame))
  }))
  
  #print("SIMPER contributions extracted and combined.")
  
  # Add row names as "feature"
  combined_contributions$feature <- rownames(combined_contributions)
  print(paste("Combined contributions dimensions:", dim(combined_contributions)))
  
  # Calculate summary statistics
  averaged_contributions <- combined_contributions %>%
    group_by(feature) %>%
    summarise(
      average_contribution = mean(average, na.rm = TRUE),
      sd_contribution = mean(sd, na.rm = TRUE),
      ava_mean = mean(ava, na.rm = TRUE),
      avb_mean = mean(avb, na.rm = TRUE),
      cumsum_mean = mean(cumsum, na.rm = TRUE),
      p_mean = mean(p, na.rm = TRUE)
    )
  
  print("Summary statistics calculated.")
  return(averaged_contributions)
}

# Run the analysis
print("Starting the main analysis...")
averaged_simper_results <- calculate_average_simper(asv_df, metadata_clean, rarefaction_depth, num_iterations)

# Save results to a CSV file
print("Saving results to file...")
write.csv(averaged_simper_results, "averaged_simper_results.csv", row.names = FALSE)

# Print summarized results
#message("Summarized Results:")
#print(averaged_simper_results)

