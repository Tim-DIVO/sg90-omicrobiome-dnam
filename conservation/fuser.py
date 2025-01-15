import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Load data
scores = pd.read_csv("conservation_scores.csv")
cpgs = pd.read_csv("methylation_coordinates.csv")

# Ensure the columns are cast to integers
cpgs['Start_hg38'] = cpgs['Start_hg38'].astype(int)
cpgs['End_hg38'] = cpgs['End_hg38'].astype(int)

# Define the function to process each chunk of scores
def process_chunk(score_chunk):
    result = pd.DataFrame()
    
    for _, score_row in score_chunk.iterrows():
        # Find matching rows in cpgs based on the specified conditions
        matching_rows = cpgs[
            (cpgs['CHR_hg38'] == score_row['chromosome']) &
            (cpgs['Start_hg38'] <= score_row['position']) &
            (cpgs['End_hg38'] >= score_row['position'])
        ]
        
        if not matching_rows.empty:
            matching_rows = matching_rows.copy()
            matching_rows['position'] = score_row['position']
            matching_rows['chromosome'] = score_row['chromosome']
            matching_rows['score'] = score_row['score']
            
            # Define the column order
            ordered_columns = ['IlmnID', 'Name', 'chromosome', 'position', 'score'] + \
                              list(matching_rows.columns.difference(['IlmnID', 'Name', 'chromosome', 'position', 'score']))
            
            # Append matching rows to the result DataFrame with the specified column order
            result = pd.concat([result, matching_rows[ordered_columns]])

    return result

# Main function to set up multiprocessing
if __name__ == '__main__':
    # Number of cores to use
    num_cores = cpu_count()

    # Split scores into chunks for each process
    scores_chunks = np.array_split(scores, num_cores)

    # Create a Pool and map the chunks to the process_chunk function
    with Pool(num_cores) as pool:
        results = pool.map(process_chunk, scores_chunks)
    
    # Combine all results into a single DataFrame
    filtered_cpgs = pd.concat(results).reset_index(drop=True)

    # Save to CSV
    filtered_cpgs.to_csv("filtered_cpgs.csv", index=False)
