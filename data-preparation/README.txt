This is the data preparation needed to obtain many of the dataframes that are being used in all other folders. Generally, 3 things are being done:

1. Preparing ASV data and epigenetic clock data (median calculation and mAge deviation calculation)
2. Merging SG90 metadata + microbiome + mAge data
3. Cleaning and plotting data. Obtaining distributions for Table 1.

The numbers in filenames annotate the sequence they were used in.

Important: Many of these scripts have last been used for i.e. genus-level data merging, however they generally work with any level of microbial classification, you only need to switch out the input file names. 

Also, many different versions have been used over the months, you can find these annotated by "other_...". They might be useful when running other analyses.
