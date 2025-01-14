# List of R script names extracted from the image manually
r_scripts = [
    "se1e1.R",
    "se1e2.R",
    "se1e3.R",
    "se2_5e1.R",
    "se5e1.R",
    "se5e2.R",
    "se1e1_full.R",
    "se1e2_full.R",
    "se1e3_full.R",
    "se2_5e1_full.R",
    "se5e1_full.R",
    "se2_5e2_full.R",
    
]

# Template for the job submission script
job_template = """#!/bin/bash
#PBS -P {name}  
#PBS -q parallel24                 
#PBS -l select=1:ncpus=24:mpiprocs=24:mem=160GB
#PBS -l walltime=480:00:00          
#PBS -j oe                         
#PBS -N {name}   

# Move to the working directory
cd ${{PBS_O_WORKDIR}}

# Log the nodefile contents and check np
echo "Nodefile content:"
cat ${{PBS_NODEFILE}}
np=$(cat ${{PBS_NODEFILE}} | wc -l)
echo "Number of processors (np): $np"

# Load the environment for R 4.2.1
source /app1/ebenv R-4.2.1
if [ $? -ne 0 ]; then
  echo "Error loading R environment"
  exit 1
fi

# Check if mpirun is available
which mpirun
if [ $? -ne 0 ]; then
  echo "Error: mpirun command not found"
  exit 1
fi

# Run the R script with mpirun using the nodefile
mpirun -np ${{np}} --hostfile ${{PBS_NODEFILE}} Rscript {script_name}
if [ $? -ne 0 ]; then
  echo "Error during mpirun execution"
  exit 1
fi

exit 0
"""

# Generate job submission scripts for each R script
job_files = {}
for script in r_scripts:
    name = script.replace(".R", "")  # Remove the .R extension for the PBS name
    job_script_content = job_template.format(name=name, script_name=script)
    job_files[script] = job_script_content

# Save all job scripts to individual text files

import os

for script, content in job_files.items():
    file_name = f"{script.replace('.R', '')}_job.txt"
    with open(file_name, "w") as f:
        f.write(content)
