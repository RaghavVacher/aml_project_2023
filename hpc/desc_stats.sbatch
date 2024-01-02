#!/bin/bash

#SBATCH --job-name=desc_stats   # Job name
#SBATCH --output=job.%j.out         # Name of output file (%j expands to jobId)
#SBATCH --partition=brown       # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END  # send an email when the job starts and ends (or fails)
#SBATCH --mem=250G
#SBATCH --time=12:00:00


#Specify python version to run
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if you are using one
source venv/bin/activate

# pip install -r requirements.txt

# Run the Python script
python desc_stats.py
