#!/bin/bash

#SBATCH --job-name=TrainWithID    # Job name
#SBATCH --output=job.%j.out         # Name of output file (%j expands to jobId)
#SBATCH --partition=brown       # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END  # send an email when the job starts and ends (or fails)

#Specify python version to run
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if you are using one
source venv/bin/activate

# pip install -r requirements.txt

# Run the Python script
# $1 - epochs
# $2 - model
python train_ID_test.py $1 $2 $3 $4 $5 $6 $7 $8
