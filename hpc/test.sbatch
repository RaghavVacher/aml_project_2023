#!/bin/bash

#SBATCH --job-name=TrainWithIDTest    # Job name
#SBATCH --output=job.%j.out         # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8           # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu:8                  # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=00:05:00             # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown       # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END  # send an email when the job starts and ends (or fails)

#Specify python version to run
module load Python/3.10.4-GCCcore-11.3.0

# Activate your virtual environment if you are using one
source venv/bin/activate

# pip install -r requirements.txt

# Run the Python script
python test.py
