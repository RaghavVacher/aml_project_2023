#template

#!/bin/bash

#SBATCH --job-name=model_training    # Job name
#SBATCH --output=job.%j.out         # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8           # Schedule 8 cores (includes hyperthreading)
#SBATCH --gres=gpu                  # Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=01:00:00             # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red,brown       # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END  # send an email when the job starts and ends (or fails)


# Activate your virtual environment if you are using one
source /Users/emilykruger/Documents/GitHub/aml_project_2023/aml_project/bin/activate

# Run the Python script
python test.py