#!/bin/bash

# Command to submit sbatch job with passed arguments
# $1 = number of cpus per task
# $2 = number of gpus per task
# $3 = time limit for job (hh:mm:ss)

# Default value assignment
CPU=${1:-4}
GPU=${2:-0} # Default to X GPUs if not provided
TIME=${3:-00:05:00} # Default to 5 minutes if not provided
EPOCHS=${4:-10} # Default to X epochs if not provided
MODEL=${5:-resnet18} # Default to X model if not provided

echo "Number of CPUs: $1"
echo "Number of GPUs: $2"
echo "Time limit: $3"
echo "Number of epochs: $4"
echo "Model used: $5"

# sbatch --cpus-per-task=$1 --gres=gpu:$2 --time=$3  batch.sbatch $4 $5

# Submit the job and capture the job ID
JOB_ID=$(sbatch --cpus-per-task=$1 --gres=gpu:$2 --time=$3  batch.sbatch $4 $5 | awk '{print $4}')

# Print a banner
echo "===================================="
echo " Waiting for job $JOB_ID to start "
echo "===================================="

# Wait for the output file to be created
while [ ! -f job.$JOB_ID.out ]; do
  sleep 1
done

# Now, tail the file
tail -f job.$JOB_ID.out