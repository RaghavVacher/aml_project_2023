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
LR=${6:-0.0001} # Default to X learning rate if not provided
SAMPLES=${7:-all} # Default to X samples per subject if not provided
MEMORY=${8:-10G} # Default to X samples per subject if not provided

echo "===================================="
echo "Number of CPUs: $CPU"
echo "Number of GPUs: $GPU"
echo "Time limit: $TIME"
echo "Number of epochs: $EPOCHS"
echo "Model used: $MODEL"
echo "Learning rate: $LR"
echo "Samples per subject: $SAMPLES"
echo "Memory per CPU: $MEMORY"

# sbatch --cpus-per-task=$1 --gres=gpu:$2 --time=$3  batch.sbatch $4 $5

# Submit the job and capture the job ID
JOB_ID=$(sbatch --cpus-per-task=$CPU --gres=gpu:$GPU --time=$TIME --mem-per-cpu=$MEMORY batch.sbatch $EPOCHS $MODEL $LR $SAMPLES | awk '{print $4}')

# Print a banner
echo "===================================="
echo " Waiting for job $JOB_ID to start "
echo "===================================="

# Wait for the output file to be created
while [ ! -f job.$JOB_ID.out ]; do
  sleep 1
done

# While the output file is empty, sleep
while [ ! -s job.$JOB_ID.out ]; do
    # clear last line
echo "===================================="
    echo " Job $JOB_ID started | $(date -d@$SECONDS -u +%H:%M:%S)/$TIME elapsed"
    echo "===================================="
    sleep 1
  # Move the cursor up 3 lines and clear each line
    for i in {1..3}; do
        tput cuu1
        tput el
    done
done

echo "===================================="
echo " Job $JOB_ID started | $(date -d@$SECONDS -u +%H:%M:%S)/$TIME elapsed"
echo "===================================="

# Now, tail the file
tail -f job.$JOB_ID.out | while read LOGLINE
do
    echo "${LOGLINE}"
    if [[ "${LOGLINE}" == *"Model and training history saved"* ]]; then
            echo "===================================="
            echo " Job $JOB_ID finished | $(date -d@$SECONDS -u +%H:%M:%S)/$TIME elapsed"
            echo "===================================="
        pkill -P $$ tail
    fi
done