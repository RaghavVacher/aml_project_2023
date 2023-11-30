    How to HPC

## Create connection

in your terminal type: ssh itu_user@hpc.itu.dk, then you will be asked to give your password.
Connecting for the first time takes a while.


## Loading data onto HPC

- to copy an entire directory to the hpc, use: scp -r <testfolder> itu_user@hpc.itu.dk:~/<testfolder> 
- this requires "scp" which according to ITU is available by default on mac and linux, but might need to be installe don windows


## Submitting a job

- Create py script to train the model
- create an sbatch file using the template
- upload the files to the hpc using: scp <path_to_sbatch_file.sbatch> itu_user@hpc.itu.dk:~/example.txt
- submit by using this command: sbatch <filename.sbatch> 
- monitor using this command: squeue -u your_username
